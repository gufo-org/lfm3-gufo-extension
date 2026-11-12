"""One LFM3 decode token on the whole NPU (8 columns x 4 rows), Q8_0 weights.

The cores run a phase interpreter (decode.cc, dec_next): 109 phases per token
(embedding, 13 convolution layers x 4, 11 attention layers x 5, LM head).
Rows 2-3 (j = 0, 1) also attend (one head each); rows 4-5 (j = 2, 3) also
convolve (128 channels each). The runtime sequence feeds every phase's inputs
at the head of the weight streams and drains its outputs into the state
buffer S; `pos` and the per-layer attention spans are dispatch-time values.

S holds, per layer, a convolution or attention input block and an FFN input
block, then the activations, the convolution/attention vectors, the
convolution states, the RoPE tables, the KV caches and the logits; layout.py
computes it for the host.
"""
import argparse
import os

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, CompileTime, DispatchTime, ExternalFunction, In, InOut, Kernel, ObjectFifo, Program, Runtime,
                      TaskGroup, Worker, WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name

import layout as L

COLS, ROWS, REC, OUT = 8, 4, L.REC, 32
HERE = os.path.dirname(os.path.abspath(__file__))


def tap(total, offset, sizes, strides):
    return TensorAccessPattern((total,), offset, sizes, strides)


@iron.jit
def token(S: InOut, W: In, *, tok: DispatchTime[np.int32], pos: DispatchTime[np.int32],
          swa_first: DispatchTime[np.int32], swa_cached: DispatchTime[np.int32],
          global_recs: DispatchTime[np.int32], swa_recs: DispatchTime[np.int32],
          global_a: DispatchTime[np.int32], global_b: DispatchTime[np.int32], ctx: CompileTime[int] = 4096):
    lay = L.Layout(ctx)
    in_ty = np.ndarray[(L.BLOCK,), np.dtype[np.int8]]
    rec_ty = np.ndarray[(REC,), np.dtype[np.int8]]
    rec2_ty = np.ndarray[(2 * REC,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT,), np.dtype[np.int8]]
    out4_ty = np.ndarray[(ROWS * OUT,), np.dtype[np.int8]]
    q8_ty = np.ndarray[(8192,), np.dtype[np.int8]]
    scratch_ty = np.ndarray[(512,), np.dtype[np.float32]]
    cnt_ty = np.ndarray[(8,), np.dtype[np.int32]]
    rtp_ty = np.ndarray[(2 * L.N_ATTN,), np.dtype[np.int32]]
    src = os.path.join(HERE, "decode.cc")
    flags = ["-Os"] + os.environ.get("DECODE_FLAGS", "").split()
    kernels = {}
    for role, extra in (("conv", []), ("attn", ["-DDECODE_ATTN_ROLE"])):
        obj = f"decode_{role}.o"
        k = {"next": ExternalFunction(f"{role}_next", source_file=src, arg_types=[cnt_ty, rtp_ty, np.int32, np.int32],
                                      object_file_name=obj, compile_flags=flags + extra)}
        for name, types in {"copy": [rec_ty, in_ty], "own": [rec_ty], "begin": [in_ty, q8_ty, scratch_ty],
                            "pre_out": [in_ty, out_ty], "rec": [q8_ty, rec_ty], "flush": [out_ty, cnt_ty],
                            "finish": [], "post_out": [out_ty]}.items():
            k[name] = Kernel(f"{role}_{name}", obj, types)
        kernels[role] = k

    w_l2l1, y_l1l2, w_fifos, y_fifos = [], [], [], []
    for c in range(COLS):
        col = []
        for half in range(2):
            # A stream broadcasts to both of its cores: shared inputs and KV cross DDR once;
            # each core keeps its own records of a weight stream (alternating) and skips the rest.
            # deep memtile buffering: the stream keeps running through a phase's prologue
            f = ObjectFifo(rec_ty, name=f"wl3_{c}_{half}", depth=int(os.environ.get("DECODE_L2_DEPTH", "16")))
            w_fifos.append(f)
            fwd = f.cons().forward(obj_type=rec_ty, name=f"w{c}_{half}", depth=2)  # L1 depth
            col += [fwd, fwd]
        w_l2l1.append(col)
        o = ObjectFifo(out4_ty, name=f"y{c}", depth=2)
        y_fifos.append(o)
        y_l1l2.append(o.prod().join([OUT * j for j in range(ROWS)], obj_types=[out_ty] * ROWS,
                                    names=[f"y{c}_{j}" for j in range(ROWS)], depths=[2] * ROWS))

    rtps = [[Buffer(rtp_ty, name=f"rtp_{c}_{j}", use_write_rtp=True) for j in range(ROWS)] for c in range(COLS)]
    barriers = [[WorkerRuntimeBarrier() for _ in range(ROWS)] for _ in range(COLS)]

    def body(w_in, y_out, x, q8, scratch, cnt, rtp, barrier, c, j, nxt, copy, own, begin, pre_out, rec, flush,
             finish, post_out):
        barrier.wait_for_value(1)
        barrier.release_with_value(1)
        for _ in range_(L.PHASES):
            nxt(cnt, rtp, c, j)
            for _ in range_(cnt[0]):
                w = w_in.acquire(1)
                copy(w, x)
                w_in.release(1)
            for _ in range_(cnt[1]):
                w = w_in.acquire(1)
                own(w)
                w_in.release(1)
            begin(x, q8, scratch)
            for _ in range_(cnt[2]):
                o = y_out.acquire(1)
                pre_out(x, o)
                y_out.release(1)
            for _ in range_(cnt[3]):
                o = y_out.acquire(1)
                for _ in range_(cnt[4]):
                    w = w_in.acquire(1)
                    rec(q8, w)
                    w_in.release(1)
                flush(o, cnt)  # may set the next inner count (down projection)
                y_out.release(1)
            finish()
            for _ in range_(cnt[5]):
                o = y_out.acquire(1)
                post_out(o)
                y_out.release(1)

    workers = []
    for c in range(COLS):
        for j in range(ROWS):
            k = kernels["attn" if j < 2 else "conv"]
            bufs = [Buffer(in_ty, name=f"x_{c}_{j}"), Buffer(q8_ty, name=f"q8_{c}_{j}"),
                    Buffer(scratch_ty, name=f"scratch_{c}_{j}"), Buffer(cnt_ty, name=f"cnt_{c}_{j}")]
            workers.append(Worker(body, [w_l2l1[c][j].cons(), y_l1l2[c][j].prod(), *bufs, rtps[c][j],
                                         barriers[c][j], c, j, *k.values()],
                                  tile=Tile(c, 2 + j), stack_size=0x800, dynamic_objfifo_lowering=True))

    s_ty = np.ndarray[(lay.s_bytes,), np.dtype[np.int8]]
    w_ty = np.ndarray[(lay.w_bytes,), np.dtype[np.int8]]
    w_prods = [f.prod() for f in w_fifos]
    y_conses = [f.cons() for f in y_fifos]

    def sequence(S, W, tok, pos, swa_first, swa_cached, global_recs, swa_recs, global_a, global_b, wh, yh):
        # A DMA task repeats at most 64 times, so the global KV stream is two tasks of
        # global_a + global_b = global_recs records (both at least 1).
        d = dict(tok=tok, pos=pos, swa_first=swa_first, swa_cached=swa_cached, global_recs=global_recs,
                 swa_recs=swa_recs, global_a=global_a, global_b=global_b)
        for c in range(COLS):
            for j in range(ROWS):
                for i, l in enumerate(L.ATTN_LAYERS):
                    swa = l in L.SWA_LAYERS
                    rtps[c][j][2 * i] = swa_cached if swa else pos
                    rtps[c][j][2 * i + 1] = swa_recs if swa else global_recs
                barriers[c][j].set(1)
        for phase in lay.phases(d):
            # One task group per phase when every shim tile holds at most 8 tasks: the FIFO
            # already orders a stream's inputs before its weights, so only the drains are
            # waited on. Otherwise the inputs go first in a group of their own.
            per_col = [sum(len(phase.heads(s)) + len(phase.stream(s)) for s in (2 * c, 2 * c + 1)) + len(phase.drains(c))
                       for c in range(COLS)]
            tg = TaskGroup()
            for s in range(2 * COLS):
                for (buf, t) in phase.heads(s):
                    wh[s].fill(S if buf == "S" else W, t, wait=max(per_col) > 8, group=tg)
            if max(per_col) > 8:
                tg.finish()
                tg = TaskGroup()
            for s in range(2 * COLS):
                for (buf, t) in phase.stream(s):
                    wh[s].fill(S if buf == "S" else W, t, group=tg)
            for c in range(COLS):
                for t in phase.drains(c):
                    yh[c].drain(S, t, wait=True, group=tg)
            tg.finish()

    rt = Runtime(sequence, [s_ty, w_ty, tok, pos, swa_first, swa_cached, global_recs, swa_recs, global_a, global_b,
                            w_prods, y_conses])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xclbin-path", required=True)
    a = p.parse_args()
    p2 = token.specialize(ctx=4096)
    p2.compile(xclbin_path=a.xclbin_path)
    print("dispatch library:", p2.get_dispatch_lib_path())
