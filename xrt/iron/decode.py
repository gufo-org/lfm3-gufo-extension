"""LFM3 decode on the whole NPU (8 columns x 4 rows), Q8_0 weights.

This version runs one convolution layer: C1 (residual, operator norm, B/C/x
projection and the causal convolution), C2 (out projection), F1 (residual,
FFN norm, gate/up and SwiGLU), F2 (down projection).

All phase inputs and outputs live in one state buffer S. A phase input
block is [h f32 | o f32 | post-norm w bf16 | pre-norm w bf16]: the phase
adds the post-normed o to h, writes the new h into the next block's h slot
and normalizes it for its projection. Drains write the block outputs into
the next block's o slot.
"""
import argparse
import os

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, ExternalFunction, In, InOut, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name

COLS, ROWS, H, I = 8, 4, 2048, 7680
REC = 4 * H + 4 * (H // 32) * 2  # 8704
BLOCK = 12 * H                   # phase input block, 24576 bytes
OUT = 32                         # output object bytes
HERE = os.path.dirname(os.path.abspath(__file__))

# Per-core record counts of each phase.
X_RECS = -(-BLOCK // REC)        # 3: a phase input block
ACT_RECS = -(-(2 * I) // REC)    # 2: the SwiGLU activations
VEC_RECS = 1                     # a bf16 H vector
CONV_RECS, OUT_RECS, GU_RECS, DOWN_RECS = 64, 16, 120, 64

# State buffer layout (bytes).
S_A = 0                   # convolution block input
S_B = S_A + BLOCK         # FFN block input
S_NEXT = S_B + BLOCK      # next layer's block input
S_ACT = S_NEXT + BLOCK    # activations, padded to ACT_RECS records
S_VEC = S_ACT + ACT_RECS * REC  # convolution output, padded to one record
S_STATE = S_VEC + REC     # convolution state: per column [chunk 48][core 4][32 bytes]
STATE_COL = 48 * ROWS * 32
S_BYTES = S_STATE + COLS * STATE_COL + REC * 4  # reads of the last column run past it

# Weight buffer layout: per phase, per stream (2 per column), records.
W_PHASES = [("conv", CONV_RECS), ("out", OUT_RECS), ("gu", GU_RECS), ("down", DOWN_RECS)]
W_OFFSET, _o = {}, 0
for name, n in W_PHASES:
    W_OFFSET[name] = _o
    _o += COLS * 2 * 2 * n * REC
W_BYTES = _o


def stream_tap(total, offset, records):
    """`records` whole records, one buffer descriptor."""
    return TensorAccessPattern((total,), offset, [1, records, REC // 512, 512], [0, REC, 512, 1])


def head_tap(total, offset, records):
    """`records` records of the same bytes for both of a stream's cores."""
    return TensorAccessPattern((total,), offset, [records, 2, REC // 512, 512], [REC, 0, 512, 1])


def bytes_tap(total, offset, length):
    assert length % 32 == 0 and length // 32 < 1024
    return TensorAccessPattern((total,), offset, [1, 1, length // 32, 32], [0, 0, 32, 1])


@iron.jit
def decode(S: InOut, W: In):
    in_ty = np.ndarray[(BLOCK,), np.dtype[np.int8]]
    rec_ty = np.ndarray[(REC,), np.dtype[np.int8]]
    rec2_ty = np.ndarray[(2 * REC,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT,), np.dtype[np.int8]]
    out4_ty = np.ndarray[(ROWS * OUT,), np.dtype[np.int8]]
    q8_ty = np.ndarray[(8192,), np.dtype[np.int8]]
    scratch_ty = np.ndarray[(512,), np.dtype[np.float32]]
    src = os.path.join(HERE, "decode.cc")
    flags = os.environ.get("DECODE_FLAGS", "").split()
    k = {"copy": ExternalFunction("dec_copy", source_file=src, arg_types=[rec_ty, in_ty],
                                  object_file_name="decode.o", compile_flags=flags)}
    for name, types in {
        "pre": [in_ty, q8_ty, scratch_ty], "slice": [in_ty, out_ty, np.int32, np.int32],
        "quant": [in_ty, q8_ty, scratch_ty, np.int32],
        "gate_up": [q8_ty, rec_ty], "gate_flush": [out_ty],
        "row": [q8_ty, rec_ty, np.int32], "rows4": [q8_ty, rec_ty], "rows_flush": [out_ty],
        "conv_load": [rec_ty], "conv": [q8_ty, rec_ty], "conv_flush": [out_ty],
        "conv_store": [out_ty],
    }.items():
        k[name] = Kernel("dec_" + name, "decode.o", types)

    w_l2l1, y_l1l2, w_fifos, y_fifos = [], [], [], []
    for c in range(COLS):
        col = []
        for half in range(2):
            f = ObjectFifo(rec2_ty, name=f"wl3_{c}_{half}", depth=2)
            w_fifos.append(f)
            col += f.cons().split([0, REC], obj_types=[rec_ty] * 2, names=[f"w{c}_{2 * half + i}" for i in range(2)])
        w_l2l1.append(col)
        o = ObjectFifo(out4_ty, name=f"y{c}", depth=2)
        y_fifos.append(o)
        y_l1l2.append(o.prod().join([OUT * j for j in range(ROWS)], obj_types=[out_ty] * ROWS,
                                    names=[f"y{c}_{j}" for j in range(ROWS)], depths=[2] * ROWS))

    def body(w_in, y_out, x, q8, scratch, c, j, k):
        def take(records):
            for _ in range_(records):
                w = w_in.acquire(1)
                k["copy"](w, x)
                w_in.release(1)

        def pre():
            take(X_RECS)
            k["pre"](x, q8, scratch)
            for _ in range_(8):
                o = y_out.acquire(1)
                k["slice"](x, o, c, j)
                y_out.release(1)

        # C1: convolution.
        pre()
        w = w_in.acquire(1)
        k["conv_load"](w)
        w_in.release(1)
        for _ in range_(CONV_RECS // 16):
            o = y_out.acquire(1)
            for _ in range_(16):
                w = w_in.acquire(1)
                k["conv"](q8, w)
                w_in.release(1)
            k["conv_flush"](o)
            y_out.release(1)
        for _ in range_(48):
            o = y_out.acquire(1)
            k["conv_store"](o)
            y_out.release(1)
        # C2: out projection.
        take(VEC_RECS)
        k["quant"](x, q8, scratch, H)
        for _ in range_(OUT_RECS // 2):
            o = y_out.acquire(1)
            for _ in range_(2):
                w = w_in.acquire(1)
                k["rows4"](q8, w)
                w_in.release(1)
            k["rows_flush"](o)
            y_out.release(1)
        # F1: gate/up.
        pre()
        for _ in range_(GU_RECS // 8):
            o = y_out.acquire(1)
            for _ in range_(8):
                w = w_in.acquire(1)
                k["gate_up"](q8, w)
                w_in.release(1)
            k["gate_flush"](o)
            y_out.release(1)
        # F2: down.
        take(ACT_RECS)
        k["quant"](x, q8, scratch, I)
        for _ in range_(DOWN_RECS // 8):
            o = y_out.acquire(1)
            for _ in range_(8):
                w = w_in.acquire(1)
                k["row"](q8, w, I)
                w_in.release(1)
            k["rows_flush"](o)
            y_out.release(1)

    workers = []
    for c in range(COLS):
        for j in range(ROWS):
            x = Buffer(in_ty, name=f"x_{c}_{j}")
            q8 = Buffer(q8_ty, name=f"q8_{c}_{j}")
            scratch = Buffer(scratch_ty, name=f"scratch_{c}_{j}")
            workers.append(Worker(lambda *a, c=c, j=j: body(*a[:5], c, j, dict(zip(k.keys(), a[5:]))),
                                  [w_l2l1[c][j].cons(), y_l1l2[c][j].prod(), x, q8, scratch, *k.values()],
                                  tile=Tile(c, 2 + j), stack_size=0xA00, dynamic_objfifo_lowering=True))

    s_ty = np.ndarray[(S_BYTES,), np.dtype[np.int8]]
    w_ty = np.ndarray[(W_BYTES,), np.dtype[np.int8]]
    w_prods = [f.prod() for f in w_fifos]
    y_conses = [f.cons() for f in y_fifos]

    def sequence(S, W, wh, yh):
        def phase(name, records, heads, drains):
            # A shim tile holds 8 tasks at once (two buffer descriptors each),
            # so the phase input finishes before the weights and drains start;
            # the cores read it first anyway.
            tg = TaskGroup()
            for c in range(COLS):
                for half in range(2):
                    s = 2 * c + half
                    for (offset, n, per_core) in heads:
                        if per_core:  # each of this stream's two cores' own state, padded to a record
                            t = TensorAccessPattern((S_BYTES,), offset + c * STATE_COL + 2 * half * 32,
                                                    [1, 2, REC // 32, 32], [0, 32, ROWS * 32, 1])
                            wh[s].fill(S, t, wait=True, group=tg)
                        else:
                            wh[s].fill(S, head_tap(S_BYTES, offset, n), wait=True, group=tg)
            tg.finish()
            tg = TaskGroup()
            for c in range(COLS):
                for half in range(2):
                    s = 2 * c + half
                    per_stream = 2 * records * REC
                    # No wait: the phase's last drains follow every record's use.
                    wh[s].fill(W, stream_tap(W_BYTES, W_OFFSET[name] + s * per_stream, 2 * records), group=tg)
            for c in range(COLS):
                for tap_of in drains:
                    yh[c].drain(S, tap_of(c), wait=True, group=tg)
            tg.finish()

        # Hidden-row chunks of column c: 8 steps x 4 cores x 8 floats, natural order.
        def hidden(slot):
            return lambda c: bytes_tap(S_BYTES, slot + c * 1024, 1024)

        state_out = lambda c: TensorAccessPattern((S_BYTES,), S_STATE + c * STATE_COL, [1, 1, STATE_COL // 64, 64], [0, 0, 64, 1])
        only = os.environ.get("DECODE_PHASES", "conv,out,gu,down").split(",")
        if "conv" in only: phase("conv", CONV_RECS, [(S_A, X_RECS, False)] + ([(S_STATE, 1, True)] if not os.environ.get("NOSTATE") else []),
              [hidden(S_B), lambda c: bytes_tap(S_BYTES, S_VEC + c * 512, 512)] + ([state_out] if not os.environ.get("NOSTATE") else []))
        if "out" in only: phase("out", OUT_RECS, [(S_VEC, VEC_RECS, False)], [hidden(S_B + 4 * H)])
        if "gu" in only: phase("gu", GU_RECS, [(S_B, X_RECS, False)],
              [hidden(S_NEXT), lambda c: bytes_tap(S_BYTES, S_ACT + c * 1920, 1920)])
        if "down" in only: phase("down", DOWN_RECS, [(S_ACT, ACT_RECS, False)], [hidden(S_NEXT + 4 * H)])

    rt = Runtime(sequence, [s_ty, w_ty, w_prods, y_conses])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xclbin-path", required=True)
    p.add_argument("--insts-path", required=True)
    a = p.parse_args()
    decode.specialize().compile(xclbin_path=a.xclbin_path, inst_path=a.insts_path)
