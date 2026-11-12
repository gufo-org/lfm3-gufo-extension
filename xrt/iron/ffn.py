"""One LFM3 decode FFN block on the whole NPU (8 columns x 4 rows).

F1: every core reads [h | o | post-norm w | ffn-norm w], computes
h += rmsnorm(o) * w_post, writes its 64 rows of h, quantizes rmsnorm(h) * w_ffn
and runs its 240 gate/up channels (two per weight record), writing
SiLU(gate) * up as bf16. F2: every core reads the 7680 activations, quantizes
them and runs its 64 down rows. One broadcast FIFO, one weight-record FIFO
and one output FIFO serve both phases.
"""
import argparse
import os

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, ExternalFunction, In, Kernel, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name

COLS, ROWS, H, I = 8, 4, 2048, 7680
REC = 4 * H + 4 * (H // 32) * 2          # gate/up record; a down record (I + I / 32 * 2) pads to it
IN_BYTES = 4 * H + 4 * H + 2 * H + 2 * H  # 24576
OUT_BYTES = 32
GU_RECS, DOWN_RECS = 120, 64
X1_RECS, X2_RECS = -(-IN_BYTES // REC), -(-(2 * I) // REC)  # 3 and 2
HERE = os.path.dirname(os.path.abspath(__file__))
# State buffer layout (bytes).
S_IN, S_X1, S_ACT, S_O = 0, IN_BYTES, IN_BYTES + 4 * H, IN_BYTES + 4 * H + IN_BYTES
S_BYTES = S_O + 4 * H
W_GU = COLS * GU_RECS * ROWS * REC
W_BYTES = W_GU + COLS * DOWN_RECS * ROWS * REC


def tap(total, offset, length):
    # Every dimension under 1024 elements, so one buffer descriptor moves it.
    if length % REC == 0:
        return TensorAccessPattern((total,), offset, [1, length // REC, REC // 512, 512], [0, REC, 512, 1])
    assert length % 64 == 0 and length // 64 < 1024
    return TensorAccessPattern((total,), offset, [1, 1, length // 64, 64], [0, 0, 64, 1])


@iron.jit
def ffn(S: In, W: In, Y: Out):
    in_ty = np.ndarray[(IN_BYTES,), np.dtype[np.int8]]
    rec_ty = np.ndarray[(REC,), np.dtype[np.int8]]
    rec2_ty = np.ndarray[(2 * REC,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_BYTES,), np.dtype[np.int8]]
    out4_ty = np.ndarray[(ROWS * OUT_BYTES,), np.dtype[np.int8]]
    q8_ty = np.ndarray[(8192,), np.dtype[np.int8]]
    scratch_ty = np.ndarray[(512,), np.dtype[np.float32]]
    src = os.path.join(HERE, "ffn.cc")

    # One object holds every entry point; ffn_pre compiles it.
    pre = ExternalFunction("ffn_pre", source_file=src, arg_types=[in_ty, q8_ty], object_file_name="ffn.o",
                         compile_flags=os.environ.get("FFN_FLAGS", "").split())

    def fn(name, types):
        return Kernel(name, "ffn.o", types)

    piece = fn("ffn_slice", [in_ty, out_ty, np.int32, np.int32, np.int32])
    gate_up = fn("ffn_gate_up", [q8_ty, rec_ty])
    gate_flush = fn("ffn_gate_flush", [out_ty])
    act_pre = fn("ffn_act_pre", [in_ty, q8_ty, scratch_ty])
    down = fn("ffn_down", [q8_ty, rec_ty])
    down_flush = fn("ffn_down_flush", [out_ty])
    copy = fn("ffn_copy", [rec_ty, in_ty, np.int32])

    w_l2l1, y_l1l2, w_fifos, y_fifos = [], [], [], []
    # Two weight streams per column (both shim MM2S channels), each split
    # between two cores.
    for c in range(COLS):
        col = []
        for half in range(2):
            f = ObjectFifo(rec2_ty, name=f"wl3_{c}_{half}", depth=2)
            w_fifos.append(f)
            col += f.cons().split([0, REC], obj_types=[rec_ty] * 2,
                                  names=[f"w{c}_{2 * half + k}" for k in range(2)])
        w_l2l1.append(col)
        o = ObjectFifo(out4_ty, name=f"y{c}", depth=2)
        y_fifos.append(o)
        y_l1l2.append(o.prod().join([OUT_BYTES * j for j in range(ROWS)], obj_types=[out_ty] * ROWS,
                                    names=[f"y{c}_{j}" for j in range(ROWS)], depths=[2] * ROWS))

    def take(w_in, x, records, copy):
        # Each phase's input rides at the head of the core's weight stream.
        for r in range(records):
            w = w_in.acquire(1)
            copy(w, x, r)
            w_in.release(1)

    def body(w_in, y_out, x, q8, scratch, c, j, pre, piece, gate_up, gate_flush, act_pre, down, down_flush, copy):
        # F1
        take(w_in, x, X1_RECS, copy)
        pre(x, q8)
        for t in range(8):
            o = y_out.acquire(1)
            piece(x, o, c, j, t)
            y_out.release(1)
        for _ in range_(GU_RECS // 8):
            o = y_out.acquire(1)
            for _ in range_(8):
                w = w_in.acquire(1)
                gate_up(q8, w)
                w_in.release(1)
            gate_flush(o)
            y_out.release(1)
        # F2
        take(w_in, x, X2_RECS, copy)
        act_pre(x, q8, scratch)
        for _ in range_(DOWN_RECS // 8):
            o = y_out.acquire(1)
            for _ in range_(8):
                w = w_in.acquire(1)
                down(q8, w)
                w_in.release(1)
            down_flush(o)
            y_out.release(1)

    workers = []
    for c in range(COLS):
        for j in range(ROWS):
            x = Buffer(in_ty, name=f"x_{c}_{j}")
            q8 = Buffer(q8_ty, name=f"q8_{c}_{j}")
            scratch = Buffer(scratch_ty, name=f"scratch_{c}_{j}")
            workers.append(Worker(body, [w_l2l1[c][j].cons(), y_l1l2[c][j].prod(), x, q8, scratch, c, j,
                                         pre, piece, gate_up, gate_flush, act_pre, down, down_flush, copy],
                                  tile=Tile(c, 2 + j), stack_size=0xA00))

    s_ty = np.ndarray[(S_BYTES,), np.dtype[np.int8]]
    w_ty = np.ndarray[(W_BYTES,), np.dtype[np.int8]]
    y_ty = np.ndarray[(S_BYTES,), np.dtype[np.int8]]
    w_prods = [f.prod() for f in w_fifos]
    y_conses = [f.cons() for f in y_fifos]

    def head(total, offset, records):
        # `records` records per core, the same bytes for both of a stream's cores.
        return TensorAccessPattern((total,), offset, [records, 2, REC // 512, 512], [REC, 0, 512, 1])

    def sequence(S, W, Y, wh, yh):
        # F1: x1 and activations land in Y at the state layout's offsets.
        tg = TaskGroup()
        for c in range(COLS):
            for half in range(2):
                n = GU_RECS * 2 * REC
                wh[2 * c + half].fill(S, head(S_BYTES, S_IN, X1_RECS), wait=True, group=tg)
                wh[2 * c + half].fill(W, tap(W_BYTES, (2 * c + half) * n, n), wait=True, group=tg)
        for c in range(COLS):
            yh[c].drain(Y, tap(S_BYTES, S_X1 + c * 1024, 1024), wait=True, group=tg)
            yh[c].drain(Y, tap(S_BYTES, S_ACT + c * 1920, 1920), wait=True, group=tg)
        tg.finish()
        # F2 reads the activations F1 wrote.
        tg = TaskGroup()
        for c in range(COLS):
            for half in range(2):
                n = DOWN_RECS * 2 * REC
                wh[2 * c + half].fill(Y, head(S_BYTES, S_ACT, X2_RECS), wait=True, group=tg)
                wh[2 * c + half].fill(W, tap(W_BYTES, W_GU + (2 * c + half) * n, n), wait=True, group=tg)
        for c in range(COLS):
            yh[c].drain(Y, tap(S_BYTES, S_O + c * 1024, 1024), wait=True, group=tg)
        tg.finish()

    rt = Runtime(sequence, [s_ty, w_ty, y_ty, w_prods, y_conses])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--xclbin-path", required=True)
    p.add_argument("--insts-path", required=True)
    a = p.parse_args()
    ffn.specialize().compile(xclbin_path=a.xclbin_path, inst_path=a.insts_path)
