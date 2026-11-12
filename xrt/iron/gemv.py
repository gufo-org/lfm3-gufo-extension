"""Q8_0 GEMV for decode: y = W x over the whole NPU (8 columns x 4 rows).

Each core owns weight records of TM rows (int8 codes, then bf16 block
scales); column c streams its four cores' records through one shim channel
into the memory tile, which splits them. x (int8 codes, then bf16 scales) is
broadcast to every core once per call.
"""
import argparse
import os

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name

COLS, ROWS = 8, 4
HERE = os.path.dirname(os.path.abspath(__file__))


@iron.jit
def gemv(X: In, W: In, Y: Out, *, M: CompileTime[int], K: CompileTime[int], TM: CompileTime[int], P: CompileTime[int] = 1):
    cores = COLS * ROWS
    assert M % (cores * TM) == 0
    steps = M // (cores * TM)
    rec = TM * K + TM * (K // 32) * 2
    xrec = K + (K // 32) * 2
    x_ty = np.ndarray[(xrec,), np.dtype[np.int8]]
    w_l1 = np.ndarray[(rec,), np.dtype[np.int8]]
    w_l2 = np.ndarray[(ROWS * rec,), np.dtype[np.int8]]
    y_l1 = np.ndarray[(TM,), np.dtype[np.float32]]
    y_l2 = np.ndarray[(ROWS * TM,), np.dtype[np.float32]]

    kernel = ExternalFunction(
        "gemv_q8", source_file=os.path.join(HERE, "gemv_q8.cc"), arg_types=[x_ty, w_l1, y_l1],
        compile_flags=[f"-DGEMV_K={K}", f"-DGEMV_TM={TM}"])

    x_fifo = ObjectFifo(x_ty, name="x", depth=1)
    w_l3l2, w_l2l1, y_l1l2, y_l2l3 = [], [], [], []
    for c in range(COLS):
        f = ObjectFifo(w_l2, name=f"w{c}", depth=2)
        w_l3l2.append(f)
        w_l2l1.append(f.cons().split([rec * j for j in range(ROWS)], obj_types=[w_l1] * ROWS,
                                     names=[f"w{c}_{j}" for j in range(ROWS)]))
        o = ObjectFifo(y_l2, name=f"y{c}", depth=2)
        y_l2l3.append(o)
        y_l1l2.append(o.prod().join([TM * j for j in range(ROWS)], obj_types=[y_l1] * ROWS,
                                    names=[f"y{c}_{j}" for j in range(ROWS)], depths=[2] * ROWS))

    def body(x_in, w_in, y_out, k):
        for _ in range_(P):
            x = x_in.acquire(1)
            for _ in range_(steps):
                w = w_in.acquire(1)
                y = y_out.acquire(1)
                k(x, w, y)
                w_in.release(1)
                y_out.release(1)
            x_in.release(1)

    workers = [Worker(body, [x_fifo.cons(), w_l2l1[c][j].cons(), y_l1l2[c][j].prod(), kernel],
                      tile=Tile(c, 2 + j))
               for c in range(COLS) for j in range(ROWS)]
    w_prods = [f.prod() for f in w_l3l2]
    y_conses = [f.cons() for f in y_l2l3]
    x_prod = x_fifo.prod()
    wsz, ysz = COLS * steps * ROWS * rec, COLS * steps * ROWS * TM
    xs_ty = np.ndarray[(P * xrec,), np.dtype[np.int8]]
    w_ty = np.ndarray[(P * wsz,), np.dtype[np.int8]]
    yo_ty = np.ndarray[(P * ysz,), np.dtype[np.float32]]
    x_taps = TensorAccessPattern.full((P * xrec,)).partition(P)
    w_taps = TensorAccessPattern.full((P * wsz,)).partition(P * COLS)
    y_taps = TensorAccessPattern.full((P * ysz,)).partition(P * COLS)

    def sequence(X, W, Y, xh, wh, yh):
        # Phase p starts only after phase p - 1's outputs drained, as a
        # decode layer's next projection would.
        for p in range(P):
            tg = TaskGroup()
            xh.fill(X, x_taps[p], group=tg)
            for c in range(COLS):
                wh[c].fill(W, w_taps[p * COLS + c], group=tg)
            for c in range(COLS):
                yh[c].drain(Y, y_taps[p * COLS + c], wait=True, group=tg)
            tg.finish()

    rt = Runtime(sequence, [xs_ty, w_ty, yo_ty, x_prod, w_prods, y_conses])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=8192)
    p.add_argument("-K", type=int, default=2048)
    p.add_argument("--tm", type=int, default=8)
    p.add_argument("-P", type=int, default=1)
    p.add_argument("--xclbin-path", required=True)
    p.add_argument("--insts-path", required=True)
    a = p.parse_args()
    gemv.specialize(M=a.M, K=a.K, TM=a.tm, P=a.P).compile(xclbin_path=a.xclbin_path, inst_path=a.insts_path)
