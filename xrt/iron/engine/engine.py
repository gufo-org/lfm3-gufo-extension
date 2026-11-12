"""The stack engine: the whole NPU (8 columns x 4 rows) as a phase interpreter
for Q8_0 products and row operations (engine.cc), so a tower segment of many
phases runs in one dispatch. Each of the 16 shim streams broadcasts to two
cores of a column (rows 2-3 or 4-5); each column joins its four cores' output
records into one shim channel.

The host writes the instruction streams itself (xrt/lib/engine_xrt.cpp): per
phase, a header record then the phase's records on every stream, and the
drains. The sequence here only exercises every stream and drain once, so the
host can read the shim channel of each from the generated instructions.

Usage: engine.py <out xclbin> <out reference instructions>
"""
import os
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, ExternalFunction, In, InOut, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name

COLS, ROWS = 8, 4
REC, OUT = 1096, 128            # words per input and output record
A_WORDS = 8 * 1024              # the cores' A / packet buffer (32 KiB)
S_WORDS, W_WORDS = 256 << 20, 256 << 20
HERE = os.path.dirname(os.path.abspath(__file__))


def tap(total, offset, sizes, strides):
    return TensorAccessPattern((total,), offset, sizes, strides)


@iron.jit
def engine(S: InOut, W: In):
    rec_ty = np.ndarray[(REC,), np.dtype[np.int32]]
    out_ty = np.ndarray[(OUT,), np.dtype[np.int32]]
    out4_ty = np.ndarray[(ROWS * OUT,), np.dtype[np.int32]]
    abuf_ty = np.ndarray[(A_WORDS,), np.dtype[np.int32]]
    cnt_ty = np.ndarray[(8,), np.dtype[np.int32]]
    obj = "engine.o"
    head = ExternalFunction("eng_head", source_file=os.path.join(HERE, "engine.cc"), arg_types=[rec_ty, cnt_ty, np.int32],
                            object_file_name=obj, compile_flags=["-O2"] + os.environ.get("ENGINE_FLAGS", "").split())
    k = {"zero": Kernel("eng_zero", obj, []), "acopy": Kernel("eng_acopy", obj, [rec_ty, abuf_ty]),
         "mac": Kernel("eng_mac", obj, [abuf_ty, rec_ty]), "store": Kernel("eng_store", obj, [out_ty]),
         "rin": Kernel("eng_rin", obj, [rec_ty, abuf_ty]), "rout": Kernel("eng_rout", obj, [abuf_ty, out_ty])}

    streams, l1, outs, joins = [], {}, [], {}
    for c in range(COLS):
        for h in range(2):
            f = ObjectFifo(rec_ty, name=f"w_{c}_{h}", depth=4)
            streams.append(f)
            fwd = f.cons().forward(obj_type=rec_ty, name=f"l1_{c}_{h}", depth=2)
            l1[(c, 2 * h)] = l1[(c, 2 * h + 1)] = fwd
        o = ObjectFifo(out4_ty, name=f"y_{c}", depth=2)
        outs.append(o)
        parts = o.prod().join([OUT * j for j in range(ROWS)], obj_types=[out_ty] * ROWS,
                              names=[f"y_{c}_{j}" for j in range(ROWS)], depths=[2] * ROWS)
        for j in range(ROWS):
            joins[(c, j)] = parts[j]

    def body(w_in, y_out, a, cnt, parity, head, zero, acopy, mac, store, rin, rout):
        r = w_in.acquire(1)
        head(r, cnt, parity)
        w_in.release(1)
        for _ in range_(cnt[0]):
            r = w_in.acquire(1)
            rin(r, a)
            w_in.release(1)
        for _ in range_(cnt[1]):
            y = y_out.acquire(1)
            rout(a, y)
            y_out.release(1)
        for _ in range_(cnt[2]):
            zero()
            for _ in range_(cnt[3]):
                for _ in range_(cnt[4]):
                    r = w_in.acquire(1)
                    acopy(r, a)
                    w_in.release(1)
                for _ in range_(cnt[5]):
                    r = w_in.acquire(1)
                    mac(a, r)
                    w_in.release(1)
            for _ in range_(cnt[6]):
                y = y_out.acquire(1)
                store(y)
                y_out.release(1)

    workers = [Worker(body, [l1[key].cons(), joins[key].prod(), Buffer(abuf_ty, name=f"a_{key[0]}_{key[1]}"),
                             Buffer(cnt_ty, name=f"cnt_{key[0]}_{key[1]}"), key[1] % 2, head, *k.values()],
                      tile=Tile(key[0], 2 + key[1]), stack_size=0xF00, dynamic_objfifo_lowering=True)
               for key in sorted(l1)]

    s_ty = np.ndarray[(S_WORDS,), np.dtype[np.int32]]
    w_ty = np.ndarray[(W_WORDS,), np.dtype[np.int32]]

    def sequence(S, W, wh, yh):
        # stream s reads W at s REC (one record), column c drains 512 words to S at c 512
        tg = TaskGroup()
        for s in range(2 * COLS):
            wh[s].fill(W, tap(W_WORDS, (s + 1) * REC, [1, 1, 1, REC], [0, 0, REC, 1]), group=tg)
        for c in range(COLS):
            yh[c].drain(S, tap(S_WORDS, (c + 1) * ROWS * OUT, [1, 1, 1, ROWS * OUT], [0, 0, ROWS * OUT, 1]), wait=True,
                        group=tg)
        tg.finish()

    rt = Runtime(sequence, [s_ty, w_ty, [f.prod() for f in streams], [o.cons() for o in outs]])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    iron.set_current_device(from_name("npu2", n_cols=None))
    engine.specialize().compile(xclbin_path=sys.argv[1], inst_path=sys.argv[2])
