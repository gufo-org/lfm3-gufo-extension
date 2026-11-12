"""DDR read bandwidth probe: 16 shim MM2S streams (two per column) into 16 cores
that only release their objects. Usage: bw.py <out-prefix> <object bytes> [via memtile 0|1]"""
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import CompileTime, In, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import from_name

COLS, TOTAL = 8, 1 << 30


@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def bw(A: In, *, obj: CompileTime[int] = 8192, memtile: CompileTime[int] = 1):
    n_obj = TOTAL // 16 // obj
    ty = np.ndarray[(obj // 4,), np.dtype[np.int32]]
    fifos, workers = [], []
    for c in range(COLS):
        for s in range(2):
            f = ObjectFifo(ty, name=f"in_{c}_{s}", depth=2)
            fifos.append(f)
            cons = f.cons().forward(obj_type=ty, name=f"l1_{c}_{s}", depth=2) if memtile else f
            def body(i, n=n_obj):
                for _ in range_(n):
                    i.acquire(1)
                    i.release(1)
            workers.append(Worker(body, [cons.cons()]))
    A_ty = np.ndarray[(TOTAL // 4,), np.dtype[np.int32]]
    per = TOTAL // 4 // 16

    def seq(a, prods):
        tg = TaskGroup()
        for i, p in enumerate(prods):
            p.fill(a, tap=TensorAccessPattern((TOTAL // 4,), i * per, [1, per // 1024, 1, 1024], [0, 1024, 0, 1]), group=tg,
                   wait=True)
        tg.finish()

    rt = Runtime(seq, [A_ty, [f.prod() for f in fifos]])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    out, obj = sys.argv[1], int(sys.argv[2])
    mt = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    iron.set_current_device(from_name("npu2", n_cols=None))
    bw.specialize(obj=obj, memtile=mt).compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
