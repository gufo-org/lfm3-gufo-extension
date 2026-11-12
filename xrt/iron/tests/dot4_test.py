import sys
from pathlib import Path
import aie.iron as iron
import numpy as np
from aie.iron import ExternalFunction, In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import from_name
HERE = Path(__file__).resolve().parent
N = 2048 + 128 + 128 + 8704

@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def t(A: In, B: Out):
    in_ty = np.ndarray[(N,), np.dtype[np.int8]]
    out_ty = np.ndarray[(16,), np.dtype[np.float32]]
    k = ExternalFunction("dot4_test", source_file=str(HERE / "dot4_test.cc"), arg_types=[in_ty, out_ty],
                         compile_flags=["-Os"], object_file_name="dot4_test.o")
    fi, fo = ObjectFifo(in_ty, name="i", depth=1), ObjectFifo(out_ty, name="o", depth=1)
    def body(i, o, k):
        a = i.acquire(1); b = o.acquire(1); k(a, b); i.release(1); o.release(1)
    w = Worker(body, [fi.cons(), fo.prod(), k])
    rt = Runtime(lambda a, b, p, c: (p.fill(a), c.drain(b, wait=True)), [np.ndarray[(N,), np.dtype[np.int8]], np.ndarray[(16,), np.dtype[np.float32]], fi.prod(), fo.cons()])
    return Program(from_name("npu2", n_cols=None), rt, workers=[w]).resolve_program()

if __name__ == "__main__":
    iron.set_current_device(from_name("npu2", n_cols=None))
    t.specialize().compile(xclbin_path=sys.argv[1] + ".xclbin", inst_path=sys.argv[1] + ".bin")
