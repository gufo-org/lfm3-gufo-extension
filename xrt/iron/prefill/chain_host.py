"""Chained bf16 x bfp16 GEMMs in one run: checks each against numpy, times the run."""
import sys, time
import numpy as np
import pyxrt
sys.path.insert(0, "/home/anvil/npu-work/iron")
from ffn_host import bf16, from_bf16
from bfp_gemm_host import shuffle_b, to_bfp16

M = 1024
shapes = [(2048, 16384), (7680, 2048)]
rng = np.random.default_rng(1)
As, Bs, encs = [], [], []
for K, N in shapes:
    As.append(bf16(rng.normal(0, 1, (M, K)).astype(np.float32)))
    B = (rng.normal(0, 0.02, (K, N))).astype(np.float32)
    enc, deq = to_bfp16(shuffle_b(B))
    Bs.append(B); encs.append(enc)
A = np.concatenate([a.reshape(-1) for a in As]); E = np.concatenate(encs)
xclbin, insts = sys.argv[1], sys.argv[2]
dev = pyxrt.device(0); xb = pyxrt.xclbin(xclbin); dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kern = pyxrt.kernel(ctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
ins = np.fromfile(insts, dtype=np.uint32)
td, fd = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
csz = sum(M * N for _, N in shapes)
bo_i = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_a = pyxrt.bo(dev, A.nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo_b = pyxrt.bo(dev, E.nbytes, pyxrt.bo.host_only, kern.group_id(4))
bo_c = pyxrt.bo(dev, csz * 2, pyxrt.bo.host_only, kern.group_id(5))
for bo, arr in ((bo_i, ins), (bo_a, A), (bo_b, E)):
    bo.write(arr.tobytes(), 0); bo.sync(td)
kern(3, bo_i, ins.size, bo_a, bo_b, bo_c).wait()
t0 = time.perf_counter()
for _ in range(10):
    kern(3, bo_i, ins.size, bo_a, bo_b, bo_c).wait()
dt = (time.perf_counter() - t0) / 10
bo_c.sync(fd)
C = from_bf16(np.frombuffer(bo_c.read(csz * 2, 0), np.uint16))
off = 0
flops = 0
for (K, N), a, B in zip(shapes, As, Bs):
    ref = from_bf16(a).reshape(M, K).astype(np.float64) @ B
    got = C[off:off + M * N].reshape(M, N)
    print(f"  {M}x{K}x{N}: rel err {np.linalg.norm(got - ref) / np.linalg.norm(ref):.2e}")
    off += M * N
    flops += 2 * M * K * N
print(f"chain: {dt * 1e3:.2f} ms, {flops / dt / 1e12:.1f} TFLOPS")
