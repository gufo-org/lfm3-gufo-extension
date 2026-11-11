"""Prefill FFN block (gate/up GEMM, SwiGLU epilogue, down GEMM) on LFM3 layer
weights vs float64. Usage: ffn_host.py <xclbin> <insts> <gguf> [layer]"""
import sys, time
import numpy as np
import pyxrt
sys.path.insert(0, "/home/anvil/npu-work/iron")
from ffn_host import bf16, from_bf16, gguf_tensors, q8_rows, f32
from bfp_gemm_host import shuffle_b, to_bfp16

M, H, I = 1024, 2048, 7680
xclbin, insts, gguf = sys.argv[1:4]
L = int(sys.argv[4]) if len(sys.argv) > 4 else 0
d, info, base = gguf_tensors(gguf)
deq = lambda cs: (cs[0].astype(np.float32).reshape(cs[0].shape[0], -1, 32) * cs[1][:, :, None]).reshape(cs[0].shape[0], -1)
Wg, Wu, Wd = [deq(q8_rows(d, info, base, f"blk.{L}.ffn_{n}.weight")) for n in ("gate", "up", "down")]
emb = deq(q8_rows(d, info, base, "token_embd.weight"))
rng = np.random.default_rng(2)
x = emb[rng.integers(0, emb.shape[0], M)]
xn = x / np.sqrt((x * x).mean(1, keepdims=True) + 1e-5) * f32(d, info, base, f"blk.{L}.ffn_norm.weight")
A = bf16(xn.astype(np.float32))
B1 = np.zeros((H, 15360), np.float32)
for t in range(120):
    B1[:, 128 * t:128 * t + 64] = Wg[64 * t:64 * t + 64].T
    B1[:, 128 * t + 64:128 * t + 128] = Wu[64 * t:64 * t + 64].T
E1, _ = to_bfp16(shuffle_b(B1))
E2, _ = to_bfp16(shuffle_b(Wd.T.copy()))
E = np.concatenate([E1, E2])
dev = pyxrt.device(0); xb = pyxrt.xclbin(xclbin); dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kern = pyxrt.kernel(ctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
ins = np.fromfile(insts, dtype=np.uint32)
td, fd = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
csz = M * 15360 + M * 2048
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
out = C[M * 15360:].reshape(M, H)
a = from_bf16(A).reshape(M, H).astype(np.float64)
g, u = a @ Wg.T, a @ Wu.T
act = g / (1 + np.exp(-g)) * u
ref = act @ Wd.T
got_act = C[:M * 15360].reshape(M, 120, 128)[:, :, :64].reshape(M, I)
rel = lambda p, q: np.linalg.norm(p - q) / np.linalg.norm(q)
print(f"layer {L} FFN, {M} tokens: {dt * 1e3:.2f} ms ({2 * M * H * I * 3 / dt / 1e12:.1f} TFLOPS); act rel err {rel(got_act, act):.2e}, out rel err {rel(out, ref):.2e}")
