"""Prefill layer FFN half on LFM3 Q8_0 weights vs float64:
norm(post_attention | ffn_norm), gate/up GEMM + SwiGLU, down GEMM into the
records' o slot, norm(post_ffw), with the pre-norm weights folded into the GEMMs that consume xn.
Records per token: [xn | x | o | pad] bf16, 8192 elements.
Usage: layer_host.py <xclbin> <insts> <gguf> [layer]"""
import sys, time
import numpy as np
import pyxrt
sys.path.insert(0, "/home/anvil/npu-work/iron")
from ffn_host import bf16, from_bf16, gguf_tensors, q8_rows, f32
from bfp_gemm_host import shuffle_b, to_bfp16

M, H, I, REC = 1024, 2048, 7680, 8192
xclbin, insts, gguf = sys.argv[1:4]
L = int(sys.argv[4]) if len(sys.argv) > 4 else 0
d, info, base = gguf_tensors(gguf)
deq = lambda cs: (cs[0].astype(np.float32).reshape(cs[0].shape[0], -1, 32) * cs[1][:, :, None]).reshape(cs[0].shape[0], -1)
F = lambda n: f32(d, info, base, n)
Wg, Wu, Wd = [deq(q8_rows(d, info, base, f"blk.{L}.ffn_{n}.weight")) for n in ("gate", "up", "down")]
w_post, w_ffn = F(f"blk.{L}.post_attention_norm.weight"), F(f"blk.{L}.ffn_norm.weight")
w_post_ffw = F(f"blk.{L}.post_ffw_norm.weight")
emb = deq(q8_rows(d, info, base, "token_embd.weight"))
rng = np.random.default_rng(2)
x0 = emb[rng.integers(0, emb.shape[0], M)]
o0 = rng.normal(0, 0.05, (M, H)).astype(np.float32)

R = np.zeros((M, REC), np.uint16)
R[:, H:2 * H] = bf16(x0).reshape(M, H)
R[:, 2 * H:3 * H] = bf16(o0).reshape(M, H)

B1 = np.zeros((H, 15360), np.float32)
for t in range(120):
    B1[:, 128 * t:128 * t + 64] = (Wg[64 * t:64 * t + 64] * w_ffn).T
    B1[:, 128 * t + 64:128 * t + 128] = (Wu[64 * t:64 * t + 64] * w_ffn).T
E1, _ = to_bfp16(shuffle_b(B1))
E2, _ = to_bfp16(shuffle_b(Wd.T.copy()))
def normw(wp):
    """One 128 x 512 A block whose first L1 object (rows 0..31 of the first 64 columns,
    blocked [m/r][k/s][r][s] with r = s = 8) holds wp linearly."""
    z = np.zeros((128, 512), np.uint16)
    p = np.arange(2048)
    z[(p // 512) * 8 + (p // 8) % 8, ((p // 64) % 8) * 8 + p % 8] = bf16(wp)
    return z.reshape(-1)
E = np.concatenate([E1, E2])
CW = np.concatenate([np.zeros(M * 15360, np.uint16), normw(w_post), normw(w_post_ffw)])

dev = pyxrt.device(0); xb = pyxrt.xclbin(xclbin); dev.register_xclbin(xb)
ctx = pyxrt.hw_context(dev, xb.get_uuid())
kern = pyxrt.kernel(ctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
ins = np.fromfile(insts, dtype=np.uint32)
td, fd = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
bo_i = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
bo_a = pyxrt.bo(dev, R.nbytes, pyxrt.bo.host_only, kern.group_id(3))
bo_b = pyxrt.bo(dev, E.nbytes, pyxrt.bo.host_only, kern.group_id(4))
bo_c = pyxrt.bo(dev, CW.nbytes, pyxrt.bo.host_only, kern.group_id(5))
for bo, arr in ((bo_i, ins), (bo_a, R), (bo_b, E), (bo_c, CW)):
    bo.write(arr.tobytes(), 0); bo.sync(td)
kern(3, bo_i, ins.size, bo_a, bo_b, bo_c).wait()
bo_a.sync(fd)
G = from_bf16(np.frombuffer(bo_a.read(R.nbytes, 0), np.uint16)).reshape(M, REC)
# Timing: the records are updated in place, so rerun from fresh records each time.
dts = []
for _ in range(10):
    bo_a.write(R.tobytes(), 0); bo_a.sync(td)
    t0 = time.perf_counter()
    kern(3, bo_i, ins.size, bo_a, bo_b, bo_c).wait()
    dts.append(time.perf_counter() - t0)
dt = min(dts)

rms = lambda v, w: v / np.sqrt((v * v).mean(1, keepdims=True) + 1e-5) * w
x = from_bf16(R[:, H:2 * H].reshape(-1)).reshape(M, H).astype(np.float64)
o = from_bf16(R[:, 2 * H:3 * H].reshape(-1)).reshape(M, H).astype(np.float64)
x1 = x + rms(o, w_post)
xn = rms(x1, w_ffn)
g, u = xn @ Wg.T, xn @ Wu.T
down = (g / (1 + np.exp(-g)) * u) @ Wd.T
x2 = x1 + rms(down, w_post_ffw)
# The device leaves each pre-norm weight to the next GEMM, so its xn is the unweighted norm.
xn2 = rms(x2, np.ones(H))
rel = lambda p, q: np.linalg.norm(p - q) / np.linalg.norm(q)
print(f"layer {L} FFN half, {M} tokens: {dt * 1e3:.2f} ms (median {np.median(dts) * 1e3:.2f})")
print(f"  down rel err {rel(G[:, 2 * H:3 * H], down):.2e}, x rel err {rel(G[:, H:2 * H], x2):.2e}, xn rel err {rel(G[:, :H], xn2):.2e}")
