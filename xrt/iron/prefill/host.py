"""Host side of the prefill xclbin: weight packing for a Plan, a float32 numpy
reference, and a pyxrt runner.

Usage: host.py <xclbin> <insts> <gguf> <kinds> [tokens]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import plan as P  # noqa: E402
from bfp_gemm_host import shuffle_b, to_bfp16  # noqa: E402
from ffn_host import bf16, from_bf16, gguf_tensors, q8_rows, f32  # noqa: E402

H, I = P.H, P.I


class Model:
    def __init__(self, gguf):
        self.d, self.info, self.base = gguf_tensors(gguf)
        self._cache = {}

    def deq(self, name):
        if name not in self._cache:
            c, s = q8_rows(self.d, self.info, self.base, name)
            self._cache[name] = (c.astype(np.float32).reshape(c.shape[0], -1, 32) * s[:, :, None]).reshape(c.shape[0], -1)
        return self._cache[name]

    def f(self, name):
        return f32(self.d, self.info, self.base, name)

    def drop(self):
        self._cache.clear()


def enc(Bkn):
    return to_bfp16(shuffle_b(np.ascontiguousarray(Bkn, np.float32)))[0]


def gemm_weight(md, what):
    """[K, N] float32 weight of a GEMM phase, pre-norm weights folded into K."""
    l, name = what
    p = f"blk.{l}."
    if name in ("conv_bx", "conv_c"):
        W = md.deq(p + "shortconv.in_proj.weight") * md.f(p + "attn_norm.weight")
        if name == "conv_c":
            return W[H:2 * H].T
        Bm = np.zeros((H, 4096), np.float32)
        for j in range(32):
            Bm[:, 128 * j:128 * j + 64] = W[64 * j:64 * j + 64].T
            Bm[:, 128 * j + 64:128 * j + 128] = W[2 * H + 64 * j:2 * H + 64 * j + 64].T
        return Bm
    if name == "conv_out":
        return md.deq(p + "shortconv.out_proj.weight").T
    if name == "gate_up":
        w = md.f(p + "ffn_norm.weight")
        Wg, Wu = md.deq(p + "ffn_gate.weight") * w, md.deq(p + "ffn_up.weight") * w
        Bm = np.zeros((H, 2 * I), np.float32)
        for j in range(I // 64):
            Bm[:, 128 * j:128 * j + 64] = Wg[64 * j:64 * j + 64].T
            Bm[:, 128 * j + 64:128 * j + 128] = Wu[64 * j:64 * j + 64].T
        return Bm
    if name == "down":
        return md.deq(p + "ffn_down.weight").T
    raise KeyError(what)


def ablock(md, what):
    if what[0] == "embed":
        return P.ablock_put(np.zeros(1, np.uint16))
    l, name = what[:2]
    p = f"blk.{l}."
    if name == "post":
        return P.ablock_put(bf16(md.f(p + "post_attention_norm.weight")))
    if name == "ffn_post":
        return P.ablock_put(bf16(md.f(p + "post_ffw_norm.weight")))
    if name == "taps":
        q = what[2]
        taps = md.f(p + "shortconv.conv.weight").reshape(H, 7)[512 * q:512 * q + 512]
        return P.ablock_put(bf16(taps.T.reshape(-1)))
    raise KeyError(what)


def pack(plan, md):
    """Returns (weights bytes, scratch init bf16)."""
    E = []
    for l in range(len(plan.kinds)):
        for ph in plan.phases:
            if ph["kind"] == "gemm" and ph["what"][0] == l:
                E.append(enc(gemm_weight(md, ph["what"])))
        md.drop()
    Cs = np.zeros(plan.c_size, np.uint16)
    for off, what in plan.ablocks:
        Cs[off:off + P.ABLOCK] = ablock(md, what)
    return np.concatenate(E), Cs


def rms(v, w=None):
    v = v / np.sqrt((v * v).mean(-1, keepdims=True) + 1e-5)
    return v if w is None else v * w


def reference(md, kinds, x):
    """float32 forward of the layers; returns x after every layer."""
    xs = []
    for l, kind in enumerate(kinds):
        p = f"blk.{l}."
        z = rms(x, md.f(p + "attn_norm.weight"))
        assert kind == "conv"
        pr = z @ md.deq(p + "shortconv.in_proj.weight").T
        bx = pr[:, :H] * pr[:, 2 * H:]
        taps = md.f(p + "shortconv.conv.weight").reshape(H, 7)
        pad = np.concatenate([np.zeros((6, H), np.float32), bx])
        conv = sum(taps[:, k] * pad[k:k + len(x)] for k in range(7))
        out = (conv * pr[:, H:2 * H]) @ md.deq(p + "shortconv.out_proj.weight").T
        x = x + rms(out, md.f(p + "post_attention_norm.weight"))
        z = rms(x, md.f(p + "ffn_norm.weight"))
        g, u = z @ md.deq(p + "ffn_gate.weight").T, z @ md.deq(p + "ffn_up.weight").T
        x = x + rms((g / (1 + np.exp(-g)) * u) @ md.deq(p + "ffn_down.weight").T, md.f(p + "post_ffw_norm.weight"))
        xs.append(x)
        md.drop()
    return xs


class Runner:
    def __init__(self, xclbin, insts, plan, E, Cs):
        import pyxrt
        self.pyxrt = pyxrt
        self.dev = pyxrt.device(0)
        xb = pyxrt.xclbin(xclbin)
        self.dev.register_xclbin(xb)
        self.ctx = pyxrt.hw_context(self.dev, xb.get_uuid())
        self.kern = pyxrt.kernel(self.ctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
        ins = np.fromfile(insts, dtype=np.uint32)
        self.n_ins = ins.size
        self.td = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self.fd = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        bo = lambda nbytes, g, kind=pyxrt.bo.host_only: pyxrt.bo(self.dev, nbytes, kind, self.kern.group_id(g))
        self.bo_i = bo(ins.nbytes, 1, pyxrt.bo.cacheable)
        self.bo_a = bo(plan.M * P.REC * 2, 3)
        self.bo_b = bo(E.nbytes, 4)
        self.bo_c = bo(Cs.nbytes, 5)
        for b, arr in ((self.bo_i, ins), (self.bo_b, E), (self.bo_c, Cs)):
            b.write(arr.tobytes(), 0)
            b.sync(self.td)
        self.M = plan.M

    def run(self, x):
        """x: [M, H] float32 embeddings. Returns (records [M, REC] float32, seconds)."""
        R = np.zeros((self.M, P.REC), np.uint16)
        R[:, H:2 * H] = bf16(x).reshape(self.M, H)
        self.bo_a.write(R.tobytes(), 0)
        self.bo_a.sync(self.td)
        t0 = time.perf_counter()
        self.kern(3, self.bo_i, self.n_ins, self.bo_a, self.bo_b, self.bo_c).wait()
        dt = time.perf_counter() - t0
        self.bo_a.sync(self.fd)
        G = from_bf16(np.frombuffer(self.bo_a.read(R.nbytes, 0), np.uint16)).reshape(self.M, P.REC)
        return G, dt

    def scratch(self):
        self.bo_c.sync(self.fd)
        return np.frombuffer(self.bo_c.read(self.bo_c.size(), 0), np.uint16)


def main():
    xclbin, insts, gguf, kinds = sys.argv[1:5]
    kinds = kinds.split(",")
    md = Model(gguf)
    plan = P.Plan(kinds)
    t0 = time.perf_counter()
    E, Cs = pack(plan, md)
    print(f"packed {E.nbytes / 1e6:.0f} MB of weights in {time.perf_counter() - t0:.1f} s")
    emb = md.deq("token_embd.weight")
    rng = np.random.default_rng(2)
    x0 = emb[rng.integers(0, emb.shape[0], plan.M)]
    md.drop()
    run = Runner(xclbin, insts, plan, E, Cs)
    G, dt = run.run(x0)
    dts = [run.run(x0)[1] for _ in range(8)]
    print("  runs (ms):", " ".join(f"{d * 1e3:.1f}" for d in dts))
    print(f"{len(kinds)} layers ({','.join(kinds)}), {plan.M} tokens: {min(dts) * 1e3:.2f} ms (median {np.median(dts) * 1e3:.2f})")
    xs = reference(md, kinds, from_bf16(bf16(x0)).reshape(x0.shape))
    rel = lambda p, q: np.linalg.norm(p - q) / np.linalg.norm(q)
    print(f"  x rel err {rel(G[:, H:2 * H], xs[-1]):.3e}, xn rel err {rel(G[:, :H], rms(xs[-1])):.3e}")
    if len(sys.argv) > 5 and sys.argv[5] == "debug":
        S = from_bf16(run.scratch())
        l = len(kinds) - 1
        cs = plan.cs[l] + P.STATE * P.CONV_ROW
        rows = S[cs:cs + plan.M * P.CONV_ROW].reshape(plan.M, 32, 2, 64)
        p = f"blk.{l}."
        xin = xs[-2] if len(xs) > 1 else from_bf16(bf16(x0)).reshape(x0.shape)
        pr = rms(xin, md.f(p + "attn_norm.weight")) @ md.deq(p + "shortconv.in_proj.weight").T
        print(f"  bx rel err {rel(rows[:, :, 0].reshape(plan.M, H), pr[:, :H] * pr[:, 2 * H:]):.3e}, "
              f"C rel err {rel(rows[:, :, 1].reshape(plan.M, H), pr[:, H:2 * H]):.3e}")
        bx = pr[:, :H] * pr[:, 2 * H:]
        taps = md.f(p + "shortconv.conv.weight").reshape(H, 7)
        pad = np.concatenate([np.zeros((6, H), np.float32), bx])
        y = sum(taps[:, k] * pad[k:k + plan.M] for k in range(7)) * pr[:, H:2 * H]
        Y = S[plan.ys:plan.ys + plan.M * P.Y_ROW].reshape(plan.M, P.Y_ROW)[:, :H]
        print(f"  y rel err {rel(Y, y):.3e}")


if __name__ == "__main__":
    main()
