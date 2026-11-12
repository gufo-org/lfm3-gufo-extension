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
    if what == ("head",):
        return (md.deq("token_embd.weight") * md.f("output_norm.weight")).T
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
    if name == "qkv":
        W = np.concatenate([md.deq(p + f"attn_{n}.weight") for n in ("q", "k", "v", "gate")]) * md.f(p + "attn_norm.weight")
        Bm = np.zeros((H, 4096), np.float32)
        Bm[:, :W.shape[0]] = W.T
        return Bm
    if name == "attn_out":
        return md.deq(p + "attn_output.weight").T
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
    if name == "qkn":
        qn = md.f(p + "attn_q_norm.weight") * (np.log2(np.e) / np.sqrt(128))
        return P.ablock_put(bf16(np.concatenate([qn, md.f(p + "attn_k_norm.weight")])))
    if name == "taps":
        q = what[2]
        taps = md.f(p + "shortconv.conv.weight").reshape(H, 7)[512 * q:512 * q + 512]
        return P.ablock_put(bf16(taps.T.reshape(-1)))
    raise KeyError(what)


def pack(plan, md):
    """Returns (weights bytes, scratch init bf16)."""
    E = []
    if not plan.kinds:  # swtest
        E.append(enc(gemm_weight(md, (0, "gate_up"))))
    for l in range(len(plan.kinds)):
        for ph in plan.phases:
            if ph["kind"] == "gemm" and ph["what"][0] == l:
                E.append(enc(gemm_weight(md, ph["what"])))
        md.drop()
    if hasattr(plan, "logits"):
        E.append(enc(gemm_weight(md, ("head",))))
        md.drop()
    Cs = np.zeros(plan.c_size, np.uint16)
    for off, what in plan.ablocks:
        Cs[off:off + P.ABLOCK] = ablock(md, what)
    # RoPE tables after each qkv row: [cos 64 | sin 64] for the global then the sliding-window kind
    rope = rope_tables(md, plan.M)
    tails = Cs[plan.qs:plan.qs + plan.M * P.QKV_ROW].reshape(plan.M, P.QKV_ROW)
    tails[:, 4096:4352] = bf16(np.concatenate([rope["attn"], rope["swa"]], 1))
    return np.concatenate(E), Cs


def rope_tables(md, n, start=0):
    i = np.arange(64)
    out = {}
    for kind, theta, f in (("attn", 1e6, md.f("rope_freqs.weight")), ("swa", 1e4, np.ones(64))):
        ang = np.arange(start, start + n)[:, None] * (theta ** (-2.0 * i / 128) / f)[None, :]
        out[kind] = np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    return out


def rms(v, w=None):
    v = v / np.sqrt((v * v).mean(-1, keepdims=True) + 1e-5)
    return v if w is None else v * w


def reference(md, kinds, x):
    """float32 forward of the layers; returns x after every layer."""
    xs = []
    for l, kind in enumerate(kinds):
        p = f"blk.{l}."
        z = rms(x, md.f(p + "attn_norm.weight"))
        if kind == "conv":
            pr = z @ md.deq(p + "shortconv.in_proj.weight").T
            bx = pr[:, :H] * pr[:, 2 * H:]
            taps = md.f(p + "shortconv.conv.weight").reshape(H, 7)
            pad = np.concatenate([np.zeros((6, H), np.float32), bx])
            conv = sum(taps[:, k] * pad[k:k + len(x)] for k in range(7))
            out = (conv * pr[:, H:2 * H]) @ md.deq(p + "shortconv.out_proj.weight").T
        else:
            out = attention(md, l, kind, z) @ md.deq(p + "attn_output.weight").T
        x = x + rms(out, md.f(p + "post_attention_norm.weight"))
        z = rms(x, md.f(p + "ffn_norm.weight"))
        g, u = z @ md.deq(p + "ffn_gate.weight").T, z @ md.deq(p + "ffn_up.weight").T
        x = x + rms((g / (1 + np.exp(-g)) * u) @ md.deq(p + "ffn_down.weight").T, md.f(p + "post_ffw_norm.weight"))
        xs.append(x)
        md.drop()
    return xs


def attention(md, l, kind, z):
    p = f"blk.{l}."
    n = len(z)
    q = (z @ md.deq(p + "attn_q.weight").T).reshape(n, 16, 128)
    k = (z @ md.deq(p + "attn_k.weight").T).reshape(n, 4, 128)
    v = (z @ md.deq(p + "attn_v.weight").T).reshape(n, 4, 128)
    gate = z @ md.deq(p + "attn_gate.weight").T
    cs = rope_tables(md, n)[kind]
    c, s = cs[:, None, :64], cs[:, None, 64:]
    rope = lambda x: np.concatenate([x[..., :64] * c - x[..., 64:] * s, x[..., 64:] * c + x[..., :64] * s], -1)
    q = rope(rms(q, md.f(p + "attn_q_norm.weight")))
    k = rope(rms(k, md.f(p + "attn_k_norm.weight")))
    t = np.arange(n)
    allowed = t[None, :] <= t[:, None]
    if kind == "swa":
        allowed &= t[None, :] > t[:, None] - P.WINDOW
    a = np.zeros((n, 16, 128), np.float32)
    for h in range(16):
        sc = q[:, h] @ k[:, h // 4].T / np.sqrt(128)
        sc = np.where(allowed, sc, -np.inf)
        e = np.exp(sc - sc.max(1, keepdims=True))
        a[:, h] = (e / e.sum(1, keepdims=True)) @ v[:, h // 4]
    return (a / (1 + np.exp(-gate[:, :, None]))).reshape(n, H)


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
    kinds = P.lfm3_kinds(int(kinds)) if kinds.isdigit() else kinds.split(",")
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
    if len(sys.argv) > 5 and sys.argv[5] == "debug" and kinds[-1] == "conv":
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


if __name__ == "__main__" and sys.argv[1] not in ("debug-attn", "kl", "head"):
    main()


def debug_attn(argv):
    """host.py debug-attn <xclbin> <insts> <gguf> <n layers>: last layer must be attention."""
    xclbin, insts, gguf, n = argv
    kinds = P.lfm3_kinds(int(n))
    md = Model(gguf)
    plan = P.Plan(kinds)
    E, Cs = pack(plan, md)
    emb = md.deq("token_embd.weight")
    x0 = emb[np.random.default_rng(2).integers(0, emb.shape[0], plan.M)]
    md.drop()
    run = Runner(xclbin, insts, plan, E, Cs)
    G, _ = run.run(x0)
    S = from_bf16(run.scratch())
    xs = reference(md, kinds[:-1], from_bf16(bf16(x0)).reshape(x0.shape))
    l = len(kinds) - 1
    z = rms(xs[-1] if xs else from_bf16(bf16(x0)).reshape(x0.shape), md.f(f"blk.{l}.attn_norm.weight"))
    a = attention(md, l, kinds[-1], z).reshape(plan.M, 16, 128)
    O = S[plan.ao:plan.ao + plan.oh].reshape(16, plan.M, 128).transpose(1, 0, 2)
    rel = lambda p, q: np.linalg.norm(p - q) / np.linalg.norm(q)
    print(f"attention out rel err {rel(O, a):.3e}")
    for h in (0, 5, 15):
        print(f"  head {h}: {rel(O[:, h], a[:, h]):.3e}; tokens 0..31 {rel(O[:32, h], a[:32, h]):.3e}, 992.. {rel(O[992:, h], a[992:, h]):.3e}")
    per_tok = np.linalg.norm(O - a, axis=(1, 2)) / np.linalg.norm(a, axis=(1, 2))
    print("  worst tokens", np.argsort(per_tok)[-8:], per_tok[np.argsort(per_tok)[-8:]].round(3))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "debug-attn":
    debug_attn(sys.argv[2:])
    sys.exit(0)


def kl_check(argv):
    """host.py kl <xclbin> <insts> <gguf> <ids.json>: NPU vs float32 reference on real text,
    full-vocabulary KL of the next-token distributions every 16 positions."""
    xclbin, insts, gguf, ids_path = argv
    import json
    kinds = P.lfm3_kinds()
    md = Model(gguf)
    plan = P.Plan(kinds)
    E, Cs = pack(plan, md)
    ids = np.array(([1] + json.load(open(ids_path)))[:plan.M])
    emb = md.deq("token_embd.weight")
    x0 = from_bf16(bf16(emb[ids])).reshape(plan.M, H)
    run = Runner(xclbin, insts, plan, E, Cs)
    G, dt = run.run(x0)
    print(f"NPU prefill {plan.M} tokens: {dt * 1e3:.1f} ms")
    md.drop()
    xs = reference(md, kinds, x0)
    pos = np.arange(15, plan.M, 16)
    wo = md.f("output_norm.weight")
    head = emb
    def dist(x):
        lg = rms(x[pos], wo) @ head.T
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg)
        return p / p.sum(1, keepdims=True)
    pr, pn = dist(xs[-1]), dist(G[:, H:2 * H])
    kl = (pr * (np.log(pr + 1e-30) - np.log(pn + 1e-30))).sum(1)
    top = (pr.argmax(1) == pn.argmax(1)).mean()
    nxt = np.arange(len(pos) - 1)  # positions whose next token is in the text
    nll_r = -np.log(pr[nxt, ids[pos[nxt] + 1]] + 1e-30).mean()
    nll_n = -np.log(pn[nxt, ids[pos[nxt] + 1]] + 1e-30).mean()
    print(f"KL(ref || npu) mean {kl.mean():.3e} median {np.median(kl):.2e} max {kl.max():.2e}; top-1 agree {top * 100:.1f}% "
          f"({len(pos)} positions); NLL ref {nll_r:.3f} npu {nll_n:.3f}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "kl":
    kl_check(sys.argv[2:])
    sys.exit(0)


def head_check(argv):
    """host.py head <xclbin> <insts row0=0> <insts row0=512> <gguf> <ids.json> <n>..."""
    xclbin, ins0, ins512, gguf, ids_path, *ns = argv
    import json
    kinds = P.lfm3_kinds()
    md = Model(gguf)
    plan = P.Plan(kinds, head_row0=0)
    E, Cs = pack(plan, md)
    ids_all = np.array([1] + json.load(open(ids_path)))
    emb = md.deq("token_embd.weight")
    runs = {0: Runner(xclbin, ins0, plan, E, Cs)}
    del E
    for n in map(int, ns):
        row0 = 0 if n - 1 < P.HEAD_ROWS else P.HEAD_ROWS
        if row0 not in runs:
            r = runs[0]
            ins = np.fromfile(ins512, dtype=np.uint32)
            r.bo_i2 = r.pyxrt.bo(r.dev, ins.nbytes, r.pyxrt.bo.cacheable, r.kern.group_id(1))
            r.bo_i2.write(ins.tobytes(), 0)
            r.bo_i2.sync(r.td)
            runs[row0] = r
        r = runs[0]
        x = np.zeros((plan.M, H), np.float32)
        x[:n] = from_bf16(bf16(emb[ids_all[:n]])).reshape(n, H)
        bo_i, n_ins = (r.bo_i, r.n_ins) if row0 == 0 else (r.bo_i2, r.bo_i2.size() // 4)
        R = np.zeros((plan.M, P.REC), np.uint16)
        R[:, H:2 * H] = bf16(x).reshape(plan.M, H)
        r.bo_a.write(R.tobytes(), 0)
        r.bo_a.sync(r.td)
        t0 = time.perf_counter()
        r.kern(3, bo_i, n_ins, r.bo_a, r.bo_b, r.bo_c).wait()
        dt = time.perf_counter() - t0
        r.bo_c.sync(r.fd)
        row = n - 1 - row0
        lg = from_bf16(np.frombuffer(r.bo_c.read(P.VOCAB * 2, (plan.logits + row * P.VOCAB) * 2), np.uint16))
        md.drop()
        xs = reference(md, kinds, x[:n])
        ref = rms(xs[-1][n - 1], md.f("output_norm.weight")) @ emb.T
        pr = np.exp(ref - ref.max()); pr /= pr.sum()
        pn = np.exp(lg - lg.max()); pn /= pn.sum()
        kl = float((pr * (np.log(pr + 1e-30) - np.log(pn + 1e-30))).sum())
        print(f"n {n} (head rows from {row0}): {dt * 1e3:.1f} ms, KL {kl:.3e}, top1 npu {int(pn.argmax())} ref {int(pr.argmax())} "
              f"text next {ids_all[n] if n < len(ids_all) else -1}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "head":
    head_check(sys.argv[2:])
    sys.exit(0)
