"""Runs lfm3_token.py's xclbin token by token on an LFM3 Q8_0 GGUF and compares
the logits with a float32 model on the same dequantized weights.
Usage: token_host.py <xclbin> <dispatch.so> <gguf> [tokens...]"""
import ctypes
import sys
import time

import numpy as np
import pyxrt

import layout as L
from ffn_host import bf16, f32, gguf_tensors, q8_rows

H, I, COLS, ROWS, REC, D = L.H, L.I, L.COLS, L.ROWS, L.REC, L.D


def hrow(c, j, l, chunks=8):
    return ((c * chunks + l // 8) * 4 + j) * 8 + l % 8


def stream_pack(builder, records, streams=range(16), cores_of=lambda s: (s % 2 * 2, s % 2 * 2 + 1)):
    out = []
    for s in streams:
        c = s // 2
        for r in range(records):
            for j in cores_of(s):
                rec = np.zeros(REC, np.int8)
                data = builder(c, j, r)
                rec[:data.size] = data
                out.append(rec)
    return np.concatenate(out)


def conv_channel(c, j, lane):
    """Channel of conv core (c, j in {2, 3}) lane 0..127."""
    return ((c * 8 + lane // 16) * 2 + (j - 2)) * 16 + lane % 16


def tile4(cs, ss):
    """Four rows of H codes (int8 [4, H]) and their bf16 block scales ([4, H / 32]) as
    decode.cc's Dot4 reads them: per 32-code block two 8 x 8 tiles t [k][n] (n < 4: row n,
    codes 16 t + k of the block; n >= 4: row n - 4, codes 16 t + 8 + k), then scales [block][row]."""
    c = np.asarray(cs, np.int8)
    nb = c.shape[1] // 32
    c = c.reshape(4, nb, 2, 2, 8)                                    # [row][block][t][half][k]
    t = c.transpose(1, 2, 4, 3, 0).reshape(nb, 2, 8, 8)              # [block][t][k][half * 4 + row]
    return np.concatenate([t.reshape(-1), np.asarray(ss, np.uint16).T.copy().view(np.int8).reshape(-1)])


def rows_record(codes, scales, rows):
    m = codes.shape[0]
    cs = [codes[r] if r < m else np.zeros(codes.shape[1], np.int8) for r in rows]
    ss = [bf16(scales[r]) if r < m else np.zeros(scales.shape[1], np.uint16) for r in rows]
    if len(rows) == 4 and codes.shape[1] == H:
        return tile4(cs, ss)
    return np.concatenate(cs + [np.concatenate(ss).view(np.int8)])


def pack(gguf, ctx=4096):
    """Decode state S and weights W for a GGUF, with the float32 reference weights."""
    lay = L.Layout(ctx)
    d, info, base = gguf_tensors(gguf)
    T = lambda n: q8_rows(d, info, base, n)
    F = lambda n: f32(d, info, base, n)

    print("packing weights ...", flush=True)
    W = np.zeros(lay.w_bytes, np.int8)
    ec, es = T("token_embd.weight")
    emb = (ec.astype(np.float32).reshape(ec.shape[0], -1, 32) * es[:, :, None]).reshape(ec.shape[0], H)
    ref = {"emb": emb}
    def put(key, arr):
        W[lay.w[key]:lay.w[key] + arr.size] = arr
    for l in range(L.LAYERS):
        p = f"blk.{l}."
        lw = {}
        if l in L.ATTN_LAYERS:
            q, qs = T(p + "attn_q.weight"); k, ks = T(p + "attn_k.weight"); v, vs = T(p + "attn_v.weight")
            g, gs = T(p + "attn_gate.weight"); o, os_ = T(p + "attn_output.weight")
            qkv = np.concatenate([q, k, v, g]); qkvs = np.concatenate([qs, ks, vs, gs])
            put((l, "qkv"), stream_pack(lambda c, j, r: rows_record(qkv, qkvs, [hrow(c, j, 4 * r + i, 13) for i in range(4)]),
                                        L.QKV_RECS))
            put((l, "attn_out"), stream_pack(lambda c, j, r: rows_record(o, os_, [hrow(c, j, 4 * r + i) for i in range(4)]),
                                             L.OUT_RECS))
            lw.update(qkv=(qkv, qkvs), out=(o, os_), qn=F(p + "attn_q_norm.weight"), kn=F(p + "attn_k_norm.weight"))
        else:
            ic, isc = T(p + "shortconv.in_proj.weight"); oc, osc = T(p + "shortconv.out_proj.weight")
            taps = F(p + "shortconv.conv.weight").reshape(H, 7)
            def conv_in(c, j, r):
                # ConvIn row l of core j drains to position p of column c's block
                # [conv core jj][B | C | x][128 channels] (ConvMix reads it per conv core)
                rows = []
                for l4 in range(4 * r, 4 * r + 4):
                    p_ = ((l4 // 8) * 4 + j) * 8 + l4 % 8
                    jj, part, rr = p_ // 384, (p_ % 384) // 128, p_ % 128
                    rows.append(part * H + conv_channel(c, jj + 2, rr))
                return rows_record(ic, isc, rows)
            put((l, "conv_in"), stream_pack(conv_in, L.CONVIN_RECS))
            put((l, "conv_out"), stream_pack(lambda c, j, r: rows_record(oc, osc, [hrow(c, j, 4 * r + i) for i in range(4)]),
                                             L.OUT_RECS))
            lw.update(inp=(ic, isc), out=(oc, osc), taps=taps)
        gc, gs = T(p + "ffn_gate.weight"); uc, us = T(p + "ffn_up.weight"); dc, ds = T(p + "ffn_down.weight")
        def gu(c, j, r):
            a, b = [((c * 15 + x // 16) * 4 + j) * 16 + x % 16 for x in (2 * r, 2 * r + 1)]
            return tile4([gc[a], gc[b], uc[a], uc[b]], [bf16(gs[a]), bf16(gs[b]), bf16(us[a]), bf16(us[b])])
        put((l, "gate_up"), stream_pack(gu, L.GU_RECS))
        def down(c, j, r):
            # four rows (r // 4) over slice r % 4 of I: 2048, 2048, 2048, 1536 codes
            rows, q = [hrow(c, j, 4 * (r // 4) + i) for i in range(4)], r % 4
            k0, k1 = 2048 * q, min(2048 * q + 2048, I)
            return tile4([dc[h][k0:k1] for h in rows], [bf16(ds[h][k0 // 32:k1 // 32]) for h in rows])
        put((l, "down"), stream_pack(down,
                                     L.DOWN_RECS))
        lw.update(gate=(gc, gs), up=(uc, us), down=(dc, ds), norm=F(p + "attn_norm.weight"),
                  post=F(p + "post_attention_norm.weight"), ffn_norm=F(p + "ffn_norm.weight"),
                  ffn_post=F(p + "post_ffw_norm.weight"))
        ref[l] = lw
    put((L.LAYERS, "head"), stream_pack(lambda c, j, r: rows_record(ec, es, [hrow(c, j, 4 * r + i, 500) for i in range(4)]),
                                        L.HEAD_RECS))
    out_norm = F("output_norm.weight")
    factors = F("rope_freqs.weight")

    S = np.zeros(lay.s_bytes, np.int8)
    S[lay.emb:lay.emb + emb.nbytes] = emb.view(np.int8).reshape(-1)
    def norms(b, wpost, wpre):
        at = lay.block(b)
        S[at + 8 * H:at + 10 * H] = bf16(wpost).view(np.int8)
        S[at + 10 * H:at + 12 * H] = bf16(wpre).view(np.int8)
    for l in range(L.LAYERS):
        norms(2 * l, np.zeros(H) if l == 0 else ref[l - 1]["ffn_post"], ref[l]["norm"])
        norms(2 * l + 1, ref[l]["post"], ref[l]["ffn_norm"])
    norms(48, ref[L.LAYERS - 1]["ffn_post"], out_norm)
    for i, l in enumerate(L.ATTN_LAYERS):
        at = lay.qkv[i] + L.QKV_ROWS * 4
        S[at:at + 2 * D] = bf16(ref[l]["qn"]).view(np.int8)
        S[at + 2 * D:at + 4 * D] = bf16(ref[l]["kn"]).view(np.int8)
    # conv taps after each conv core's state in its record: chunk k >= STATE_CHUNKS of
    # [chunk][core][8 f32], f = 8 (k - STATE_CHUNKS) + e = tap * 128 + lane
    f_ = np.arange(L.TAP_CHUNKS * 8)
    for i, l in enumerate(L.CONV_LAYERS):
        taps = ref[l]["taps"]
        for c in range(COLS):
            for j in (2, 3):
                vals = taps[conv_channel(c, j, f_ % 128), f_ // 128].astype(np.float32).reshape(L.TAP_CHUNKS, 8)
                at = lay.state[i] + c * L.STATE_COL + (L.STATE_CHUNKS * ROWS + j) * 32
                for k in range(L.TAP_CHUNKS):
                    S[at + k * ROWS * 32:at + k * ROWS * 32 + 32] = vals[k].view(np.int8)

    def rope_table(theta, f):
        i = np.arange(D // 2)
        inv = theta ** (-2.0 * i / D) / f
        ang = np.arange(ctx + 1)[:, None] * inv[None, :]
        return np.concatenate([np.cos(ang), np.sin(ang)], 1).astype(np.float32)
    rope = {"global": rope_table(1e6, factors), "swa": rope_table(1e4, np.ones(D // 2))}
    for kind, tab in rope.items():
        S[lay.rope[kind]:lay.rope[kind] + tab.nbytes] = tab.view(np.int8).reshape(-1)

    return lay, S, W, ref, emb, out_norm, rope

def dispatch_args(tok, pos, window=L.WINDOW):
    """Dispatch-time scalars of one decode token (see lfm3_token.token)."""
    first = max(0, pos - window + 1)
    cached_swa = pos - first
    recs = max(2, -(-pos // 16))     # the global stream is two DMA tasks of at least one record
    b = max(1, recs - 64)
    assert recs - b <= 64 and b <= 64, "decode context above 2048 positions needs more stream tasks"
    return [tok, pos, first, cached_swa, recs, max(1, -(-cached_swa // 16)), recs - b, b]


def main():
    xclbin, so, gguf = sys.argv[1:4]
    tokens = [int(x) for x in sys.argv[4:]] or [1, 1785, 3758, 314, 7428, 25, 1003, 2091]
    ctx = 4096
    lay, S, W, ref, emb, out_norm, rope = pack(gguf, ctx)
    d, info, base = gguf_tensors(gguf)
    T = lambda n: q8_rows(d, info, base, n)
    F = lambda n: f32(d, info, base, n)
    ec, es = T("token_embd.weight")
    print("device ...", flush=True)
    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xclbin)
    dev.register_xclbin(xb)
    hctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(hctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
    to_dev, from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo_s = pyxrt.bo(dev, S.nbytes, pyxrt.bo.host_only, kern.group_id(3))
    bo_w = pyxrt.bo(dev, W.nbytes, pyxrt.bo.host_only, kern.group_id(4))
    bo_s.write(S.tobytes(), 0); bo_s.sync(to_dev)
    bo_w.write(W.tobytes(), 0); bo_w.sync(to_dev)
    del W
    lib = ctypes.CDLL(so)
    lib.dispatch_generate.restype = ctypes.c_int64
    lib.dispatch_generate.argtypes = [ctypes.c_int32] * 8 + [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))]

    # float32 model on the same weights.
    deq = lambda cs: (cs[0].astype(np.float32).reshape(cs[0].shape[0], -1, 32) * cs[1][:, :, None]).reshape(cs[0].shape[0], -1)
    cache = {}
    def mat(l, name):
        key = (l, name)
        if key not in cache:
            cache[key] = deq(ref[l][name])
        return cache[key]
    def rms(v, w):
        return v / np.sqrt(np.mean(v * v) + 1e-5) * w
    hist = {l: np.zeros((H, 6), np.float32) for l in L.CONV_LAYERS}
    kvc = {l: ([], []) for l in L.ATTN_LAYERS}
    head = deq((ec, es))
    def rope_apply(x, p, kind):
        c, s = rope[kind][p, :D // 2], rope[kind][p, D // 2:]
        a, b = x[..., :D // 2].copy(), x[..., D // 2:].copy()
        return np.concatenate([a * c - b * s, b * c + a * s], -1)
    trace = []
    def ref_step(tok, p):
        trace.clear()
        x = emb[tok].astype(np.float32)
        for l in range(L.LAYERS):
            trace.append(x.copy())
            z = rms(x, ref[l]["norm"])
            if l in L.CONV_LAYERS:
                pr = mat(l, "inp") @ z
                bx = pr[:H] * pr[2 * H:]
                t = ref[l]["taps"]
                conv = (t[:, :6] * hist[l]).sum(1) + t[:, 6] * bx
                hist[l] = np.concatenate([hist[l][:, 1:], bx[:, None]], 1)
                out = mat(l, "out") @ (conv * pr[H:2 * H])
            else:
                kind = "swa" if l in L.SWA_LAYERS else "global"
                pr = mat(l, "qkv") @ z
                q = pr[:2048].reshape(16, D); k = pr[2048:2560].reshape(4, D); v = pr[2560:3072].reshape(4, D)
                gate = pr[3072:3088]
                q = rope_apply(np.stack([rms(h, ref[l]["qn"]) for h in q]), p, kind)
                k = rope_apply(np.stack([rms(h, ref[l]["kn"]) for h in k]), p, kind)
                ks, vs = kvc[l]
                ks.append(k); vs.append(v)
                first = max(0, p - L.WINDOW + 1) if kind == "swa" else 0
                K, V = np.stack(ks[first:]), np.stack(vs[first:])
                a = np.zeros((16, D), np.float32)
                for h in range(16):
                    sc = K[:, h // 4] @ q[h] / np.sqrt(D)
                    e = np.exp(sc - sc.max()); e /= e.sum()
                    a[h] = (e @ V[:, h // 4]) / (1 + np.exp(-gate[h]))
                out = mat(l, "out") @ a.reshape(-1)
            x = x + rms(out, ref[l]["post"])
            trace.append(x.copy())
            z = rms(x, ref[l]["ffn_norm"])
            g, u = mat(l, "gate") @ z, mat(l, "up") @ z
            x = x + rms(mat(l, "down") @ (g / (1 + np.exp(-g)) * u), ref[l]["ffn_post"])
        trace.append(x.copy())
        return head @ rms(x, out_norm)

    ptr = ctypes.POINTER(ctypes.c_uint32)()
    times = []
    for p, tok in enumerate(tokens):
        first = max(0, p - L.WINDOW + 1)
        cached_swa = p - first
        args = dispatch_args(tok, p)
        n = lib.dispatch_generate(*args, ctypes.byref(ptr))
        ins = np.ctypeslib.as_array(ptr, shape=(n,)).copy()
        bo_i = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
        bo_i.write(ins.tobytes(), 0); bo_i.sync(to_dev)
        t0 = time.perf_counter()
        kern(3, bo_i, ins.size, bo_s, bo_w).wait()
        times.append(time.perf_counter() - t0)
        bo_s.sync(from_dev)
        Y = np.frombuffer(bo_s.read(lay.s_bytes, 0), np.int8)
        got = Y[lay.logits:lay.logits + 4 * L.VOCAB].view(np.float32)
        want = ref_step(tok, p)
        if p == 0:
            for b in range(1, 49):
                at = lay.block(b)
                hh = Y[at:at + 4 * H].view(np.float32)
                e = np.linalg.norm(hh - trace[b - 1]) / np.linalg.norm(trace[b - 1])
                oo = Y[at + 4 * H:at + 8 * H].view(np.float32)
                print(f"  residual into block {b}: rel err {e:.2e}; o finite {np.isfinite(oo).sum()} max {np.nanmax(np.abs(oo)):.3g}")
                if not np.isfinite(e) or e > 0.5:
                    break
            print("  logits finite:", np.isfinite(got).sum(), "of", got.size, "max", np.nanmax(np.abs(got)))
        pg, pw = np.exp(got - got.max()), np.exp(want - want.max())
        pg /= pg.sum(); pw /= pw.sum()
        kl = float(np.sum(pw * (np.log(pw + 1e-30) - np.log(pg + 1e-30))))
        print(f"pos {p} tok {tok}: {times[-1] * 1e3:.1f} ms ({n * 4 // 1024} KB insts), top1 npu {int(got.argmax())} ref {int(want.argmax())}, "
              f"KL {kl:.2e}, logit rel err {np.linalg.norm(got - want) / np.linalg.norm(want):.2e}", flush=True)
    print(f"median {np.median(times) * 1e3:.1f} ms per token ({1 / np.median(times):.1f} tok/s)")


if __name__ == "__main__":
    main()
