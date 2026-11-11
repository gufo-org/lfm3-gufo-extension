"""Runs decode.xclbin (one convolution layer) for two tokens of an LFM3 Q8_0
GGUF and checks both against float64 model math.
Usage: decode_host.py <xclbin> <insts> <gguf> [layer]"""
import sys
import time

import numpy as np
import pyxrt

from ffn_host import bf16, f32, gguf_tensors, q8_rows

H, I, COLS, ROWS = 2048, 7680, 8, 4
REC = 4 * H + 4 * (H // 32) * 2
BLOCK = 12 * H
CONV_RECS, OUT_RECS, GU_RECS, DOWN_RECS = 64, 16, 120, 64
ACT_RECS = -(-(2 * I) // REC)
S_A = 0
S_B = S_A + BLOCK
S_NEXT = S_B + BLOCK
S_ACT = S_NEXT + BLOCK
S_VEC = S_ACT + ACT_RECS * REC
S_STATE = S_VEC + REC
STATE_COL = 48 * ROWS * 32
S_BYTES = S_STATE + COLS * STATE_COL + REC * 4


def hrow(c, j, l):
    return ((c * 8 + l // 8) * 4 + j) * 8 + l % 8


def ichannel(c, j, l):
    return ((c * 15 + l // 16) * 4 + j) * 16 + l % 16


def cchannel(c, j, l):
    return ((c * 4 + l // 16) * 4 + j) * 16 + l % 16


def pack(records_of):
    """records_of(c, j, r) -> bytes of record r of core (c, j); stream order."""
    out = []
    for c in range(COLS):
        for half in range(2):
            for r in range(records_of.n):
                for j in (2 * half, 2 * half + 1):
                    rec = np.zeros(REC, dtype=np.int8)
                    data = records_of(c, j, r)
                    rec[:data.size] = data
                    out.append(rec)
    return np.concatenate(out)


def rows_record(codes, scales, rows):
    sc = np.concatenate([bf16(scales[r]) for r in rows]).view(np.int8)
    return np.concatenate([codes[r] for r in rows] + [sc])


def main():
    xclbin, insts, gguf = sys.argv[1:4]
    L = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    d, info, base = gguf_tensors(gguf)
    p = f"blk.{L}."
    ic, isc = q8_rows(d, info, base, p + "shortconv.in_proj.weight")
    oc, osc = q8_rows(d, info, base, p + "shortconv.out_proj.weight")
    taps = f32(d, info, base, p + "shortconv.conv.weight").reshape(H, 7)
    gc, gs = q8_rows(d, info, base, p + "ffn_gate.weight")
    uc, us = q8_rows(d, info, base, p + "ffn_up.weight")
    dc, ds = q8_rows(d, info, base, p + "ffn_down.weight")
    n_attn = f32(d, info, base, p + "attn_norm.weight")
    n_post = f32(d, info, base, p + "post_attention_norm.weight")
    n_ffn = f32(d, info, base, p + "ffn_norm.weight")
    n_ffn_post = f32(d, info, base, p + "post_ffw_norm.weight")
    ec, es = q8_rows(d, info, base, "token_embd.weight")

    def conv_rec(c, j, r):
        ch = cchannel(c, j, r)
        rows = [ch, H + ch, 2 * H + ch]
        return np.concatenate([rows_record(ic, isc, rows), taps[ch].astype(np.float32).view(np.int8)])
    conv_rec.n = CONV_RECS
    def out_rec(c, j, r):
        return rows_record(oc, osc, [hrow(c, j, 4 * r + i) for i in range(4)])
    out_rec.n = OUT_RECS
    def gu_rec(c, j, r):
        a, b = ichannel(c, j, 2 * r), ichannel(c, j, 2 * r + 1)
        return np.concatenate([np.concatenate([gc[a], gc[b], uc[a], uc[b]]),
                               np.concatenate([bf16(gs[a]), bf16(gs[b]), bf16(us[a]), bf16(us[b])]).view(np.int8)])
    gu_rec.n = GU_RECS
    def down_rec(c, j, r):
        row = hrow(c, j, r)
        return np.concatenate([dc[row], bf16(ds[row]).view(np.int8)])
    down_rec.n = DOWN_RECS
    W = np.concatenate([pack(f) for f in (conv_rec, out_rec, gu_rec, down_rec)])

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xclbin)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
    ins = np.fromfile(insts, dtype=np.uint32)
    bo_i = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_s = pyxrt.bo(dev, S_BYTES, pyxrt.bo.host_only, kern.group_id(3))
    bo_w = pyxrt.bo(dev, W.nbytes, pyxrt.bo.host_only, kern.group_id(4))
    to_dev, from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    for bo, arr in ((bo_i, ins), (bo_w, W)):
        bo.write(arr.tobytes(), 0)
        bo.sync(to_dev)

    S = np.zeros(S_BYTES, dtype=np.int8)
    def block(at, h, o, wpost, wpre):
        S[at:at + 4 * H] = np.asarray(h, np.float32).view(np.int8)
        S[at + 4 * H:at + 8 * H] = np.asarray(o, np.float32).view(np.int8)
        S[at + 8 * H:at + 10 * H] = bf16(wpost).view(np.int8)
        S[at + 10 * H:at + 12 * H] = bf16(wpre).view(np.int8)

    def rms(v):
        v = np.asarray(v, np.float64)
        return v / np.sqrt(np.mean(v ** 2) + 1e-5)
    deq = lambda c, s: (c.astype(np.float64).reshape(c.shape[0], -1, 32) * s[:, :, None]).reshape(c.shape[0], -1)
    Win, Wout, Wg, Wu, Wd = deq(ic, isc), deq(oc, osc), deq(gc, gs), deq(uc, us), deq(dc, ds)
    hist = np.zeros((H, 6))
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(b)

    for step, token in enumerate([1, 5000]):
        emb = (ec[token].astype(np.float64).reshape(-1, 32) * es[token][:, None]).reshape(-1)
        if step == 0:
            S[:] = 0
        block(S_A, emb, np.zeros(H), np.zeros(H), n_attn)
        block(S_B, np.zeros(H), np.zeros(H), n_post, n_ffn)
        block(S_NEXT, np.zeros(H), np.zeros(H), n_ffn_post, n_attn)
        if step == 1:  # keep the device's convolution state from token 0
            S[S_STATE:] = Y[S_STATE:]
        bo_s.write(S.tobytes(), 0)
        bo_s.sync(to_dev)
        t0 = time.perf_counter()
        kern(3, bo_i, ins.size, bo_s, bo_w).wait()
        dt = time.perf_counter() - t0
        bo_s.sync(from_dev)
        Y = np.frombuffer(bo_s.read(S_BYTES, 0), dtype=np.int8).copy()
        o_conv = Y[S_B + 4 * H:S_B + 8 * H].view(np.float32)
        x_mid = Y[S_NEXT:S_NEXT + 4 * H].view(np.float32)
        o_ffn = Y[S_NEXT + 4 * H:S_NEXT + 8 * H].view(np.float32)

        # Model math.
        z = rms(emb) * n_attn
        pr = Win @ z
        bx = pr[:H] * pr[2 * H:]
        conv = (taps[:, :6] * hist).sum(1) + taps[:, 6] * bx
        hist = np.concatenate([hist[:, 1:], bx[:, None]], 1)
        r_conv = Wout @ (conv * pr[H:2 * H])
        r_mid = emb + rms(r_conv) * n_post
        z = rms(r_mid) * n_ffn
        g, u = Wg @ z, Wu @ z
        r_ffn = Wd @ (g / (1 + np.exp(-g)) * u)
        print(f"token {step}: {dt * 1e6:.0f} us; conv out {rel(o_conv, r_conv):.2e}, residual {rel(x_mid, r_mid):.2e}, "
              f"ffn out {rel(o_ffn, r_ffn):.2e} (post-normed {rel(rms(o_ffn) * n_ffn_post, rms(r_ffn) * n_ffn_post):.2e})")
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        kern(3, bo_i, ins.size, bo_s, bo_w).wait()
    dt = (time.perf_counter() - t0) / n
    print(f"layer: {dt * 1e6:.0f} us per run, {W.nbytes / 1e6:.1f} MB of records ({W.nbytes / dt / 1e9:.1f} GB/s)")


if __name__ == "__main__":
    main()
