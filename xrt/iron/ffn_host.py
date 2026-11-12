"""Runs ffn.xclbin on layer L of an LFM3 Q8_0 GGUF and checks it against a
float64 reference. Usage: ffn_host.py <xclbin> <insts> <gguf> [layer]"""
import struct
import sys
import time

import numpy as np
import pyxrt

H, I, COLS, ROWS = 2048, 7680, 8, 4
REC = 4 * H + 4 * (H // 32) * 2
IN_BYTES = 4 * H * 2 + 2 * H * 2
S_IN, S_X1 = 0, IN_BYTES
S_ACT = S_X1 + 4 * H
S_O = S_ACT + IN_BYTES
S_BYTES = S_O + 4 * H
GU_RECS, DOWN_RECS = 120, 64


def gguf_tensors(path):
    d = np.memmap(path, dtype=np.uint8, mode="r")
    o = 8
    nt, nkv = struct.unpack_from("<QQ", d, o)
    o += 16
    def rstr(o):
        n = struct.unpack_from("<Q", d, o)[0]
        return bytes(d[o + 8:o + 8 + n]).decode(), o + 8 + n
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(t, o):
        if t == 8:
            return rstr(o)[1]
        if t == 9:
            et, n = struct.unpack_from("<IQ", d, o)
            o += 12
            for _ in range(n):
                o = skip(et, o)
            return o
        return o + sizes[t]
    align = 32
    for _ in range(nkv):
        k, o = rstr(o)
        t = struct.unpack_from("<I", d, o)[0]
        o += 4
        if k == "general.alignment":
            align = struct.unpack_from("<I", d, o)[0]
        o = skip(t, o)
    info = {}
    for _ in range(nt):
        name, o = rstr(o)
        nd = struct.unpack_from("<I", d, o)[0]
        o += 4
        dims = struct.unpack_from("<" + "Q" * nd, d, o)
        o += 8 * nd
        t, off = struct.unpack_from("<IQ", d, o)
        o += 12
        info[name] = (dims, t, off)
    base = (o + align - 1) // align * align
    return d, info, base


def q8_rows(d, info, base, name):
    (k, m), t, off = info[name][0], info[name][1], info[name][2]
    assert t == 8, name
    raw = np.asarray(d[base + off: base + off + m * (k // 32) * 34]).reshape(m, k // 32, 34)
    scales = raw[:, :, :2].copy().view(np.float16).reshape(m, k // 32).astype(np.float32)
    codes = raw[:, :, 2:].copy().view(np.int8).reshape(m, k)
    return codes, scales


def f32(d, info, base, name):
    dims, t, off = info[name]
    assert t == 0
    n = int(np.prod(dims))
    return np.frombuffer(bytes(d[base + off: base + off + 4 * n]), dtype=np.float32)


def bf16(x):
    b = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(h):
    return (h.astype(np.uint32) << 16).view(np.float32)


def main():
    xclbin, insts, gguf = sys.argv[1:4]
    layer = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    d, info, base = gguf_tensors(gguf)
    p = f"blk.{layer}."
    gc, gs = q8_rows(d, info, base, p + "ffn_gate.weight")
    uc, us = q8_rows(d, info, base, p + "ffn_up.weight")
    dc, ds = q8_rows(d, info, base, p + "ffn_down.weight")
    w_post = f32(d, info, base, p + "post_attention_norm.weight")
    w_pre = f32(d, info, base, p + "ffn_norm.weight")
    w_ffn_post = f32(d, info, base, p + "post_ffw_norm.weight")

    # Weight records.
    gsb, usb, dsb = bf16(gs), bf16(us), bf16(ds)
    W = np.zeros(COLS * (GU_RECS + DOWN_RECS) * ROWS * REC, dtype=np.int8)
    def channel(c, j, l):
        return ((c * 15 + l // 16) * 4 + j) * 16 + l % 16
    def hrow(c, j, l):
        return ((c * 8 + l // 8) * 4 + j) * 8 + l % 8
    pos = 0
    for c in range(COLS):
      for half in range(2):
        for r in range(GU_RECS):
            for j in (2 * half, 2 * half + 1):
                ch = [channel(c, j, 2 * r), channel(c, j, 2 * r + 1)]
                rec = W[pos:pos + REC]
                rows = [gc[ch[0]], gc[ch[1]], uc[ch[0]], uc[ch[1]]]
                rec[:4 * H] = np.concatenate(rows)
                sc = np.concatenate([gsb[ch[0]], gsb[ch[1]], usb[ch[0]], usb[ch[1]]])
                rec[4 * H:] = sc.view(np.int8)
                pos += REC
    for c in range(COLS):
      for half in range(2):
        for l in range(DOWN_RECS):
            for j in (2 * half, 2 * half + 1):
                row = hrow(c, j, l)
                rec = W[pos:pos + REC]
                rec[:I] = dc[row]
                rec[I:I + 2 * (I // 32)] = dsb[row].view(np.int8)
                pos += REC
    assert pos == W.size

    rng = np.random.default_rng(5)
    h = rng.normal(0, 1, H).astype(np.float32)
    o = rng.normal(0, 0.5, H).astype(np.float32)
    S = np.zeros(S_BYTES, dtype=np.int8)
    S[0:4 * H] = h.view(np.int8)
    S[4 * H:8 * H] = o.view(np.int8)
    S[8 * H:10 * H] = bf16(w_post).view(np.int8)
    S[10 * H:12 * H] = bf16(w_pre).view(np.int8)

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xclbin)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kname = [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0]
    kern = pyxrt.kernel(ctx, kname)
    ins = np.fromfile(insts, dtype=np.uint32)
    bo_i = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_s = pyxrt.bo(dev, S.nbytes, pyxrt.bo.host_only, kern.group_id(3))
    bo_w = pyxrt.bo(dev, W.nbytes, pyxrt.bo.host_only, kern.group_id(4))
    bo_y = pyxrt.bo(dev, S.nbytes, pyxrt.bo.host_only, kern.group_id(5))
    for bo, arr in ((bo_i, ins), (bo_s, S), (bo_w, W), (bo_y, np.zeros_like(S))):
        bo.write(arr.tobytes(), 0)
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    def run():
        r = kern(3, bo_i, ins.size, bo_s, bo_w, bo_y)
        r.wait()
    run()
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        run()
    dt = (time.perf_counter() - t0) / n
    bo_y.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    Y = np.frombuffer(bo_y.read(S.nbytes, 0), dtype=np.int8)
    x1 = Y[S_X1:S_X1 + 4 * H].view(np.float32)
    act = from_bf16(Y[S_ACT:S_ACT + 2 * I].view(np.uint16))
    out = Y[S_O:S_O + 4 * H].view(np.float32)

    # float64 reference with the GGUF's own scales (the model's math).
    def rms(v):
        return v / np.sqrt(np.mean(v.astype(np.float64) ** 2) + 1e-5)
    rx1 = h.astype(np.float64) + rms(o) * w_post
    xn = rms(rx1) * w_pre
    deq = lambda c, s: (c.astype(np.float64).reshape(c.shape[0], -1, 32) * s[:, :, None]).reshape(c.shape[0], -1)
    g = deq(gc, gs) @ xn
    u = deq(uc, us) @ xn
    ract = g / (1 + np.exp(-g)) * u
    rout = deq(dc, ds) @ ract
    def rel(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)
    nbytes = W.nbytes
    print(f"layer {layer}: {dt * 1e6:.0f} us per block ({nbytes / dt / 1e9:.1f} GB/s of weights)")
    print(f"  x1 rel err {rel(x1, rx1):.2e}  act rel err {rel(act, ract):.2e}  down rel err {rel(out, rout):.2e}")
    rd = deq(dc, ds) @ act.astype(np.float64)
    print(f"  down vs NPU act: rel err {rel(out, rd):.2e}; first rows got {out[:4]} ref {rd[:4]}")
    best = [int(np.argmin(np.abs(rd - out[i]))) for i in range(6)]
    print("  nearest ref rows of out[0:6]:", best, "ratio", (out[:4] / rd[:4]))
    print(f"  block output after post-norm: rel err {rel(rms(out) * w_ffn_post, rms(rout) * w_ffn_post):.2e}")


if __name__ == "__main__":
    main()
