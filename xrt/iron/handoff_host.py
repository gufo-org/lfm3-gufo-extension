"""Prefill on the NPU, hand the KV caches and convolution states over to the decode
state, then decode on the NPU; compares the logits with the float32 model.
The same steps liblfm3_xrt.so runs.

Usage: handoff_host.py <bundle dir> <gguf> <ids.json> <prompt tokens> [decode steps]
"""
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyxrt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "prefill"))
sys.path.insert(0, str(HERE))
import host as PH  # noqa: E402
import plan as P  # noqa: E402
from ffn_host import bf16, from_bf16  # noqa: E402
from token_host import dispatch_args  # noqa: E402

TD, FD = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE


def kernel_of(dev, path):
    xb = pyxrt.xclbin(str(path))
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    k = pyxrt.kernel(ctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
    return ctx, k


def handoff(meta, S, C, n):
    """Writes the decode KV caches and convolution states for prefilled positions 0..n-1.
    S: decode state bytes (np.int8 view, writable); C: prefill scratch (uint16)."""
    d, p = meta["decode"], meta["prefill"]
    ctx, piece = meta["ctx"], p["piece"]
    t = np.arange(n)
    s_idx, d_idx = t % 16, np.arange(128)
    # piece element of (key s, dim d) in [kb8][dblk][8][8]
    inner = (((s_idx[:, None] >> 3) * 16 + (d_idx[None, :] >> 3)) << 6) + ((s_idx[:, None] & 7) << 3) + (d_idx[None, :] & 7)
    for i, l in enumerate(d["attn_layers"]):
        base = p["pieces"][str(l)]
        for g in range(4):
            off = base + ((t[:, None] // 16) * 4 + g) * piece + inner
            k, v = C[off + 9216], C[off + 11264]                     # [n, 128] bf16
            entry = np.concatenate([k, v], 1).view(np.int8)          # [n, 512 bytes]
            for h in range(4 * g, 4 * g + 4):
                c, j = h // 2, h % 2
                at = d["kv"][i] + (c * 4 + j) * ctx * 512
                S[at:at + n * 512] = entry.reshape(-1)
    f = np.arange(768)
    tap, lane = f // 128, f % 128
    for i, l in enumerate(d["conv_layers"]):
        rows = p["cs"][str(l)]
        st = np.zeros((8, 96, 4, 8), np.float32)
        for c in range(8):
            for j in (2, 3):
                ch = ((c * 8 + lane // 16) * 2 + (j - 2)) * 16 + lane % 16
                tok = n - 6 + tap + p["state_rows"]                  # scratch row of token n - 6 + tap
                vals = from_bf16(C[rows + tok * p["conv_row"] + (ch // 64) * 128 + ch % 64])
                st[c, :, j, :] = vals.reshape(96, 8)
        S[d["state"][i]:d["state"][i] + st.nbytes] = st.view(np.int8).reshape(-1)


def main():
    bundle, gguf, ids_path, n = sys.argv[1:5]
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    bundle, n = Path(bundle), int(n)
    meta = json.loads((bundle / "bundle.json").read_text())
    ids = [1] + json.load(open(ids_path))
    md = PH.Model(gguf)
    emb = md.deq("token_embd.weight")

    dev = pyxrt.device(0)
    pctx, pk = kernel_of(dev, bundle / "prefill.xclbin")
    dctx, dk = kernel_of(dev, bundle / "decode.xclbin")
    pins = np.fromfile(bundle / "prefill.insts", np.uint32)
    E = np.fromfile(bundle / "prefill_weights.bin", np.uint8)
    Cs = np.fromfile(bundle / "prefill_scratch.bin", np.uint16)
    M, REC, H = meta["prefill"]["M"], meta["prefill"]["rec"], meta["hidden"]
    bo = lambda k, size, g, kind=pyxrt.bo.host_only: pyxrt.bo(dev, size, kind, k.group_id(g))
    p_i, p_a, p_b, p_c = bo(pk, pins.nbytes, 1, pyxrt.bo.cacheable), bo(pk, M * REC * 2, 3), bo(pk, E.nbytes, 4), bo(pk, Cs.nbytes, 5)
    for b, arr in ((p_i, pins), (p_b, E), (p_c, Cs)):
        b.write(arr.tobytes(), 0)
        b.sync(TD)
    del E
    S = np.fromfile(bundle / "decode_state.bin", np.int8)
    W = np.fromfile(bundle / "decode_weights.bin", np.int8)
    d_s, d_w = bo(dk, S.nbytes, 3), bo(dk, W.nbytes, 4)
    d_w.write(W.tobytes(), 0)
    d_w.sync(TD)
    del W
    lib = ctypes.CDLL(str(bundle / "decode_dispatch.so"))
    lib.dispatch_generate.restype = ctypes.c_int64
    lib.dispatch_generate.argtypes = [ctypes.c_int32] * 8 + [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))]
    ptr = ctypes.POINTER(ctypes.c_uint32)()

    # prefill n - 1 tokens
    m = n - 1
    R = np.zeros((M, REC), np.uint16)
    R[:m, H:2 * H] = bf16(emb[ids[:m]])
    t0 = time.perf_counter()
    p_a.write(R.tobytes(), 0)
    p_a.sync(TD)
    t1 = time.perf_counter()
    pk(3, p_i, pins.size, p_a, p_b, p_c).wait()
    t2 = time.perf_counter()
    p_c.sync(FD)
    C = np.frombuffer(p_c.read(Cs.nbytes, 0), np.uint16)
    handoff(meta, S, C, m)
    d_s.write(S.tobytes(), 0)
    d_s.sync(TD)
    t3 = time.perf_counter()
    print(f"prefill {m} tokens: records {1e3 * (t1 - t0):.1f} ms, NPU {1e3 * (t2 - t1):.1f} ms, handoff {1e3 * (t3 - t2):.1f} ms")

    W_ = meta["window"]
    logits = []
    for k_ in range(steps):
        pos = m + k_
        tok = ids[pos]
        args = dispatch_args(tok, pos, W_)
        cnt = lib.dispatch_generate(*args, ctypes.byref(ptr))
        ins = np.ctypeslib.as_array(ptr, shape=(cnt,)).copy()
        d_i = bo(dk, ins.nbytes, 1, pyxrt.bo.cacheable)
        d_i.write(ins.tobytes(), 0)
        d_i.sync(TD)
        t4 = time.perf_counter()
        dk(3, d_i, ins.size, d_s, d_w).wait()
        t5 = time.perf_counter()
        d_s.sync(FD)
        lg = np.frombuffer(d_s.read(4 * meta["vocab"], meta["decode"]["logits"]), np.float32).copy()
        logits.append(lg)
        print(f"decode pos {pos}: {1e3 * (t5 - t4):.1f} ms, top1 {int(lg.argmax())} (next in text {ids[pos + 1]})")

    # float32 reference over the same tokens
    md.drop()
    kinds = P.lfm3_kinds()
    x0 = from_bf16(bf16(emb[ids[:n + steps - 1]])).reshape(-1, H)
    xs = PH.reference(md, kinds, x0)
    ref = PH.rms(xs[-1][m:m + steps], md.f("output_norm.weight")) @ emb.T
    for k_ in range(steps):
        pr = np.exp(ref[k_] - ref[k_].max()); pr /= pr.sum()
        pn = np.exp(logits[k_] - logits[k_].max()); pn /= pn.sum()
        kl = float((pr * (np.log(pr + 1e-30) - np.log(pn + 1e-30))).sum())
        print(f"  pos {m + k_}: KL {kl:.3e}, top1 npu {int(pn.argmax())} ref {int(pr.argmax())}")


if __name__ == "__main__":
    main()
