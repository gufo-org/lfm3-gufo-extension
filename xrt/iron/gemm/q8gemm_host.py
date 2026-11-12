"""Checks q8gemm.py against numpy: y = W x with Q8_0 weights and Q8_0-quantized x.
Usage: q8gemm_host.py <xclbin> <dispatch.so> [rows] [N] [K] [reps]"""
import ctypes
import sys
import time

import numpy as np
import pyxrt

OBJ, CORES, MAX_C, STREAMS = 1088, 32, 32, 16
TD, FD = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE


def q8(x):
    """Q8_0 along the last axis (blocks of 32): codes int8, scales f32 (f16-rounded)."""
    b = x.reshape(*x.shape[:-1], -1, 32)
    d = (np.abs(b).max(-1) / 127).astype(np.float16).astype(np.float32)
    q = np.where(d[..., None] > 0, np.round(b / np.where(d > 0, d, 1)[..., None]), 0).clip(-127, 127).astype(np.int8)
    return q.reshape(x.shape), d


def shape(rows, N, K):
    g, kc = -(-rows // 8), -(-K // 512)
    nt = -(-N // (CORES * 8))
    assert g * kc <= 4 and g * nt <= MAX_C, (g, kc, nt)
    return g, kc, nt


def fractions(d):
    """Block scales [N][K/32] -> int16 fractions u of each row's largest, and that largest."""
    S = d.max(1)
    S = np.where(S > 0, S, 1).astype(np.float32)
    return np.round(d / S[:, None] * 32767).astype(np.int16), S


def pack_w(q, u, nt, kc):
    """W codes [N][K] and fractions [N][K/32] -> records [stream][chunk][tile][parity]
    (core = 2 stream + parity takes columns (core nt + tile) 8 ..)."""
    N, K = q.shape
    Np, Kp = CORES * nt * 8, kc * 512
    qp = np.zeros((Np, Kp), np.int8); qp[:N, :K] = q
    up = np.zeros((Np, Kp // 32), np.int16); up[:N, :K // 32] = u
    # codes: [stream][parity][tile][nn][chunk][blk][k] -> [stream][chunk][tile][parity][blk][k][nn]
    c = qp.reshape(STREAMS, 2, nt, 8, kc, 16, 32).transpose(0, 4, 2, 1, 5, 6, 3)
    s = up.reshape(STREAMS, 2, nt, 8, kc, 16).transpose(0, 4, 2, 1, 5, 3)
    rec = np.concatenate([c.reshape(STREAMS, kc, nt, 2, 4096).view(np.uint8),
                          s.reshape(STREAMS, kc, nt, 2, 128).view(np.uint8)], -1)
    return rec.reshape(-1)


def q16(x):
    """Rows to int16 codes with one scale per row."""
    r = np.abs(x).max(-1) / 32767
    r = np.where(r > 0, r, 1).astype(np.float32)
    return np.round(x / r[:, None]).clip(-32767, 32767).astype(np.int16), r


def pack_a(q, g, kc):
    """int16 rows -> A records [g][chunk][half]: [blk 8][k4 8][mm 8][k 4], 512 bytes of padding."""
    M, K = q.shape
    qp = np.zeros((g * 8, kc * 512), np.int16); qp[:M, :K] = q
    c = qp.reshape(g, 8, kc, 2, 8, 8, 4).transpose(0, 2, 3, 4, 5, 1, 6)  # [g][chunk][half][blk][k4][mm][k]
    rec = np.zeros((g, kc, 2, OBJ * 4), np.uint8)
    rec[..., :4096] = c.reshape(g, kc, 2, 2048).view(np.uint8)
    return rec.reshape(-1)


def unpack_c(C, rows, N, g, nt):
    """Joined records [column][tile][group][row j][hi 64 | lo 64] -> int64 [rows][N] (core = 4 column + j)."""
    r = C.view(np.int32)[:8 * nt * g * 4 * 128].reshape(8, nt, g, 4, 2, 64)
    v = (r[..., 0, :].astype(np.int64) << 24) | (r[..., 1, :].astype(np.int64) & 0xFFFFFF)
    v = v.reshape(8, nt, g, 4, 8, 8).transpose(2, 4, 0, 3, 1, 5)  # [g][mm][col][j][tile][nn]
    return v.reshape(g * 8, CORES * nt * 8)[:rows, :N]


def main():
    xclbin, so = sys.argv[1:3]
    rows, N, K, reps = (int(a) for a in (sys.argv[3:7] + ["12", "2816", "1024", "20"][len(sys.argv) - 3:]))
    rng = np.random.default_rng(0)
    W = rng.standard_normal((N, K)).astype(np.float32)
    x = rng.standard_normal((rows, K)).astype(np.float32)
    wq, wd = q8(W)
    xq, xr = q16(x)
    wu, wS = fractions(wd)
    iref = ((xq.reshape(rows, 1, -1, 32).astype(np.int64) * wq.reshape(1, N, -1, 32)).sum(-1) * wu[None]).sum(-1)
    scale = xr[:, None].astype(np.float64) * wS[None].astype(np.float64) / 32767
    ref = iref * scale
    exact = x @ (wq.reshape(N, -1, 32) * wd[..., None]).reshape(N, K).T
    g, kc, nt = shape(rows, N, K)
    Wb, Ab = pack_w(wq, wu, nt, kc), pack_a(xq, g, kc)

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xclbin)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    k = pyxrt.kernel(ctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
    lib = ctypes.CDLL(so)
    lib.dispatch_generate.restype = ctypes.c_int64
    lib.dispatch_generate.argtypes = [ctypes.c_int32] * 10 + [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32))]
    ptr = ctypes.POINTER(ctypes.c_uint32)()
    cnt = lib.dispatch_generate(g, kc, nt, 2 * g * kc, g * nt, 2 * nt, 2 * kc * nt, 0, 0, 0, ctypes.byref(ptr))
    assert cnt > 0, cnt
    ins = np.ctypeslib.as_array(ptr, shape=(cnt,)).copy()
    bo = lambda size, gi, kind=pyxrt.bo.host_only: pyxrt.bo(dev, size, kind, k.group_id(gi))
    bi, ba, bw, bc = bo(ins.nbytes, 1, pyxrt.bo.cacheable), bo(64 * OBJ * 4, 3), bo(Wb.nbytes, 4), bo(8 * MAX_C * 4 * 512, 5)
    for b, arr in ((bi, ins), (ba, Ab), (bw, Wb)):
        b.write(arr.tobytes(), 0)
        b.sync(TD)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        k(3, bi, ins.size, ba, bw, bc).wait()
        ts.append(time.perf_counter() - t0)
    bc.sync(FD)
    yi = unpack_c(np.frombuffer(bc.read(8 * MAX_C * 4 * 512, 0), np.uint8).copy(), rows, N, g, nt)
    print("integer mismatches", int((yi != iref).sum()), "of", yi.size)
    y = yi * scale
    err = np.abs(y - ref).max() / np.abs(ref).max()
    ms = 1e3 * np.median(ts)
    err_x = np.abs(y - exact).max() / np.abs(exact).max()
    print(f"rows {rows} N {N} K {K}: rel err {err:.2e} (vs f32 x {err_x:.2e}), {ms:.3f} ms median ({1e3 * min(ts):.3f} min), "
          f"{Wb.nbytes / ms / 1e6:.1f} GB/s of weights")


if __name__ == "__main__":
    main()
