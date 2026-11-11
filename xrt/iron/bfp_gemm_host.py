"""Checks the asymmetric-tile-buffering bf16 x bfp16ebs8 GEMM (mlir-aie
config1) on LFM3 layer weights: C = A @ B with B = [gate; up]^T of a layer,
padded to N. Usage: bfp_gemm_host.py <xclbin> <insts> <gguf> M K N"""
import sys
import time

import numpy as np
import pyxrt

from ffn_host import bf16, from_bf16, gguf_tensors, q8_rows


def shuffle_b(B, k=64, n=128):
    """layout_transpose_L1_1x2_8x8block: L1 tiles column-major; per tile, 1x2
    super-blocks of 8x8 column-major blocks."""
    K, N = B.shape
    t = B.reshape(K // k, k, N // n, n).transpose(2, 0, 1, 3)        # [Lcol][Lrow][k][n]
    t = t.reshape(N // n, K // k, k // 8, 8, n // 16, 2, 8)          # rows: blockrow, r; cols: superblock, b, c
    t = t.transpose(0, 1, 4, 2, 5, 6, 3)                              # [Lc][Lr][sb][br][b][c][r]
    return t.reshape(-1)


def to_bfp16(x):
    """bfp16ebs8, blocks of 8 consecutive values: exponent byte, 8 int8
    mantissas; value = m * 2^(e - 127) / 64. Rounds to nearest."""
    x = x.astype(np.float32).reshape(-1, 8)
    bits = x.view(np.uint32)
    exp = ((bits >> 23) & 0xFF).max(axis=1).astype(np.int32)
    scale = np.ldexp(1.0, exp - 127) / 64.0
    m = np.clip(np.rint(x / scale[:, None]), -128, 127).astype(np.int8)
    out = np.empty((x.shape[0], 9), np.uint8)
    out[:, 0] = exp.astype(np.uint8)
    out[:, 1:] = m.view(np.uint8)
    return out.reshape(-1), (m.astype(np.float32) * scale[:, None]).reshape(-1)


def main():
    xclbin, insts, gguf = sys.argv[1:4]
    M, K, N = map(int, sys.argv[4:7])
    d, info, base = gguf_tensors(gguf)
    gc, gs = q8_rows(d, info, base, "blk.0.ffn_gate.weight")
    uc, us = q8_rows(d, info, base, "blk.0.ffn_up.weight")
    deq = lambda c, s: (c.astype(np.float32).reshape(c.shape[0], -1, 32) * s[:, :, None]).reshape(c.shape[0], -1)
    Wt = np.zeros((N, K), np.float32)
    rows = np.concatenate([deq(gc, gs), deq(uc, us)])[:N]
    Wt[:rows.shape[0]] = rows
    B = Wt.T.copy()                                   # K x N
    enc, _ = to_bfp16(shuffle_b(B))
    rng = np.random.default_rng(3)
    A = rng.normal(0, 1, (M, K)).astype(np.float32)
    Ab = bf16(A)

    dev = pyxrt.device(0)
    xb = pyxrt.xclbin(xclbin)
    dev.register_xclbin(xb)
    ctx = pyxrt.hw_context(dev, xb.get_uuid())
    kern = pyxrt.kernel(ctx, [k.get_name() for k in xb.get_kernels() if k.get_name().startswith("MLIR_AIE")][0])
    ins = np.fromfile(insts, dtype=np.uint32)
    to_dev, from_dev = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
    bo_i = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, kern.group_id(1))
    bo_a = pyxrt.bo(dev, Ab.nbytes, pyxrt.bo.host_only, kern.group_id(3))
    bo_b = pyxrt.bo(dev, enc.nbytes, pyxrt.bo.host_only, kern.group_id(4))
    bo_c = pyxrt.bo(dev, M * N * 2, pyxrt.bo.host_only, kern.group_id(5))
    for bo, arr in ((bo_i, ins), (bo_a, Ab), (bo_b, enc)):
        bo.write(arr.tobytes(), 0)
        bo.sync(to_dev)
    kern(3, bo_i, ins.size, bo_a, bo_b, bo_c).wait()
    t0 = time.perf_counter()
    for _ in range(10):
        kern(3, bo_i, ins.size, bo_a, bo_b, bo_c).wait()
    dt = (time.perf_counter() - t0) / 10
    bo_c.sync(from_dev)
    C = from_bf16(np.frombuffer(bo_c.read(M * N * 2, 0), np.uint16)).reshape(M, N)
    ref = from_bf16(Ab).reshape(M, K).astype(np.float64) @ B.astype(np.float64)
    rel = np.linalg.norm(C - ref) / np.linalg.norm(ref)
    print(f"{M}x{K}x{N}: {dt * 1e3:.2f} ms ({2 * M * K * N / dt / 1e12:.1f} TFLOPS), rel err vs exact weights {rel:.2e}")


if __name__ == "__main__":
    main()
