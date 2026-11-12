"""LFM3 prefill as one IRON design: a 32-core (8 columns x 4 rows) bf16 x bfp16ebs8
GEMM (mlir-aie's asymmetric-tile-buffered config1) whose cores also run the
elementwise phases of the plan in plan.py, so a whole prompt chunk runs in one
dispatch.

Phase data paths reuse the GEMM's FIFOs: A (per core row, broadcast across the
8 columns) carries small constants as L1 objects of one 128 x 512 block; B (per
column, broadcast down the 4 rows) carries token rows; C (each core, joined per
column) carries results. The C tile has depth 1, so it is the same memory on
every acquire; the conv phase keeps state in the half of it that its drain
discards.

Usage: model.py <out-prefix> <kinds, e.g. conv,conv>
"""

import os
import sys
from pathlib import Path

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, ExternalFunction, In, InOut, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from ml_dtypes import bfloat16

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plan as P  # noqa: E402

_KERNEL_SRC = Path(__file__).resolve().parent / "mm_bfp_mixed.cc"
_AIE_KERNELS_INC = Path("/home/anvil/npu-work/mlir-aie-src/aie_kernels")
COLS, ROWS = 8, 4


@iron.jit(aiecc_flags=["--dynamic-objFifos"] + os.environ.get("PREFILL_AIECC", "").split())
def prefill(A: InOut, B: In, C: InOut, *, kinds: CompileTime[str] = "conv", M: CompileTime[int] = 1024,
            head_row0: CompileTime[int] = -1):
    plan = P.Plan(kinds.split(","), M, None if head_row0 < 0 else head_row0)
    m, k, n, mtk = P.M_TILE, P.K_TILE, P.N_TILE, P.MTK
    r, s, t = 8, 8, 8

    A_l2_ty = np.ndarray[(m, mtk), np.dtype[bfloat16]]
    B_l2_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(ROWS * m, n), np.dtype[bfloat16]]
    A_l1_ty = np.ndarray[(m // 4, k), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[bfloat16]]

    flags = [f"-I{_AIE_KERNELS_INC}", f"-DPREFILL_PLAN={','.join(map(str, plan.codes()))}", f"-DPREFILL_ROUNDS={M // 64}"]
    flags += os.environ.get("PREFILL_FLAGS", "").split()
    obj = "mm_prefill.o"
    def ext(name, args):
        return ExternalFunction(name, source_file=str(_KERNEL_SRC), arg_types=args, compile_flags=flags,
                                object_file_name=obj)
    zero_k = kernels.zero((m, n), bfloat16)
    matmul_k = ext("matmul_vectorized_different_datatypes", [A_l1_ty, B_l1_ty, C_l1_ty])
    cnt_ty = np.ndarray[(8,), np.dtype[np.int32]]
    next_k = ext("prefill_next", [cnt_ty])
    epi_k = ext("prefill_epilogue", [C_l1_ty])
    norm_k = ext("prefill_row", [A_l1_ty, B_l1_ty, C_l1_ty, np.int32])
    taps_k = ext("prefill_conv_taps", [A_l1_ty, C_l1_ty])
    conv_k = ext("prefill_conv_row", [B_l1_ty, C_l1_ty, np.int32])
    cgrp_k = ext("prefill_conv_group", [cnt_ty])
    rnd_k = ext("prefill_attn_round", [cnt_ty, C_l1_ty, np.int32, np.int32])
    aq_k = ext("prefill_attn_q", [B_l1_ty, C_l1_ty, np.int32])
    akv_k = ext("prefill_attn_kv", [B_l1_ty, C_l1_ty])
    aout_k = ext("prefill_attn_out", [C_l1_ty])

    a_l3l2_dims = TensorAccessPattern.full((mtk // k, m, k)).permute((1, 0, 2))
    a_l2l1_in = TensorAccessPattern.full((mtk // k * 4, m // 4, k // s, s)).permute((0, 2, 1, 3))
    a_l2l1_out = TensorAccessPattern.full((m // 4 // r, k // s, r * s)).permute((1, 0, 2))
    A_l3l2, A_l2l1, B_l3l2, B_l2l1, C_l2l3 = [], [], [], [], []
    C_l1l2 = [[] for _ in range(ROWS)]
    for row in range(ROWS):
        f = ObjectFifo(A_l2_ty, name=f"A_L3L2_{row}", depth=2, from_stream_per_cons=a_l3l2_dims)
        A_l3l2.append(f)
        A_l2l1.append(f.cons().forward(obj_type=A_l1_ty, name=f"A_L2L1_{row}", depth=2, to_stream=a_l2l1_in,
                                       from_stream=a_l2l1_out))
    for col in range(COLS):
        f = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=2)
        B_l3l2.append(f)
        B_l2l1.append(f.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", depth=2))
    c_dims = TensorAccessPattern.full((m, n)).tile((r, t)).inverse()
    for col in range(COLS):
        f = ObjectFifo(C_l2_ty, name=f"C_L2L3_{col}", depth=2, to_stream=c_dims)
        C_l2l3.append(f)
        parts = f.prod().join([m * n * i for i in range(ROWS)], obj_types=[C_l1_ty] * ROWS,
                              names=[f"C_L1L2_{col}_{row}" for row in range(ROWS)], depths=[1] * ROWS)
        for j in range(ROWS):
            C_l1l2[j].append(parts[j])

    rows_per_col = M // COLS
    groups = rows_per_col // 16

    rounds = M // 32 // 2

    def make_core(q, col):
        def core_fn(in_a, in_b, out_c, cnt, zero, matmul, nxt, epi, norm, taps, conv, cgrp, rnd, aq, akv, aout):
            def skip_a(nobj):
                for _ in range_(nobj):
                    in_a.acquire(1)
                    in_a.release(1)

            def rows16(o, fn):
                # the kernels count the row slot themselves
                for _ in range_(16):
                    row = in_b.acquire(1)
                    fn(row, o)
                    in_b.release(1)

            for _ in range_(len(plan.phases)):
                nxt(cnt)
                # GEMM: cnt[0] C tiles of cnt[1] k steps, then the phase's epilogue
                for _ in range_(cnt[0]):
                    acc = out_c.acquire(1)
                    zero(acc)
                    for _ in range_(cnt[1]):
                        wb = in_b.acquire(1)
                        for _ in range_(4):
                            xa = in_a.acquire(1)
                            matmul(xa, wb, acc)
                            in_a.release(1)
                        in_b.release(1)
                    epi(acc)
                    out_c.release(1)
                # Row phases (norm, or qk-norm + RoPE for qkr; the kernel dispatches):
                # Norm: the post-norm weight is the first A object of a block. Every core of a
                # column sees each row; core q writes quarter q of [xn | x1] for 16 rows per C object.
                for _ in range_(cnt[2]):
                    w = in_a.acquire(1)
                    for _ in range_(groups):
                        o = out_c.acquire(1)
                        rows16(o, lambda row, o: norm(w, row, o, q))
                        out_c.release(1)
                    in_a.release(1)
                    skip_a(mtk // k * 4 - 1)
                # Conv, per C object of 16 tokens: the first also takes the taps block (the
                # kernel copies its first two A objects) and the 6 state rows.
                for _ in range_(cnt[3]):
                    for _ in range_(groups):
                        o = out_c.acquire(1)
                        cgrp(cnt)
                        for _ in range_(cnt[7]):
                            tp = in_a.acquire(1)
                            taps(tp, o)
                            in_a.release(1)
                        for _ in range_(cnt[6]):
                            row = in_b.acquire(1)
                            conv(row, o, q)
                            in_b.release(1)
                        out_c.release(1)
                # Attention: per round, the group's four Q objects, then the round's KV objects.
                for _ in range_(cnt[5]):
                    for _ in range_(rounds):
                        o = out_c.acquire(1)
                        rnd(cnt, o, q, col)
                        for _ in range_(4):
                            qo = in_b.acquire(1)
                            aq(qo, o, q)
                            in_b.release(1)
                        for _ in range_(cnt[6]):
                            kv = in_b.acquire(1)
                            akv(kv, o)
                            in_b.release(1)
                        aout(o)
                        out_c.release(1)
        return core_fn

    workers = [Worker(make_core(row, col), [A_l2l1[row].cons(), B_l2l1[col].cons(), C_l1l2[row][col].prod(),
                                            Buffer(cnt_ty, name=f"cnt_{row}_{col}"), zero_k, matmul_k, next_k, epi_k,
                                            norm_k, taps_k, conv_k, cgrp_k, rnd_k, aq_k, akv_k, aout_k], stack_size=0x1100)
               for row in range(ROWS) for col in range(COLS)]

    a_size, b_size, c_size = M * P.REC, plan.b_size, plan.c_size
    A_ty = np.ndarray[(a_size,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(b_size,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(c_size,), np.dtype[bfloat16]]
    TAP = TensorAccessPattern

    def sequence(a, b, c, A_prods, B_prods, C_conses):
        size = {"a": a_size, "c": c_size}
        buf = {"a": a, "c": c}

        def elementwise(a_fills, b_fills, drains):
            tg = TaskGroup()
            for row, (src, off) in enumerate(a_fills):
                A_prods[row].fill(buf[src], tap=TAP((size[src],), off, [1, 1, 1, P.ABLOCK], [0, 0, 0, 1]), group=tg, wait=False)
            for col, (src, tap) in enumerate(b_fills):
                B_prods[col].fill(buf[src], tap=TAP((size[src],), *tap), group=tg, wait=False)
            for col, (dst, tap) in enumerate(drains):
                C_conses[col].drain(buf[dst], tap=TAP((size[dst],), *tap), group=tg, wait=True)
            tg.finish()

        for ph in plan.phases:
            if ph["kind"] == "norm":
                elementwise(
                    [("c", ph["w"])] * ROWS,
                    # [x | o | 512 more] of each record, as 9 x 512
                    [("a", (col * rows_per_col * P.REC + P.H, [1, rows_per_col, 9, 512], [0, P.REC, 512, 1])) for col in range(COLS)],
                    [("a", (col * rows_per_col * P.REC, [groups, 4, 16, 1024], [16 * P.REC, 1024, P.REC, 1])) for col in range(COLS)])
                continue
            if ph["kind"] == "conv":
                elementwise(
                    [("c", off) for off in ph["taps"]],
                    # 6 state rows before the column's first token, then its tokens
                    [("c", (ph["cs"] + col * rows_per_col * P.CONV_ROW, [1, rows_per_col + P.STATE, 9, 512],
                            [0, P.CONV_ROW, 512, 1])) for col in range(COLS)],
                    # per C object of 16 tokens: core q's 512 channels, then 512 discarded
                    # (overwritten by core q + 1, or the row's padding for q = 3)
                    [("c", (ph["ys"] + col * rows_per_col * P.Y_ROW, [groups, 4, 16, 1024], [16 * P.Y_ROW, 512, P.Y_ROW, 1]))
                     for col in range(COLS)])
                continue
            if ph["kind"] == "qkr":
                per_col = rows_per_col // 16 * 4 * P.PIECE
                elementwise(
                    [("c", ph["w"])] * ROWS,
                    [("c", (ph["qs"] + col * rows_per_col * P.QKV_ROW, [1, rows_per_col, 9, 512], [0, P.QKV_ROW, 512, 1]))
                     for col in range(COLS)],
                    # pieces in [column][16-token object][core] order
                    [("c", (ph["piece"] + col * per_col, [1, 1, per_col // 1024, 1024], [0, 0, 1024, 1])) for col in range(COLS)])
                continue
            if ph["kind"] == "attn":
                oh = plan.oh
                for r_ in range(rounds):
                    tg = TaskGroup()
                    for col in range(COLS):
                        g, b_ = col % 4, 2 * r_ + col // 4
                        k0 = max(0, (32 * b_ - 255) // 16) if ph["mode"] else 0
                        nkb = 2 * b_ + 2 - k0
                        # the group's four heads: [Q 16 tokens | gates] for both 16-token halves
                        B_prods[col].fill(c, tap=TAP((c_size,), ph["piece"] + (2 * b_ * 4 + g) * P.PIECE, [4, 2, 9, 256],
                                                     [2304, 4 * P.PIECE, 256, 1]), group=tg, wait=False)
                        # KV objects: the two diagonal blocks first (they set the softmax offset), then k0 .. 2 b - 1
                        for kb0, n_ in ((2 * b_, 2), (k0, 2 * b_ - k0)):
                            if n_:
                                B_prods[col].fill(c, tap=TAP((c_size,), ph["piece"] + (kb0 * 4 + g) * P.PIECE + 9216, [1, n_, 9, 512],
                                                             [0, 4 * P.PIECE, 512, 1]), group=tg, wait=False)
                    for col in range(COLS):
                        g, b_ = col % 4, 2 * r_ + col // 4
                        # per core: three junk quarters (into the junk copies), then O [32 tokens][128]
                        C_conses[col].drain(c, tap=TAP((c_size,), ph["ao"] + 4 * g * (M * 128) + b_ * 4096 - 3 * oh,
                                                       [4, 4, 32, 128], [M * 128, oh, 128, 1]), group=tg, wait=True)
                    tg.finish()
                continue
            K, N, mode = ph["K"], ph["N"], ph["mode"]
            n_rt, n_ct = ph["rows"] // m // ROWS, N // n // COLS
            slots, done = [None] * 4, [True] * 4

            def fin(i):
                if not done[i]:
                    slots[i].finish()
                    done[i] = True
            for g in range(n_rt * n_ct):
                si = g % 4
                fin(si)
                tg = TaskGroup()
                slots[si], done[si] = tg, False
                rt = g // n_ct
                for row in range(ROWS):
                    tr = rt * ROWS + row
                    src = ph["a"]
                    if src[0] == "rec":
                        row_base = src[1] if len(src) > 1 else 0
                        A_prods[row].fill(a, tap=TAP((a_size,), (row_base + tr * m) * P.REC, [1, K // mtk, m, mtk], [0, mtk, P.REC, 1]), group=tg, wait=False)
                    elif src[0] == "heads":
                        A_prods[row].fill(c, tap=TAP((c_size,), src[1] + tr * m * 128, [K // mtk, m, mtk // 128, 128],
                                                     [mtk // 128 * M * 128, 128, M * 128, 1]), group=tg, wait=False)
                    elif src[0] == "plain":
                        A_prods[row].fill(c, tap=TAP((c_size,), src[1] + tr * m * src[2], [1, K // mtk, m, mtk], [0, mtk, src[2], 1]), group=tg, wait=False)
                    else:  # half
                        A_prods[row].fill(c, tap=TAP((c_size,), src[1] + tr * m * src[2], [K // mtk, m, mtk // 64, 64], [mtk * 2, src[2], 128, 1]), group=tg, wait=False)
                for col in range(COLS):
                    ct = (g % n_ct) * COLS + col
                    B_prods[col].fill(b, tap=TAP((b_size,), ph["b"] + ct * (n * K // 8), [1, 1, 1, n * K // 8], [0, 0, 0, 1]), group=tg, wait=False)
                for col in range(COLS):
                    ct = (g % n_ct) * COLS + col
                    dst = ph["c"]
                    row0 = rt * ROWS * m
                    if dst[0] == "o":
                        C_conses[col].drain(a, tap=TAP((a_size,), row0 * P.REC + 2 * P.H + ct * n, [1, 1, ROWS * m, n], [0, 0, P.REC, 1]), group=tg, wait=True)
                    elif dst[0] == "plain":
                        C_conses[col].drain(c, tap=TAP((c_size,), dst[1] + row0 * dst[2] + ct * n, [1, 1, ROWS * m, n], [0, 0, dst[2], 1]), group=tg, wait=True)
                    else:  # halves: two 64-channel blocks into columns 64..127 of their 128
                        C_conses[col].drain(c, tap=TAP((c_size,), dst[1] + row0 * dst[2] + ct * 2 * n + 64, [1, ROWS * m, 2, 64], [0, dst[2], 128, 1]), group=tg, wait=True)
                if si == 1 and g != 1:
                    fin(2)
                    fin(3)
                if si == 3:
                    fin(0)
                    fin(1)
            for i in range(4):
                fin(i)

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, [f.prod() for f in A_l3l2], [f.prod() for f in B_l3l2], [f.cons() for f in C_l2l3]])
    return Program(from_name("npu2", n_cols=None), rt, workers=workers).resolve_program()


if __name__ == "__main__":
    out = sys.argv[1]
    kinds = sys.argv[2] if len(sys.argv) > 2 else "conv"
    if kinds.isdigit():
        kinds = ",".join(P.lfm3_kinds(int(kinds)))
    M = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    head_row0 = int(sys.argv[4]) if len(sys.argv) > 4 else -1
    iron.set_current_device(from_name("npu2", n_cols=None))
    prefill.specialize(kinds=kinds, M=M, head_row0=head_row0).compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
