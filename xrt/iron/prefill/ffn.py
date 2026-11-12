# n32_core.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""32-core (8 cols x 4 rows) asymmetric-tile-buffered bf16/bfp16 GEMM —
``@iron.jit`` IRON design.

Each AIE2P core's L1 A-tile is (m//4, k); the L2 A buffer pre-fetches
``mtk = 512`` columns at a time and the memtile fan-out splits it into
four A-subtiles per matmul cycle. B is bfp16ebs8; C is bf16. Strix-only.
"""

import argparse
from pathlib import Path

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import (
    add_compile_args,
    device_from_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from ml_dtypes import bfloat16

_KERNEL_SRC = Path(__file__).resolve().parent / "mm_bfp_mixed.cc"
_AIE_KERNELS_INC = Path("/home/anvil/npu-work/mlir-aie-src/aie_kernels")


from aie.iron.device import from_name
@iron.jit(aiecc_flags=["--dynamic-objFifos"])
def chain_gemm(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = 1024,
    shapes: CompileTime[tuple] = ((2048, 15360, "swiglu"), (7680, 2048, "act")),
    m: CompileTime[int] = 128,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 128,
):
    n_aie_cols = 8
    n_aie_rows = 4
    r, s, t = 8, 8, 8
    mtk = 512

    A_l2_ty = np.ndarray[(m, mtk), np.dtype[bfloat16]]
    B_l2_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[bfloat16]]
    A_l1_ty = np.ndarray[(m // 4, k), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[bfloat16]]

    kernel_flags = [f"-I{_AIE_KERNELS_INC}"]

    zero_kernel = kernels.zero((m, n), bfloat16)
    matmul_kernel = ExternalFunction(
        "matmul_vectorized_different_datatypes",
        source_file=str(_KERNEL_SRC),
        arg_types=[A_l1_ty, B_l1_ty, C_l1_ty],
        compile_flags=kernel_flags,
        object_file_name="mm_prefill.o",
    )

    A_l3l2_fifos: list[ObjectFifo] = []
    A_l2l1_fifos: list[ObjectFifo] = []
    B_l3l2_fifos: list[ObjectFifo] = []
    B_l2l1_fifos: list[ObjectFifo] = []
    C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(n_aie_rows)]
    C_l2l3_fifos: list[ObjectFifo] = []

    # Input A: 4 shim-rows, each shim→memtile carries an L2 strip and
    # the memtile then fans out the L1 sub-tiles to all 8 cores in that row.
    a_l3l2_dims = TensorAccessPattern.full((mtk // k, m, k)).permute((1, 0, 2))
    a_l2l1_in_dims = TensorAccessPattern.full(
        (mtk // k * 4, m // 4, k // s, s)
    ).permute((0, 2, 1, 3))
    a_l2l1_out_dims = TensorAccessPattern.full((m // 4 // r, k // s, r * s)).permute(
        (1, 0, 2)
    )

    for row in range(n_aie_rows):
        a_l3l2 = ObjectFifo(
            A_l2_ty,
            name=f"A_L3L2_{row}",
            depth=2,
            from_stream_per_cons=a_l3l2_dims,
        )
        A_l3l2_fifos.append(a_l3l2)
        # forward() emits an ObjectFifoLink at the memtile. The new
        # (A_l2l1) fifo's producer-side `to_stream` is the memtile
        # TX layout, and per-cons `from_stream` is the layout each
        # of the 8 compute-tile consumers reads from the stream.
        A_l2l1_fifos.append(
            a_l3l2.cons().forward(
                obj_type=A_l1_ty,
                name=f"A_L2L1_{row}",
                depth=2,
                to_stream=a_l2l1_in_dims,
                from_stream=a_l2l1_out_dims,
            )
        )

    # Input B: 8 cols, each shim→memtile→fan to 4 rows of cores.
    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=2)
        B_l3l2_fifos.append(b_l3l2)
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", depth=2)
        )

    # Output C: per-col, 4 cores join at memtile, memtile→shim with the
    # final layout transform.
    c_l2l3_dims = TensorAccessPattern.full((m, n)).tile((r, t)).inverse()
    for col in range(n_aie_cols):
        c_l2l3 = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=2, to_stream=c_l2l3_dims
        )
        C_l2l3_fifos.append(c_l2l3)
        of_offsets = [m * n * i for i in range(n_aie_rows)]
        c_tmp_fifos = c_l2l3.prod().join(
            of_offsets,
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[1] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j].append(c_tmp_fifos[j])

    swiglu_kernel = ExternalFunction("prefill_swiglu", source_file=str(_KERNEL_SRC), arg_types=[C_l1_ty],
                                     compile_flags=kernel_flags, object_file_name="mm_prefill.o")

    def core_fn(in_a, in_b, out_c, zero, matmul, swiglu):
        for (K, N, mode) in shapes:
            tiles_per_core = (M // m) * (N // n) // (n_aie_cols * n_aie_rows)
            for _ in range_(tiles_per_core) if tiles_per_core > 1 else range(1):
                elem_out = out_c.acquire(1)
                zero(elem_out)
                for _ in range_(K // k):
                    elem_in_b = in_b.acquire(1)
                    for _ in range(4):
                        elem_in_a = in_a.acquire(1)
                        matmul(elem_in_a, elem_in_b, elem_out)
                        in_a.release(1)
                    in_b.release(1)
                if mode == "swiglu":
                    swiglu(elem_out)
                out_c.release(1)

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
                swiglu_kernel,
            ],
            stack_size=0xA00,
        ),
    )

    a_size = sum(M * K for K, _, mode in shapes if mode != "act")
    b_size = sum(K * N // 8 for K, N, _ in shapes)
    c_size = sum(M * N for _, N, _ in shapes)
    A_ty = np.ndarray[(a_size,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(b_size,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(c_size,), np.dtype[bfloat16]]
    tb_max_n_rows = 4

    def sequence(a, b, c, A_prods, B_prods, C_conses):
        a_off = b_off = c_off = 0
        prev_c, prev_n = 0, 0
        for (K, N, mode) in shapes:
            num_row_tile = M // m // n_aie_rows
            num_col_tile = N // n // n_aie_cols
            num_groups = num_row_tile * num_col_tile
            slots = [None] * tb_max_n_rows
            done = [True] * tb_max_n_rows
            def fin(i):
                if not done[i]:
                    slots[i].finish()
                    done[i] = True
            for group_idx in range(num_groups):
                slot_idx = group_idx % tb_max_n_rows
                fin(slot_idx)
                tg = TaskGroup()
                slots[slot_idx] = tg
                done[slot_idx] = False
                a_base_idx = (group_idx // num_col_tile) * n_aie_rows
                for row in range(n_aie_rows):
                    t = a_base_idx + row
                    if mode == "act":
                        # SwiGLU output: the gate half of each 128-column tile of the previous C.
                        A_prods[row].fill(c, tap=TensorAccessPattern((c_size,), prev_c + t * m * prev_n,
                                                                     [K // mtk, m, mtk // 64, 64], [mtk * 2, prev_n, 128, 1]),
                                          group=tg, wait=False)
                    else:
                        A_prods[row].fill(a, tap=TensorAccessPattern((a_size,), a_off + t * m * K, [1, K // mtk, m, mtk], [0, mtk, K, 1]),
                                          group=tg, wait=False)
                b_base_idx = (group_idx % num_col_tile) * n_aie_cols
                for col in range(n_aie_cols):
                    t = b_base_idx + col
                    B_prods[col].fill(b, tap=TensorAccessPattern((b_size,), b_off + t * (n * K // 8), [1, 1, 1, n * K // 8], [0, 0, 0, 1]),
                                      group=tg, wait=False)
                for col in range(n_aie_cols):
                    rt_ = group_idx // num_col_tile
                    ct = b_base_idx + col
                    C_conses[col].drain(c, tap=TensorAccessPattern((c_size,), c_off + rt_ * n_aie_rows * m * N + ct * n,
                                                                   [1, 1, n_aie_rows * m, n], [0, 0, N, 1]),
                                        group=tg, wait=True)
                if slot_idx == 1 and group_idx != 1:
                    fin(2); fin(3)
                if slot_idx == 3:
                    fin(0); fin(1)
            for i in range(tb_max_n_rows):
                fin(i)
            prev_c, prev_n = c_off, N
            if mode != "act":
                a_off += M * K
            b_off += K * N // 8
            c_off += M * N

    rt = Runtime(
        sequence,
        [A_ty, B_ty, C_ty,
         [f.prod() for f in A_l3l2_fifos],
         [f.prod() for f in B_l3l2_fifos],
         [f.cons() for f in C_l2l3_fifos]],
    )

    return Program(
        from_name("npu2", n_cols=None),
        rt,
        workers=[w for row in workers for w in row],
    ).resolve_program()




if __name__ == "__main__":
    import sys
    from aie.iron.device import from_name  # noqa
    out = sys.argv[1]
    iron.set_current_device(from_name("npu2", n_cols=None))
    chain_gemm.specialize(M=1024).compile(xclbin_path=out + ".xclbin", inst_path=out + ".bin")
