//===- mm_bfp_mixed.cc ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ATB microkernel, bf16 A / bfp16ebs8 B / bf16 C variant. A is converted to
// bfp16ebs8 in a stack staging buffer first (sizeof(bfp16ebs8) == 1 on Peano,
// so all bfp16ebs8* arithmetic below is byte arithmetic), then the MAC loop
// streams A and B as 576-bit blocks while C is plain bf16 load/store.
// Register placement uses DM-bank annotations: A -> bank A, B -> bank B,
// C -> bank C, raw A input -> bank D.
//
// Compute structure: per call, 2 block rows x 8 blocks of 16x16; each block
// runs 8 expanded k-steps (2 A pops + 2 B pops + 4 8x8x8 MACs, B-first at
// fill steps 0 and 4) in a post-RA-pipelined blk loop with next-block C
// prefetch. Measured II=42, NS=2.

#include <aie_api/aie.hpp>

#include "aie_kernel_utils.h"

template <typename T, unsigned N, aie_dm_resource R>
using InBufStream = aie::block_vector_restrict_input_buffer_stream<T, N, R>;
template <typename T, unsigned N, aie_dm_resource R>
using OutBufStream = aie::block_vector_output_buffer_stream<T, N, R>;

// Per-core L1 C tile is (m, n); the L1 A sub-tile is (m/rho, k). The kernel
// is invoked rho times per (A-sub-tile, B-tile) pair and rotates over the
// rho C stripes (m/rho rows each) via g_counter.
constexpr int rho = 4;
constexpr int m = 128;
constexpr int k = 64;
constexpr int n = 128;
constexpr int m_a = m / rho;

// aie::mmul sub-tile shape (the only shape supported for bfp16ebs8 on
// AIE2P): per MAC, an r x s A panel times an s x t B panel accumulates
// into r x t of C.
constexpr int r = 8; // MAC rows (A/C)
constexpr int s = 8; // MAC reduction step (k-columns per MAC)
constexpr int t = 8; // MAC columns (B/C)

static_assert(m * rho == 512 && k == 64 && n == 128 && rho == 4);

extern "C" {

static int g_counter = 0;

void matmul_vectorized_different_datatypes(bfloat16 *__restrict pA_in,
                                           bfp16ebs8 *__restrict pB_in,
                                           bfloat16 *__restrict pC_curtile) {
  event0();
  // Round-to-nearest-even (Peano defaults to floor on converts).
  aie::set_rounding(aie::rounding_mode::conv_even);
  bfloat16 *__restrict pC_tile = pC_curtile + g_counter * (m_a * n);
  g_counter = (g_counter == rho - 1) ? 0 : g_counter + 1;

  uint8_t converted_A[m_a * k / 8 * 9] __attribute__((aligned(64)));

  OutBufStream<bfp16ebs8, r * s, aie_dm_resource::a> a_out(
      (bfp16ebs8 *)converted_A);
  const bfloat16 __aie_dm_resource_d *__restrict pA =
      (const bfloat16 __aie_dm_resource_d *)pA_in;

  for (int rp = 0; rp < m_a / (2 * r); rp++) {
    const bfloat16 __aie_dm_resource_d *__restrict q = pA + rp * (2 * r * k);
    AIE_PREPARE_FOR_POSTPIPELINING
    for (int cb = 0; cb < k / s; cb++) {
      aie::vector<bfloat16, r * s> v0 = aie::load_v<r * s>(q);
      aie::vector<bfloat16, r * s> v1 = aie::load_v<r * s>(q + r * k);
      a_out.push(
          aie::accum<accfloat, r * s>(v0).template to_vector<bfp16ebs8>());
      a_out.push(
          aie::accum<accfloat, r * s>(v1).template to_vector<bfp16ebs8>());
      q += r * s;
    }
  }

  using AStream = InBufStream<bfp16ebs8, r * s, aie_dm_resource::a>;
  using BStream = InBufStream<bfp16ebs8, s * t, aie_dm_resource::b>;
  using CPtr = bfloat16 __aie_dm_resource_c *__restrict;
  using MMUL = aie::mmul<r, s, t, bfp16ebs8>;

  const bfp16ebs8 *pA_base = (const bfp16ebs8 *)converted_A;
  CPtr const pC_base = (CPtr)pC_tile;

  for (int br = 0; br < m_a / (2 * r); br++) {
    // Persistent sliding B stream per block row.
    BStream b_stream(pB_in);
    {
      // C layout: r x t row-major sub-blocks over the (m, n) tile; the
      // +r*n offsets below step to the second r-row half (rows r..2r-1)
      // of the 2r-row block.
      CPtr pC = pC_base + br * (2 * r * n);

      aie::vector<bfloat16, r * t> c0 = aie::load_v<r * t>(pC);
      aie::vector<bfloat16, r * t> c1 = aie::load_v<r * t>(pC + r * t);
      for (int blk = 0; blk < n / (2 * t); blk++) {
        CPtr pCb = pC + blk * (2 * r * t);
        CPtr pNext = (blk < n / (2 * t) - 1) ? pCb + 2 * r * t : pCb;

        MMUL acc0((aie::accum<accfloat, r * t>(c0)));
        MMUL acc1((aie::accum<accfloat, r * t>(c1)));
        MMUL acc2(
            (aie::accum<accfloat, r * t>(aie::load_v<r * t>(pCb + r * n))));
        MMUL acc3((aie::accum<accfloat, r * t>(
            aie::load_v<r * t>(pCb + r * t + r * n))));

        // Prefetch next block's first C pair while this block computes.
        c0 = aie::load_v<r * t>(pNext);
        c1 = aie::load_v<r * t>(pNext + r * t);

        // Fresh A stream per block: re-reads this block row's band (fills
        // constant-fold before pops 0 and 8).
        AStream a_stream(pA_base);
        a_stream.seek(br * (2 * k / s));

        // One k-step: 2 A pops + 2 B pops + 4 8x8x8 MACs. On fill steps
        // (B pops 0 and 8 of each 16-pop stream buffer) the B pops come
        // first so the fifo_ld_fill emits ahead of the A-stream ops.
        auto kstep = [&]<bool BFILL>() __attribute__((always_inline)) {
          aie::block_vector<bfp16ebs8, r * s> A0, A1; // r x s A panels
          aie::block_vector<bfp16ebs8, s * t> B0, B1; // s x t B panels
          if constexpr (BFILL) {
            B0 = b_stream.pop();
            B1 = b_stream.pop();
            A0 = a_stream.pop();
            A1 = a_stream.pop();
          } else {
            A0 = a_stream.pop();
            A1 = a_stream.pop();
            B0 = b_stream.pop();
            B1 = b_stream.pop();
          }
          acc0.mac(A0, aie::op_transpose(B0));
          acc1.mac(A0, aie::op_transpose(B1));
          acc2.mac(A1, aie::op_transpose(B0));
          acc3.mac(A1, aie::op_transpose(B1));
        };
        kstep.operator()<true>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<true>();
        kstep.operator()<false>();
        kstep.operator()<false>();
        kstep.operator()<false>();

        aie::store_v(pCb, acc0.template to_vector<bfloat16>());
        aie::store_v(pCb + r * t, acc1.template to_vector<bfloat16>());
        aie::store_v(pCb + r * n, acc2.template to_vector<bfloat16>());
        aie::store_v(pCb + r * t + r * n, acc3.template to_vector<bfloat16>());
      }
    }
  }

  event1();
}
}

// ---- LFM3 prefill epilogues (Gufo).

extern "C" {

// In place on an m x n = 128 x 128 C tile held as 8x8 row-major blocks,
// block (bi, bj) at (bi * 16 + bj) * 64: columns 0..63 are gate, 64..127 up
// of the same 64 channels. Writes SiLU(gate) * up into the gate blocks:
// g * u * (1 + tanh(g / 2)) / 2.
void prefill_swiglu(bfloat16 *__restrict c) {
  const aie::vector<float, 16> half = aie::broadcast<float, 16>(0.5f);
  const aie::vector<float, 16> one = aie::broadcast<float, 16>(1.0f);
  for (int bi = 0; bi < 16; ++bi)
    for (int bj = 0; bj < 8; ++bj)
      for (int e = 0; e < 64; e += 16) {
        bfloat16 *g = c + (bi * 16 + bj) * 64 + e;
        const bfloat16 *u = c + (bi * 16 + bj + 8) * 64 + e;
        aie::accum<accfloat, 16> ga, ua, th;
        ga.from_vector(aie::load_v<16>(g));
        ua.from_vector(aie::load_v<16>(u));
        const aie::vector<float, 16> gf = ga.to_vector<float>();
        th.from_vector(aie::tanh<bfloat16>(aie::mul(gf, half).to_vector<float>()));
        const aie::vector<float, 16> sig = aie::mul(aie::add(th.to_vector<float>(), one), half).to_vector<float>();
        aie::accum<accfloat, 16> a = aie::mul(aie::mul(gf, ua.to_vector<float>()).to_vector<float>(), sig);
        aie::store_v(g, a.to_vector<bfloat16>());
      }
}

}  // extern "C"

// ---- Prefill residual/norm phase. A token row's DDR record is
// [xn bf16 2048 | x bf16 2048 | o bf16 2048 | pad] (16 KB). The phase gets
// [w_post | w_pre] (bf16) once, then each row's [x | o] as one B object,
// computes x += rmsnorm(o) * w_post and xn = rmsnorm(x) * w_pre, and writes
// [xn | x] for four rows into a C tile. The C tile drains through the
// GEMM's 8x8-tile transform, so flat element i goes to tiled position T(i).

namespace {
inline int Tiled(int i) {
  const int r = i >> 7, c = i & 127;
  return (((r >> 3) << 4) + (c >> 3)) * 64 + ((r & 7) << 3) + (c & 7);
}
inline aie::vector<float, 16> Wide(const bfloat16 *p) {
  aie::accum<accfloat, 16> a;
  a.from_vector(aie::load_v<16>(p));
  return a.to_vector<float>();
}
inline float InvRms(float sum) {
  aie::vector<float, 16> v = aie::mul(aie::broadcast<float, 16>(sum), aie::broadcast<float, 16>(1.0f / 2048.0f)).to_vector<float>();
  return aie::invsqrt(aie::add(v, aie::broadcast<float, 16>(1e-5f)))[0];
}
inline float SumSqBf16(const bfloat16 *v) {
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
  for (int i = 0; i < 2048; i += 16) { aie::vector<float, 16> x = Wide(v + i); acc = aie::mac(acc, x, x); }
  return aie::reduce_add(acc.to_vector<float>());
}
inline void Put16(bfloat16 *tile, int at, aie::vector<bfloat16, 16> v) {
  aie::store_v(tile + Tiled(at), v.extract<8>(0));
  aie::store_v(tile + Tiled(at + 8), v.extract<8>(1));
}
}  // namespace

extern "C" {

// w: [w_post | w_pre]; row: [x | o] (updated in place); slot 0..3 of the tile.
void prefill_norm_row(const bfloat16 *__restrict w, bfloat16 *__restrict row, bfloat16 *__restrict tile, int32_t slot) {
  bfloat16 *x = row;
  const bfloat16 *o = row + 2048;
  const aie::vector<float, 16> so = aie::broadcast<float, 16>(InvRms(SumSqBf16(o)));
  const int base = slot * 4096;
  for (int i = 0; i < 2048; i += 16) {
    aie::vector<float, 16> t = aie::mul(aie::mul(Wide(o + i), so).to_vector<float>(), Wide(w + i)).to_vector<float>();
    aie::accum<accfloat, 16> nx;
    nx.from_vector(aie::add(Wide(x + i), t));
    const aie::vector<bfloat16, 16> v = nx.to_vector<bfloat16>();
    aie::store_v(x + i, v);
    Put16(tile, base + 2048 + i, v);
  }
  const aie::vector<float, 16> sh = aie::broadcast<float, 16>(InvRms(SumSqBf16(x)));
  for (int i = 0; i < 2048; i += 16) {
    aie::accum<accfloat, 16> t = aie::mul(aie::mul(Wide(x + i), sh).to_vector<float>(), Wide(w + 2048 + i));
    Put16(tile, base + i, t.to_vector<bfloat16>());
  }
}

}  // extern "C"
