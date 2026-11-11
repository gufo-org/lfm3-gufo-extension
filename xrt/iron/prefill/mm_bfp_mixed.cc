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
// w_post (bf16) once as an A object, then each row's [x | o] as one B object
// broadcast to the four cores of a column. With x1 = x + rmsnorm(o) * w_post
// and xn = rmsnorm(x1), core q writes elements [1024 q, 1024 q + 1024) of
// [xn | x1] for 16 rows into its C tile (matmul 8x8-tiled layout). The
// pre-norm weight is folded into the next GEMM's weights.

namespace {
inline aie::vector<float, 16> Wide(const bfloat16 *p) {
  aie::accum<accfloat, 16> a;
  a.from_vector(aie::load_v<16>(p));
  return a.to_vector<float>();
}
inline aie::vector<float, 16> Rsqrt(aie::vector<float, 16> sum) {
  return aie::invsqrt(aie::add(aie::mul(sum, aie::broadcast<float, 16>(1.0f / 2048.0f)).to_vector<float>(),
                               aie::broadcast<float, 16>(1e-5f)));
}
inline aie::vector<float, 16> Total(aie::accum<accfloat, 16> a) {
  return aie::broadcast<float, 16>(aie::reduce_add(a.to_vector<float>()));
}
// Tile rows of the C object in the matmul layout [m/r][n/t][r][t] with r = t = 8:
// element (R, c) sits at ((R >> 3) * 16 + (c >> 3)) * 64 + (R & 7) * 8 + (c & 7).
inline bfloat16 *TileRow(bfloat16 *tile, int R) { return tile + ((R >> 3) << 10) + ((R & 7) << 3); }
// Stores 16 consecutive columns starting at column block cb (8 columns each).
inline void Store16(bfloat16 *d, int cb, aie::vector<bfloat16, 16> v) {
  aie::store_v(d + (cb << 6), v.extract<8>(0));
  aie::store_v(d + (cb << 6) + 64, v.extract<8>(1));
}
}  // namespace

extern "C" {

// Row slot within the current C object: rows of a phase come in multiples of 16.
int row_slot;

// wp: post-norm weight (2048, one A object); row: [x | o]; part q = 0..3.
void prefill_norm_row(const bfloat16 *__restrict wp, const bfloat16 *__restrict row, bfloat16 *__restrict tile,
                      int32_t part) {
  const int slot = row_slot;
  row_slot = (slot + 1) & 15;
#ifdef NORM_STUB
  return;
#endif
  const bfloat16 *x = row;
  const bfloat16 *o = row + 2048;
  // One pass: sum(o^2), sum(x^2), sum(x wo), sum(wo^2) with wo = w o; then
  // sum(x1^2) = sum(x^2) + 2 so sum(x wo) + so^2 sum(wo^2).
  aie::accum<accfloat, 16> oo = aie::zeros<accfloat, 16>(), xx = oo, xw = oo, ww = oo;
  for (int i = 0; i < 2048; i += 16) {
    const aie::vector<bfloat16, 16> xv = aie::load_v<16>(x + i), ov = aie::load_v<16>(o + i);
    const aie::vector<bfloat16, 16> wo = aie::mul(aie::load_v<16>(wp + i), ov).to_vector<bfloat16>();
    oo = aie::mac(oo, ov, ov);
    xx = aie::mac(xx, xv, xv);
    xw = aie::mac(xw, xv, wo);
    ww = aie::mac(ww, wo, wo);
  }
  const aie::vector<float, 16> so = Rsqrt(Total(oo));
  const aie::vector<float, 16> s2 = aie::mul(so, so).to_vector<float>();
  const aie::vector<float, 16> sum1 =
      aie::add(aie::add(Total(xx), aie::mul(aie::mul(so, Total(xw)).to_vector<float>(), aie::broadcast<float, 16>(2.0f)).to_vector<float>()),
               aie::mul(s2, Total(ww)).to_vector<float>());
  const aie::vector<float, 16> sh = Rsqrt(sum1);
  const int at = (part & 1) * 1024;   // element offset within xn or x1
  const bool norm = part < 2;
  for (int j = 0; j < 8; ++j) {
    bfloat16 *d = TileRow(tile, slot * 8 + j);
    for (int c = 0; c < 128; c += 16) {
      const int i = at + j * 128 + c;
      aie::vector<float, 16> t = aie::mul(aie::mul(aie::load_v<16>(o + i), aie::load_v<16>(wp + i)).to_vector<float>(), so).to_vector<float>();
      aie::vector<float, 16> x1 = aie::add(Wide(x + i), t);
      if (norm) x1 = aie::mul(x1, sh).to_vector<float>();
      aie::accum<accfloat, 16> r;
      r.from_vector(x1);
      Store16(d, c >> 3, r.to_vector<bfloat16>());
    }
  }
}

// In place on a C tile whose columns 0..63 are the conv B gate and 64..127
// the conv input of the same 64 channels: writes B * x into columns 0..63.
void prefill_bx(bfloat16 *__restrict c) {
  for (int bi = 0; bi < 16; ++bi)
    for (int bj = 0; bj < 8; ++bj)
      for (int e = 0; e < 64; e += 16) {
        bfloat16 *g = c + (bi * 16 + bj) * 64 + e;
        const aie::vector<bfloat16, 16> u = aie::load_v<16>(c + (bi * 16 + bj + 8) * 64 + e);
        aie::store_v(g, aie::mul(aie::load_v<16>(g), u).to_vector<bfloat16>());
      }
}

}  // extern "C"

// ---- Prefill short convolution phase. The scratch row of token t is
// [bx_j 64 | C_j 64] for the 32 channel blocks j; core q owns blocks
// 8q..8q+7 (512 channels). Its 7 x 512 taps (tap k of channel e at k * 512 + e)
// and a ring of the last 6 bx rows live in the half of its C tile that the
// drain discards: within each 8x8 block, elements 32..63. The other half
// holds y = conv(bx) * C for 16 tokens: token s, channel e at linear
// s * 1024 + e of the tile.

namespace {
inline bfloat16 *Spare(bfloat16 *tile, int u) { return tile + ((u >> 5) << 6) + 32 + (u & 31); }
constexpr int kHist = 7 * 512;
int conv_ring;
}  // namespace

extern "C" {

// Copies taps [which * 2048, which * 2048 + 2048) (only 1536 for which = 1)
// from an A object into the tile and restarts the history ring.
void prefill_conv_taps(const bfloat16 *__restrict a, bfloat16 *__restrict tile, int32_t which) {
  conv_ring = 0;
  const int n = which ? 1536 : 2048;
  for (int i = 0; i < n; i += 16) aie::store_v(Spare(tile, which * 2048 + i), aie::load_v<16>(a + i));
}

// One token row for core part q. With out = 0 it only enters the history.
void prefill_conv_row(const bfloat16 *__restrict row, bfloat16 *__restrict tile, int32_t part, int32_t out) {
  const int ring = conv_ring;
  const int slot = row_slot;
  if (out) row_slot = (slot + 1) & 15;
  for (int jb = 0; jb < 8; ++jb) {
    const bfloat16 *bx = row + (part * 8 + jb) * 128;
    for (int e = 0; e < 64; e += 16) {
      const int ch = jb * 64 + e;
      const aie::vector<bfloat16, 16> cur = aie::load_v<16>(bx + e);
      if (out) {
        aie::accum<accfloat, 16> acc = aie::mul(aie::load_v<16>(Spare(tile, 6 * 512 + ch)), cur);
        for (int k = 0; k < 6; ++k) {
          int h = ring + k;
          if (h >= 6) h -= 6;
          acc = aie::mac(acc, aie::load_v<16>(Spare(tile, k * 512 + ch)), aie::load_v<16>(Spare(tile, kHist + h * 512 + ch)));
        }
        const aie::vector<bfloat16, 16> y = aie::mul(acc.to_vector<bfloat16>(), aie::load_v<16>(bx + 64 + e)).to_vector<bfloat16>();
        // linear slot * 1024 + ch: tile row slot * 8 + (ch >> 7), column ch & 127
        bfloat16 *d = tile + slot * 1024 + (((ch & 127) >> 3) << 6) + ((ch >> 7) << 3);
        aie::store_v(d, y.extract<8>(0));
        aie::store_v(d + 64, y.extract<8>(1));
      }
      aie::store_v(Spare(tile, kHist + ring * 512 + ch), cur);
    }
  }
  conv_ring = ring == 5 ? 0 : ring + 1;
}

// ---- Phase interpreter. PREFILL_PLAN lists one code per phase:
// kind | mode << 2 | tiles << 4 | ksteps << 11 (kind 0 GEMM, 1 norm, 2 conv;
// mode 0 plain, 1 SwiGLU, 2 bx).
#ifndef PREFILL_PLAN
#define PREFILL_PLAN 0
#endif
namespace {
const int32_t kPlan[] = {PREFILL_PLAN};
int phase_at, epilogue_mode;
}  // namespace

// cnt: [GEMM tiles, k steps, norm, conv]
void prefill_next(int32_t *__restrict cnt) {
  const int32_t code = kPlan[phase_at];
  phase_at = phase_at + 1 == int(sizeof(kPlan) / sizeof(kPlan[0])) ? 0 : phase_at + 1;  // the core reruns per dispatch
  const int kind = code & 3;
  epilogue_mode = (code >> 2) & 3;
  cnt[0] = kind == 0 ? (code >> 4) & 127 : 0;
  cnt[1] = code >> 11;
  cnt[2] = kind == 1;
  cnt[3] = kind == 2;
}

void prefill_epilogue(bfloat16 *__restrict c) {
  if (epilogue_mode == 1) prefill_swiglu(c);
  else if (epilogue_mode == 2) prefill_bx(c);
}

}  // extern "C"
