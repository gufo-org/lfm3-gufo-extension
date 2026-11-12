// LFM3 decode kernels (one core's part of a phase; see decode.py).
//
// Vectors live in DDR in natural order. Core (c, j) owns hidden rows
// ((c * 8 + t) * 4 + j) * 8 + e (t, e < 8), intermediate channels
// ((c * 15 + s) * 4 + j) * 16 + e (s < 15, e < 16) and convolution channels
// ((c * 4 + s) * 4 + j) * 16 + e (s < 4, e < 16).
//
// A weight record is REC bytes: some int8 rows of Q8_0 codes, then their
// bf16 block scales, then (convolution records) the channel's taps.
#include <stdint.h>
#include <aie_api/aie.hpp>

#ifdef DECODE_ATTN_ROLE
#define ROLE(n) attn_##n
#else
#define ROLE(n) conv_##n
#endif


namespace {
constexpr int kH = 2048, kI = 7680, kTaps = 7;
constexpr int kRec = 4 * kH + 4 * (kH / 32) * 2;  // 8704
constexpr int kIn = 12 * kH;                      // phase input bytes
constexpr float kEps = 1e-5f;

float SumSquares(const float* v, int n) {
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> x = aie::load_v<16>(v + i);
    acc = aie::mac(acc, x, x);
  }
  return aie::reduce_add(acc.to_vector<float>());
}

// rsqrt(sum * inv_n + eps), all in vector lanes (no soft-float).
float RmsScale(float sum, float inv_n) {
  aie::vector<float, 16> v = aie::broadcast<float, 16>(sum);
  v = aie::mul(v, aie::broadcast<float, 16>(inv_n)).to_vector<float>();
  v = aie::add(v, aie::broadcast<float, 16>(kEps));
  return aie::invsqrt(v)[0];
}

aie::vector<float, 16> Bf16ToFloat(const bfloat16* p) {
  aie::accum<accfloat, 16> a;
  a.from_vector(aie::load_v<16>(p));
  return a.to_vector<float>();
}

// Q8_0 of v (n a multiple of 512): n codes at q, n / 32 scales at qs.
__attribute__((noinline)) void Quantize(const float* v, int n, int8_t* q, bfloat16* qs) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);
  for (int g = 0; g < n; g += 512) {
    alignas(64) float amax[16];
    for (int j = 0; j < 16; ++j) {
      const float* x = v + g + j * 32;
      amax[j] = aie::reduce_max(aie::max(aie::abs(aie::load_v<16>(x)), aie::abs(aie::load_v<16>(x + 16))));
    }
    aie::vector<float, 16> d = aie::mul(aie::load_v<16>(amax), aie::broadcast<float, 16>(1.0f / 127.0f)).to_vector<float>();
    aie::accum<accfloat, 16> da;
    da.from_vector(d);
    aie::vector<bfloat16, 16> db = da.to_vector<bfloat16>();  // the stored scale
    aie::store_v(qs + g / 32, db);
    aie::accum<accfloat, 16> dr;
    dr.from_vector(db);
    aie::vector<float, 16> df = dr.to_vector<float>();
    aie::mask<16> zero = aie::le(df, aie::zeros<float, 16>());
    alignas(64) float inv[16];
    aie::store_v(inv, aie::select(aie::inv(aie::select(df, aie::broadcast<float, 16>(1.0f), zero)), aie::zeros<float, 16>(), zero));
    for (int j = 0; j < 16; ++j) {
      const float* x = v + g + j * 32;
      aie::vector<float, 16> iv = aie::broadcast<float, 16>(inv[j]);
      aie::vector<int8, 16> c0 = aie::to_fixed<int8>(aie::mul(aie::load_v<16>(x), iv).to_vector<float>(), 0);
      aie::vector<int8, 16> c1 = aie::to_fixed<int8>(aie::mul(aie::load_v<16>(x + 16), iv).to_vector<float>(), 0);
      aie::store_v(q + g + j * 32, aie::concat(c0, c1));
    }
  }
}

// bf16 vector (n values) to Q8_0 at q (codes) and q + n (scales).
void QuantizeBf16(const bfloat16* in, int n, int8_t* q, float* scratch) {
  bfloat16* qs = reinterpret_cast<bfloat16*>(q + n);
  for (int c = 0; c < n; c += 512) {
    const int m = n - c < 512 ? n - c : 512;
    for (int i = 0; i < m; i += 16) aie::store_v(scratch + i, Bf16ToFloat(in + c + i));
    Quantize(scratch, m, q + c, qs + c / 32);
  }
}

// One weight row of k codes (scales at `scales`) against quantized x.

// Four rows of kH codes as one 4-row record (token_host.tile4): per 32-code block b
// two 8 x 8 int8 tiles t [k][n], column n < 4 holding codes 32 b + 16 t + k of row n
// and column n >= 4 codes 32 b + 16 t + 8 + k of row n - 4; then bf16 scales
// [b][row]. A is x's 64 aligned bytes of blocks b, b + 1 (row m = chunk m); with
// tile t's own accumulator, C rows 4 o + 2 t and 4 o + 2 t + 1 hold block b + o's
// chunk partial sums at columns 0..3 and 4..7.
alignas(64) int32_t dot4_sums[kH / 32 * 4 + 16];
inline const bfloat16* Scales(const int8_t* q8, int k) { return reinterpret_cast<const bfloat16*>(q8 + k); }
// nb blocks (a multiple of 16); xs: x's block scales (32-byte aligned).
__attribute__((noinline)) void Dot4(const int8_t* x, const bfloat16* xs, const int8_t* w, int nb, float* out) {
  const bfloat16* ws = reinterpret_cast<const bfloat16*>(w + 128 * nb);
  // 512-bit loads need 64-byte alignment: one A per pair of blocks, the odd block
  // in A rows 4..7 (C rows 4..7).
  aie::vector<int32, 16> four = aie::zeros<int32, 16>();  // sums of 4 blocks, stored aligned
  for (int b = 0; b < nb; b += 2) {
    const aie::vector<int8, 64> a = aie::load_v<64>(x + 32 * b);
#pragma clang loop unroll(disable)
    for (int o = 0; o < 2; ++o) {
      aie::mmul<8, 8, 8, int8, int8> m0, m1;
      m0.mul(a, aie::load_v<64>(w + 128 * (b + o)));
      m1.mul(a, aie::load_v<64>(w + 128 * (b + o) + 64));
      const aie::vector<int32, 16> c0 = m0.to_vector<int32>().extract<16>(2 * o);      // C rows 4 o, 4 o + 1
      const aie::vector<int32, 16> c1 = m1.to_vector<int32>().extract<16>(2 * o + 1);  // C rows 4 o + 2, 4 o + 3
      four = aie::shuffle_down_fill(four, aie::add(aie::add(c0, aie::shuffle_down(c0, 12)), aie::add(c1, aie::shuffle_down(c1, 12))), 4);
    }
    if ((b & 3) == 2) aie::store_v(dot4_sums + 4 * (b - 2), four);
  }
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
  for (int q = 0; q < nb / 16; ++q) {
    // x scales of blocks 16 q .. 16 q + 15, each repeated for the four rows
    const aie::vector<bfloat16, 16> v = aie::load_v<16>(xs + 16 * q);
    const auto [z0, z1] = aie::interleave_zip(v, v, 1);
    const auto [y0, y1] = aie::interleave_zip(z0, z0, 1);
    const auto [y2, y3] = aie::interleave_zip(z1, z1, 1);
    const aie::vector<bfloat16, 16> x4[4] = {y0, y1, y2, y3};
#pragma clang loop unroll(disable)
    for (int i = 0; i < 4; ++i) {
      const int b0 = 16 * q + 4 * i;  // blocks b0 .. b0 + 3
      const aie::vector<float, 16> sc = aie::mul(aie::load_v<16>(ws + 4 * b0), x4[i]).to_vector<float>();
      acc = aie::mac(acc, aie::to_float(aie::load_v<16>(dot4_sums + 4 * b0), 0), sc);
    }
  }
  // lanes 4 i + r -> row r
  aie::vector<float, 16> a = acc.to_vector<float>();
  a = aie::add(a, aie::shuffle_down(a, 8));
  a = aie::add(a, aie::shuffle_down(a, 4));
  aie::store_v(out, a.extract<4>(0));  // out is 16-byte aligned
}

alignas(64) float acc_a[16];
alignas(64) float acc_b[16];
alignas(64) float acc_c[16];
#ifndef DECODE_ATTN_ROLE
alignas(64) float conv_taps[kTaps][16];
#endif
#ifndef DECODE_ATTN_ROLE
alignas(64) float conv_state[128 * (kTaps - 1)];
#endif
int acc_n, conv_l, conv_k, slice_t, copy_r;
}  // namespace

extern "C" {

// The next record of a phase input: bytes [r * REC, (r + 1) * REC) of x,
// clipped; dec_pre and dec_quant start the next input.
void dec_copy(const int8_t* rec, int8_t* x) {
  const int start = copy_r++ * kRec, n = (kIn - start) < kRec ? kIn - start : kRec;
  for (int i = 0; i < n; i += 64) aie::store_v(x + start + i, aie::load_v<64>(rec + i));
}

// x: [h f32 H | o f32 H | post-norm w bf16 H | pre-norm w bf16 H].
// h += rmsnorm(o) * w_post in place; q8 = Q8_0(rmsnorm(h) * w_pre).
void dec_pre(float* x, int8_t* q8, float* scratch) {
  float* h = x;
  const float* o = x + kH;
  const bfloat16* wpost = reinterpret_cast<const bfloat16*>(x + 2 * kH);
  const bfloat16* wpre = wpost + kH;
  const float so = RmsScale(SumSquares(o, kH), 1.0f / kH);
  for (int i = 0; i < kH; i += 16) {
    aie::vector<float, 16> t = aie::mul(aie::load_v<16>(o + i), aie::broadcast<float, 16>(so)).to_vector<float>();
    t = aie::mul(t, Bf16ToFloat(wpost + i)).to_vector<float>();
    aie::store_v(h + i, aie::add(aie::load_v<16>(h + i), t));
  }
  const float sh = RmsScale(SumSquares(h, kH), 1.0f / kH);
  bfloat16* qs = reinterpret_cast<bfloat16*>(q8 + kH);
  for (int c = 0; c < kH; c += 512) {
    for (int i = 0; i < 512; i += 16) {
      aie::vector<float, 16> t = aie::mul(aie::load_v<16>(h + c + i), aie::broadcast<float, 16>(sh)).to_vector<float>();
      aie::store_v(scratch + i, aie::mul(t, Bf16ToFloat(wpre + c + i)).to_vector<float>());
    }
    Quantize(scratch, 512, q8 + c, qs + c / 32);
  }
  acc_n = slice_t = copy_r = 0;
}

// The next eight of this core's hidden rows of h (chunks 0..7 after dec_pre).
void dec_slice(const float* x, float* out, int32_t c, int32_t j) {
  const int t = slice_t++;
  aie::store_v(out, aie::load_v<8>(x + ((c * 8 + t) * 4 + j) * 8));
}

// x: bf16 vector of n values.
void dec_quant(const bfloat16* x, int8_t* q8, float* scratch, int32_t n) {
  QuantizeBf16(x, n, q8, scratch);
  acc_n = copy_r = 0;
}

// ---- FFN

// Gate rows of two channels, then their up rows.
void dec_gate_up(const int8_t* q8, const int8_t* w) {
  alignas(16) float r4[4];  // gate a, gate b, up a, up b
  Dot4(q8, Scales(q8, kH), w, kH / 32, r4);
  acc_a[acc_n] = r4[0];
  acc_a[acc_n + 1] = r4[1];
  acc_b[acc_n] = r4[2];
  acc_b[acc_n + 1] = r4[3];
  acc_n += 2;
}

// SiLU(gate) * up of the last sixteen channels, as bf16:
// g * u * (1 + tanh(g / 2)) / 2.
void dec_gate_flush(bfloat16* out) {
  aie::vector<float, 16> g = aie::load_v<16>(acc_a), u = aie::load_v<16>(acc_b);
  aie::vector<float, 16> half = aie::broadcast<float, 16>(0.5f);
  aie::accum<accfloat, 16> t;
  t.from_vector(aie::tanh<bfloat16>(aie::mul(g, half).to_vector<float>()));
  aie::vector<float, 16> sig = aie::mul(aie::add(t.to_vector<float>(), aie::broadcast<float, 16>(1.0f)), half).to_vector<float>();
  aie::vector<float, 16> a = aie::mul(aie::mul(g, u).to_vector<float>(), sig).to_vector<float>();
  aie::accum<accfloat, 16> r;
  r.from_vector(a);
  aie::store_v(out, r.to_vector<bfloat16>());
  acc_n = 0;
}

// Down projection: four rows per record over one slice of I (2048, 2048, 2048,
// 1536 codes; token_host packs them in that order), accumulated over the slices.
int down_slice;
void dec_row(const int8_t* q8, const int8_t* w, int32_t /*k*/) {
  alignas(16) float r4[4];
  const int q = down_slice;
  Dot4(q8 + 2048 * q, Scales(q8, kI) + 64 * q, w, q < 3 ? 64 : 48, r4);
  for (int r = 0; r < 4; ++r) acc_a[acc_n + r] = (q ? acc_a[acc_n + r] : 0.0f) + r4[r];
  if (++down_slice == 4) {
    down_slice = 0;
    acc_n += 4;
  }
}

// Four rows of H codes (out projections, head).
void dec_rows4(const int8_t* q8, const int8_t* w) {
  Dot4(q8, Scales(q8, kH), w, kH / 32, acc_a + acc_n);
  acc_n += 4;
}

// The last eight row results, as f32.
void dec_rows_flush(float* out) {
  aie::store_v(out, aie::load_v<8>(acc_a));
  acc_n = 0;
}

#ifndef DECODE_ATTN_ROLE
// ---- Short convolution

// Record 0 of a convolution phase's state: 6 past B*x values x this core's
// 64 channels, tap-major.
void dec_conv_load(const int8_t* rec) {
  conv_l = conv_k = acc_n = 0;
  const float* s = reinterpret_cast<const float*>(rec);
  for (int i = 0; i < 128 * (kTaps - 1); i += 16) aie::store_v(conv_state + i, aie::load_v<16>(s + i));
}

// The next channel: B, C and x rows, scales, then 7 f32 taps.
// One channel's record (token_host.tile4 of rows B, C, x and a fourth row whose
// first 28 codes are the 7 f32 taps, with zero scales).
void dec_conv(const int8_t* q8, const int8_t* w) {
  alignas(16) float r4[4];
  Dot4(q8, Scales(q8, kH), w, kH / 32, r4);
  acc_a[acc_n] = r4[0];  // B
  acc_b[acc_n] = r4[1];  // C
  acc_c[acc_n] = r4[2];  // x
  // code j of row 3 in block 0: tile j / 16, column 3 + 4 * ((j / 8) % 2), k = j % 8
  alignas(4) int8_t tb[4 * kTaps];
  for (int j = 0; j < 4 * kTaps; ++j) tb[j] = w[64 * (j >> 4) + 8 * (j & 7) + 3 + 4 * ((j >> 3) & 1)];
  const float* taps = reinterpret_cast<const float*>(tb);
  for (int t = 0; t < kTaps; ++t) conv_taps[t][acc_n] = taps[t];
  ++acc_n;
}

// The last sixteen channels' convolution, as bf16. History is kept
// tap-major for these lanes: conv_state[t * 128 + channel].
void dec_conv_flush(bfloat16* out) {
  const int l0 = conv_l;
  aie::vector<float, 16> bx = aie::mul(aie::load_v<16>(acc_a), aie::load_v<16>(acc_c)).to_vector<float>();
  aie::accum<accfloat, 16> sum = aie::mul(bx, aie::load_v<16>(conv_taps[kTaps - 1]));
  for (int t = 0; t < kTaps - 1; ++t) sum = aie::mac(sum, aie::load_v<16>(conv_state + t * 128 + l0), aie::load_v<16>(conv_taps[t]));
  for (int t = 0; t + 1 < kTaps - 1; ++t) aie::store_v(conv_state + t * 128 + l0, aie::load_v<16>(conv_state + (t + 1) * 128 + l0));
  aie::store_v(conv_state + (kTaps - 2) * 128 + l0, bx);
  aie::accum<accfloat, 16> r;
  r.from_vector(aie::mul(sum.to_vector<float>(), aie::load_v<16>(acc_b)).to_vector<float>());
  aie::store_v(out, r.to_vector<bfloat16>());
  conv_l += 16;
  acc_n = 0;
}

// The next eight floats (96 chunks) of the updated convolution state.
void dec_conv_store(float* out) {
  aie::store_v(out, aie::load_v<8>(conv_state + 8 * conv_k++));
}

#endif  // !DECODE_ATTN_ROLE
}  // extern "C"

#ifdef DECODE_ATTN_ROLE
// ---- Attention (A2). Core (c, j < 2) owns head h = 2c + j; its kv head is
// h / 4.

namespace {
constexpr int kD = 128, kQRows = 3328;  // q 2048 | k 512 | v 512 | gate 16 | pad
alignas(64) float att_q[kD];
alignas(64) float att_k[kD];
alignas(64) float att_v[kD];
alignas(64) float att_acc[kD];
alignas(64) float att_s[16];
aie::vector<float, 16> att_m, att_l, att_gate;
int att_left, att_out_k, att_kv_k;

// x[0..128): rmsnorm(x) * w (bf16), then NeoX rotation by cos/sin.
void NormRope(float* x, const bfloat16* w, const float* cs, const float* sn) {
  const float s = RmsScale(SumSquares(x, kD), 1.0f / kD);
  for (int i = 0; i < kD; i += 16) {
    aie::vector<float, 16> v = aie::mul(aie::load_v<16>(x + i), aie::broadcast<float, 16>(s)).to_vector<float>();
    aie::store_v(x + i, aie::mul(v, Bf16ToFloat(w + i)).to_vector<float>());
  }
  for (int i = 0; i < kD / 2; i += 16) {
    aie::vector<float, 16> a = aie::load_v<16>(x + i), b = aie::load_v<16>(x + i + kD / 2);
    aie::vector<float, 16> c = aie::load_v<16>(cs + i), n = aie::load_v<16>(sn + i);
    aie::accum<accfloat, 16> lo = aie::mul(a, c), hi = aie::mul(b, c);
    lo = aie::msc(lo, b, n);
    hi = aie::mac(hi, a, n);
    aie::store_v(x + i, lo.to_vector<float>());
    aie::store_v(x + i + kD / 2, hi.to_vector<float>());
  }
}

aie::vector<float, 16> Exp16(aie::vector<float, 16> x) {
  aie::accum<accfloat, 16> e;
  e.from_vector(aie::exp2<bfloat16>(aie::mul(x, aie::broadcast<float, 16>(1.4426950409f)).to_vector<float>()));
  return e.to_vector<float>();
}

// Folds sixteen scores (att_s, first `valid` real) and their values.
void Fold(const bfloat16* const* v, int valid) {
  aie::vector<float, 16> s = aie::mul(aie::load_v<16>(att_s), aie::broadcast<float, 16>(0.08838834764f)).to_vector<float>();
  alignas(64) static const int32_t lane[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  aie::mask<16> real = aie::lt(aie::load_v<16>(lane), aie::broadcast<int32, 16>(valid));
  s = aie::select(aie::broadcast<float, 16>(-1e30f), s, real);
  const aie::vector<float, 16> m = aie::max(att_m, aie::broadcast<float, 16>(aie::reduce_max(s)));
  const aie::vector<float, 16> f = Exp16(aie::sub(att_m, m));
  alignas(64) float p[16];
  aie::store_v(p, aie::select(aie::zeros<float, 16>(), Exp16(aie::sub(s, m)), real));
  att_l = aie::add(aie::mul(att_l, f).to_vector<float>(), aie::broadcast<float, 16>(aie::reduce_add(aie::load_v<16>(p))));
  att_m = m;
  for (int d = 0; d < kD; d += 16) {
    aie::accum<accfloat, 16> a = aie::mul(aie::load_v<16>(att_acc + d), f);
    for (int i = 0; i < valid; ++i) a = aie::mac(a, Bf16ToFloat(v[i] + d), aie::broadcast<float, 16>(p[i]));
    aie::store_v(att_acc + d, a.to_vector<float>());
  }
}

float DotBf16(const float* q, const bfloat16* k) {
  aie::accum<accfloat, 16> a = aie::zeros<accfloat, 16>();
  for (int i = 0; i < kD; i += 16) a = aie::mac(a, aie::load_v<16>(q + i), Bf16ToFloat(k + i));
  return aie::reduce_add(a.to_vector<float>());
}
}  // namespace

extern "C" {

// x: [q|k|v|gate f32 kQRows | q-norm w bf16 128 | k-norm w bf16 128], then at
// byte 2 * REC the position's cos (64 f32) and sin (64 f32). `cached` is the
// number of earlier positions this head attends to.
void dec_attn_begin(const float* x, int32_t c, int32_t j, int32_t cached) {
  const int h = 2 * c + j, m = h / 4;
  const bfloat16* qn = reinterpret_cast<const bfloat16*>(x + kQRows);
  const bfloat16* kn = qn + kD;
  const float* cs = reinterpret_cast<const float*>(reinterpret_cast<const int8_t*>(x) + 2 * kRec);
  const float* sn = cs + kD / 2;
  for (int i = 0; i < kD; i += 16) {
    aie::store_v(att_q + i, aie::load_v<16>(x + h * kD + i));
    aie::store_v(att_k + i, aie::load_v<16>(x + 2048 + m * kD + i));
    aie::store_v(att_v + i, aie::load_v<16>(x + 2560 + m * kD + i));
  }
  NormRope(att_q, qn, cs, sn);
  NormRope(att_k, kn, cs, sn);
  att_gate = aie::broadcast<float, 16>(x[3072 + h]);
  att_m = aie::broadcast<float, 16>(-1e30f);
  att_l = aie::zeros<float, 16>();
  for (int d = 0; d < kD; d += 16) aie::store_v(att_acc + d, aie::zeros<float, 16>());
  att_left = cached;
  att_out_k = att_kv_k = 0;
}

}  // extern "C"
namespace {
// Sixteen positions (the first `valid` real): [k bf16 128 | v bf16 128] each.
__attribute__((noinline)) void AttnRecord(const int8_t* rec, int valid) {
  const bfloat16* vs[16];
  for (int i = 0; i < 16; ++i) {
    const bfloat16* kv = reinterpret_cast<const bfloat16*>(rec + i * 512);
    att_s[i] = i < valid ? DotBf16(att_q, kv) : 0.0f;
    vs[i] = kv + kD;
  }
  Fold(vs, valid);
}
alignas(64) bfloat16 att_cur[2 * kD];

void ToBf16(const float* src, bfloat16* dst, int n) {
  aie::accum<accfloat, 16> t;
  for (int i = 0; i < n; i += 16) {
    t.from_vector(aie::load_v<16>(src + i));
    aie::store_v(dst + i, t.to_vector<bfloat16>());
  }
}
}  // namespace
extern "C" {

void dec_attn_rec(const int8_t* rec) {
  AttnRecord(rec, att_left < 16 ? att_left : 16);
  att_left -= 16;
}

// The current position (its bf16 cache entry), then the output:
// acc / l * sigmoid(gate).
void dec_attn_finish() {
  ToBf16(att_k, att_cur, kD);
  ToBf16(att_v, att_cur + kD, kD);
  AttnRecord(reinterpret_cast<const int8_t*>(att_cur), 1);
  // bf16 multiplies (native; the output is bf16): sigmoid(g) = 0.5 + 0.5 tanh(g / 2)
  const aie::vector<bfloat16, 16> half = aie::broadcast<bfloat16, 16>(0.5f);
  aie::accum<accfloat, 16> t;
  t.from_vector(att_gate);
  const aie::vector<float, 16> g2 = aie::mul(t.to_vector<bfloat16>(), half).to_vector<float>();
  aie::accum<accfloat, 16> sig;
  sig.from_vector(aie::broadcast<float, 16>(0.5f));
  sig = aie::mac(sig, aie::tanh<bfloat16>(g2), half);
  aie::accum<accfloat, 16> il;
  il.from_vector(aie::inv(att_l));
  const aie::vector<bfloat16, 16> scale = aie::mul(il.to_vector<bfloat16>(), sig.to_vector<bfloat16>()).to_vector<bfloat16>();
  for (int d = 0; d < kD; d += 16) {
    aie::accum<accfloat, 16> v;
    v.from_vector(aie::load_v<16>(att_acc + d));
    aie::store_v(att_acc + d, aie::mul(v.to_vector<bfloat16>(), scale).to_vector<float>());
  }
}

// The next sixteen of this head's 128 output dims, as bf16.
void dec_attn_out(bfloat16* out) {
  aie::accum<accfloat, 16> t;
  t.from_vector(aie::load_v<16>(att_acc + 16 * att_out_k++));
  aie::store_v(out, t.to_vector<bfloat16>());
}

// The next 32 bytes of this position's cache entry [k bf16 128 | v bf16 128].
void dec_kv_store(bfloat16* out) {
  aie::store_v(out, aie::load_v<16>(att_cur + 16 * att_kv_k++));
}

}  // extern "C"

#endif  // DECODE_ATTN_ROLE

// ---- Phase interpreter. The core program asks dec_next for each phase's
// counts and runs: heads (shared input records, copied into x), own records
// (dec_own), dec_begin, pre outputs, groups of records with a flush,
// dec_finish, post outputs. Every core of a column emits the same number of
// objects per phase (idle roles emit zeros), as the column's drains expect.

namespace {
enum Kind : int32_t { kEmbed, kConv1, kConv2, kAttn1, kAttn2, kAttn3, kFfn1, kFfn2, kHead, kIdle };
constexpr int kLayers = 24, kHeadRows = 128000;
// LFM3-1.7B: layers 2, 4, ..., 22 attend; the rest convolve.
bool Attends(int l) { return l >= 2 && l % 2 == 0; }

int ph, ph_layer, ph_step, ph_kind, ph_cached, core_c, core_j;
const int32_t* ph_rtp;
alignas(64) float zeros16[16];

struct Counts { int32_t heads, own, pre, outer, inner, post, kind, extra; };

Counts Plan(int kind, int j, const int32_t* rtp, int attn_index) {
  const bool attn_role = j < 2;
  switch (kind) {
    case kEmbed: return {1, 0, 8, 0, 0, 0, kind, 0};
    case kConv1:  // B/C/x and the convolution: rows 4-5 own 128 channels each
      return attn_role ? Counts{3, 0, 8, 8, 0, 96, kIdle, 1} : Counts{3, 1, 8, 8, 16, 96, kind, 0};
    case kConv2: return {1, 0, 0, 8, 2, 0, kind, 0};       // 64 out rows: 16 records x 4
    case kAttn1: return {3, 0, 8, 13, 2, 0, kind, 0};      // 104 q/k/v/gate rows
    case kAttn2: {
      const int cached = rtp[2 * attn_index], recs = rtp[2 * attn_index + 1];
      return attn_role ? Counts{3, 0, 0, 1, recs, 24, kind, cached} : Counts{0, 0, 0, 1, 0, 24, kIdle, 0};
    }
    case kAttn3: return {1, 0, 0, 8, 2, 0, kind, 0};
    case kFfn1: return {3, 0, 8, 15, 8, 0, kind, 0};
    case kFfn2: return {2, 0, 0, 8, 8, 0, kind, 0};
    case kHead: return {3, 0, 0, 500, 2, 0, kind, 0};      // 4000 rows: 1000 records x 4
  }
  return {0, 0, 0, 0, 0, 0, kIdle, 0};
}

int KindAt(int layer, int step) {
  if (layer < 0) return kEmbed;
  if (layer >= kLayers) return kHead;
  if (Attends(layer)) return step == 0 ? kAttn1 : step == 1 ? kAttn2 : step == 2 ? kAttn3 : step == 3 ? kFfn1 : kFfn2;
  return step == 0 ? kConv1 : step == 1 ? kConv2 : step == 2 ? kFfn1 : kFfn2;
}
int StepsAt(int layer) { return layer < 0 || layer >= kLayers ? 1 : Attends(layer) ? 5 : 4; }
}  // namespace

extern "C" {

// Counts of the next phase; rtp: per attention layer [cached positions,
// cache records]. Resets at the end of a token.
void ROLE(next)(int32_t* counts, const int32_t* rtp, int32_t c, int32_t j) {
  if (ph == 0) { ph_layer = -1; ph_step = 0; ph_rtp = rtp; core_c = c; core_j = j; }
  int attn_index = 0;
  for (int l = 2; l < ph_layer; l += 2) ++attn_index;
  ph_kind = KindAt(ph_layer, ph_step);
  Counts k = Plan(ph_kind, j, rtp, attn_index);
  ph_kind = k.kind;
  ph_cached = k.extra;
  if (++ph_step == StepsAt(ph_layer)) { ph_step = 0; ++ph_layer; }
  ph = ph_layer > kLayers ? 0 : ph + 1;
  counts[0] = k.heads; counts[1] = k.own; counts[2] = k.pre; counts[3] = k.outer;
  counts[4] = k.inner; counts[5] = k.post; counts[6] = k.kind; counts[7] = 0;
}

void ROLE(own)(const int8_t* rec) {
#ifndef DECODE_ATTN_ROLE
  if (ph_kind == kConv1) dec_conv_load(rec);
#endif
}

void ROLE(begin)(int8_t* x, int8_t* q8, float* scratch) {
  float* xf = reinterpret_cast<float*>(x);
  switch (ph_kind) {
    case kEmbed:  // x already holds h (f32): the embedding table is dequantized
      slice_t = 0;
      break;
    case kConv1: case kAttn1: case kFfn1: case kHead: case kIdle:
      if (ph_kind != kIdle || copy_r) dec_pre(xf, q8, scratch);
      if (ph_kind == kConv1) conv_l = conv_k = acc_n = 0;
      break;
    case kConv2: case kAttn3: dec_quant(reinterpret_cast<const bfloat16*>(x), q8, scratch, kH); break;
    case kFfn2: dec_quant(reinterpret_cast<const bfloat16*>(x), q8, scratch, kI); break;
#ifdef DECODE_ATTN_ROLE
    case kAttn2: dec_attn_begin(xf, core_c, core_j, ph_cached); break;
#endif
  }
  copy_r = 0;
}

void ROLE(pre_out)(const int8_t* x, int8_t* out) {
  dec_slice(reinterpret_cast<const float*>(x), reinterpret_cast<float*>(out), core_c, core_j);
}

void ROLE(rec)(const int8_t* q8, const int8_t* w) {
  switch (ph_kind) {
#ifndef DECODE_ATTN_ROLE
    case kConv1: dec_conv(q8, w); break;
#else
    case kAttn2: dec_attn_rec(w); break;
#endif
    case kConv2: case kAttn1: case kAttn3: case kHead: dec_rows4(q8, w); break;
    case kFfn1: dec_gate_up(q8, w); break;
    case kFfn2: dec_row(q8, w, kI); break;
  }
}

void ROLE(flush)(int8_t* out) {
  switch (ph_kind) {
#ifndef DECODE_ATTN_ROLE
    case kConv1: dec_conv_flush(reinterpret_cast<bfloat16*>(out)); break;
#endif
    case kFfn1: dec_gate_flush(reinterpret_cast<bfloat16*>(out)); break;
    case kConv2: case kAttn1: case kAttn3: case kFfn2: case kHead: dec_rows_flush(reinterpret_cast<float*>(out)); break;
    default: aie::store_v(reinterpret_cast<float*>(out), aie::load_v<8>(zeros16)); break;
  }
}

void ROLE(finish)() {
#ifdef DECODE_ATTN_ROLE
  if (ph_kind == kAttn2) dec_attn_finish();
#endif
}

void ROLE(post_out)(int8_t* out) {
  switch (ph_kind) {
#ifndef DECODE_ATTN_ROLE
    case kConv1: dec_conv_store(reinterpret_cast<float*>(out)); break;
#else
    case kAttn2: {  // 8 attention-output objects, then 16 cache-entry objects
      if (att_out_k < 8) dec_attn_out(reinterpret_cast<bfloat16*>(out));
      else dec_kv_store(reinterpret_cast<bfloat16*>(out));
      break;
    }
#endif
    default: aie::store_v(reinterpret_cast<float*>(out), aie::load_v<8>(zeros16)); break;
  }
}

void ROLE(copy)(const int8_t* rec, int8_t* x) { dec_copy(rec, x); }

}  // extern "C"
