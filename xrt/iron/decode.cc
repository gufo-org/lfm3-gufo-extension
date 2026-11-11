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

// rsqrt(sum / n + eps), all in vector lanes (no soft-float).
float RmsScale(float sum, int n) {
  aie::vector<float, 16> v = aie::broadcast<float, 16>(sum);
  v = aie::mul(v, aie::broadcast<float, 16>(1.0f / float(n))).to_vector<float>();
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
__attribute__((noinline)) float Dot(const int8_t* x, int k, const int8_t* row, const bfloat16* scales) {
  const bfloat16* xs = reinterpret_cast<const bfloat16*>(x + k);
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
  for (int b0 = 0; b0 < k / 32; b0 += 16) {
    alignas(64) int32_t sums[16];
    for (int j = 0; j < 16; ++j) {
      const int b = b0 + j;
      aie::accum<acc32, 32> m = aie::mul(aie::load_v<32>(row + b * 32), aie::load_v<32>(x + b * 32));
      sums[j] = aie::reduce_add(m.to_vector<int32>(0));
    }
    aie::vector<float, 16> sf = aie::to_float(aie::load_v<16>(sums), 0);
    aie::vector<float, 16> sc = aie::mul(aie::load_v<16>(scales + b0), aie::load_v<16>(xs + b0)).to_vector<float>();
    acc = aie::mac(acc, sf, sc);
  }
  return aie::reduce_add(acc.to_vector<float>());
}

alignas(64) float acc_a[16];
alignas(64) float acc_b[16];
alignas(64) float acc_c[16];
alignas(64) float conv_taps[kTaps][16];
alignas(64) float conv_state[64 * (kTaps - 1)];
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
  const float so = RmsScale(SumSquares(o, kH), kH);
  for (int i = 0; i < kH; i += 16) {
    aie::vector<float, 16> t = aie::mul(aie::load_v<16>(o + i), aie::broadcast<float, 16>(so)).to_vector<float>();
    t = aie::mul(t, Bf16ToFloat(wpost + i)).to_vector<float>();
    aie::store_v(h + i, aie::add(aie::load_v<16>(h + i), t));
  }
  const float sh = RmsScale(SumSquares(h, kH), kH);
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
  const bfloat16* s = reinterpret_cast<const bfloat16*>(w + 4 * kH);
  for (int r = 0; r < 2; ++r) {
    acc_a[acc_n + r] = Dot(q8, kH, w + r * kH, s + r * (kH / 32));
    acc_b[acc_n + r] = Dot(q8, kH, w + (2 + r) * kH, s + (2 + r) * (kH / 32));
  }
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

// One row of k codes (k = I for down).
void dec_row(const int8_t* q8, const int8_t* w, int32_t k) {
  acc_a[acc_n++] = Dot(q8, k, w, reinterpret_cast<const bfloat16*>(w + k));
}

// Four rows of H codes (out projections, head).
void dec_rows4(const int8_t* q8, const int8_t* w) {
  const bfloat16* s = reinterpret_cast<const bfloat16*>(w + 4 * kH);
  for (int r = 0; r < 4; ++r) acc_a[acc_n + r] = Dot(q8, kH, w + r * kH, s + r * (kH / 32));
  acc_n += 4;
}

// The last eight row results, as f32.
void dec_rows_flush(float* out) {
  aie::store_v(out, aie::load_v<8>(acc_a));
  acc_n = 0;
}

// ---- Short convolution

// Record 0 of a convolution phase's state: 6 past B*x values x this core's
// 64 channels, tap-major.
void dec_conv_load(const int8_t* rec) {
  conv_l = conv_k = acc_n = 0;
  const float* s = reinterpret_cast<const float*>(rec);
  for (int i = 0; i < 64 * (kTaps - 1); i += 16) aie::store_v(conv_state + i, aie::load_v<16>(s + i));
}

// The next channel: B, C and x rows, scales, then 7 f32 taps.
void dec_conv(const int8_t* q8, const int8_t* w) {
  const bfloat16* s = reinterpret_cast<const bfloat16*>(w + 3 * kH);
  const float* taps = reinterpret_cast<const float*>(w + 3 * kH + 3 * (kH / 32) * 2);
  acc_a[acc_n] = Dot(q8, kH, w, s);                         // B
  acc_b[acc_n] = Dot(q8, kH, w + kH, s + kH / 32);          // C
  acc_c[acc_n] = Dot(q8, kH, w + 2 * kH, s + 2 * (kH / 32));  // x
  for (int t = 0; t < kTaps; ++t) conv_taps[t][acc_n] = taps[t];
  ++acc_n;
}

// The last sixteen channels' convolution, as bf16. History is kept
// tap-major for these lanes: conv_state[t * 64 + channel].
void dec_conv_flush(bfloat16* out) {
  const int l0 = conv_l;
  aie::vector<float, 16> bx = aie::mul(aie::load_v<16>(acc_a), aie::load_v<16>(acc_c)).to_vector<float>();
  aie::accum<accfloat, 16> sum = aie::mul(bx, aie::load_v<16>(conv_taps[kTaps - 1]));
  for (int t = 0; t < kTaps - 1; ++t) sum = aie::mac(sum, aie::load_v<16>(conv_state + t * 64 + l0), aie::load_v<16>(conv_taps[t]));
  for (int t = 0; t + 1 < kTaps - 1; ++t) aie::store_v(conv_state + t * 64 + l0, aie::load_v<16>(conv_state + (t + 1) * 64 + l0));
  aie::store_v(conv_state + (kTaps - 2) * 64 + l0, bx);
  aie::accum<accfloat, 16> r;
  r.from_vector(aie::mul(sum.to_vector<float>(), aie::load_v<16>(acc_b)).to_vector<float>());
  aie::store_v(out, r.to_vector<bfloat16>());
  conv_l += 16;
  acc_n = 0;
}

// The next eight floats (48 chunks) of the updated convolution state.
void dec_conv_store(float* out) {
  aie::store_v(out, aie::load_v<8>(conv_state + 8 * conv_k++));
}

}  // extern "C"
