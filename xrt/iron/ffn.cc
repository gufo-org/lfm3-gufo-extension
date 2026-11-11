// LFM3 decode FFN block kernels (one core's part; see ffn.py).
// Vectors in DDR are in natural order. Core (c, j) owns hidden rows
// ((c * 8 + t) * 4 + j) * 8 + e for t < 8, e < 8, and intermediate channels
// ((c * 15 + s) * 4 + j) * 16 + e for s < 15, e < 16.
#include <stdint.h>
#include <aie_api/aie.hpp>

namespace {
constexpr int kH = 2048, kI = 7680;
constexpr float kEps = 1e-5f;

float SumSquares(const float* v, int n) {
  aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
  for (int i = 0; i < n; i += 16) {
    aie::vector<float, 16> x = aie::load_v<16>(v + i);
    acc = aie::mac(acc, x, x);
  }
  return aie::reduce_add(acc.to_vector<float>());
}

float Rsqrt(float x) {
  // Bit estimate and three Newton steps (scalar).
  union { float f; uint32_t u; } c{x};
  c.u = 0x5f3759dfu - (c.u >> 1);
  float y = c.f;
  for (int i = 0; i < 3; ++i) y = y * (1.5f - 0.5f * x * y * y);
  return y;
}

aie::vector<float, 16> Bf16ToFloat(const bfloat16* p) {
  aie::accum<accfloat, 16> a;
  a.from_vector(aie::load_v<16>(p));
  return a.to_vector<float>();
}

// Quantizes v[i] * s * w[i] (w may be null) to Q8_0: n codes at q, n / 32
// bf16 scales at qs.
void Quantize(const float* v, float s, const bfloat16* w, int n, int8_t* q, bfloat16* qs) {
  aie::set_rounding(aie::rounding_mode::conv_even);
  aie::set_saturation(aie::saturation_mode::saturate);
  for (int b = 0; b < n; b += 32) {
    aie::vector<float, 16> x0 = aie::load_v<16>(v + b), x1 = aie::load_v<16>(v + b + 16);
    aie::vector<float, 16> sv = aie::broadcast<float, 16>(s);
    if (w) {
      x0 = aie::mul(x0, aie::mul(Bf16ToFloat(w + b), sv).to_vector<float>()).to_vector<float>();
      x1 = aie::mul(x1, aie::mul(Bf16ToFloat(w + b + 16), sv).to_vector<float>()).to_vector<float>();
    } else {
      x0 = aie::mul(x0, sv).to_vector<float>();
      x1 = aie::mul(x1, sv).to_vector<float>();
    }
    const float amax = aie::max(aie::reduce_max(aie::abs(x0)), aie::reduce_max(aie::abs(x1)));
    const bfloat16 db = bfloat16(amax / 127.0f);
    const float d = float(db);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    aie::vector<float, 16> iv = aie::broadcast<float, 16>(inv);
    aie::vector<int8, 16> c0 = aie::to_fixed<int8>(aie::mul(x0, iv).to_vector<float>(), 0);
    aie::vector<int8, 16> c1 = aie::to_fixed<int8>(aie::mul(x1, iv).to_vector<float>(), 0);
    aie::store_v(q + b, aie::concat(c0, c1));
    qs[b / 32] = db;
  }
}

// One weight row of k codes (scales at `scales`) against quantized x.
float Dot(const int8_t* x, int k, const int8_t* row, const bfloat16* scales) {
#ifdef FFN_STUB_DOT
  return float(row[0]) * float(x[0]);
#endif
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

float Exp(float x) {
  if (x < -87.0f) return 0.0f;
  if (x > 88.0f) x = 88.0f;
  const float t = x * 1.4426950409f;
  const float fl = float(int(t >= 0 ? t : t - 1.0f));
  const float f = t - fl;
  float p = 1.5403530e-4f;
  p = p * f + 1.3333558e-3f;
  p = p * f + 9.6181291e-3f;
  p = p * f + 5.5504109e-2f;
  p = p * f + 2.4022651e-1f;
  p = p * f + 6.9314718e-1f;
  p = p * f + 1.0f;
  union { float f; int32_t i; } u{p};
  u.i += int32_t(fl) << 23;
  return u.f;
}

alignas(64) float gate_acc[16];
alignas(64) float up_acc[16];
alignas(64) float down_acc[8];
int gate_n, down_n;
}  // namespace

extern "C" {

// in: [h f32 H | o f32 H | post-norm w bf16 H | pre-norm w bf16 H].
// h += rmsnorm(o) * w_post in place; q8 = Q8_0(rmsnorm(h) * w_pre).
void ffn_pre(float* in, int8_t* q8) {
  float* h = in;
  const float* o = in + kH;
  const bfloat16* wpost = reinterpret_cast<const bfloat16*>(in + 2 * kH);
  const bfloat16* wpre = wpost + kH;
  const float so = Rsqrt(SumSquares(o, kH) / kH + kEps);
  for (int i = 0; i < kH; i += 16) {
    aie::vector<float, 16> t = aie::mul(aie::load_v<16>(o + i), aie::broadcast<float, 16>(so)).to_vector<float>();
    t = aie::mul(t, Bf16ToFloat(wpost + i)).to_vector<float>();
    aie::store_v(h + i, aie::add(aie::load_v<16>(h + i), t));
  }
  const float sh = Rsqrt(SumSquares(h, kH) / kH + kEps);
  Quantize(h, sh, wpre, kH, q8, reinterpret_cast<bfloat16*>(q8 + kH));
  gate_n = down_n = 0;
}

// Eight of this core's hidden rows (chunk t) of h.
void ffn_slice(const float* in, float* out, int32_t c, int32_t j, int32_t t) {
  aie::store_v(out, aie::load_v<8>(in + ((c * 8 + t) * 4 + j) * 8));
}

// One record: gate rows of two channels, then their up rows, then scales.
void ffn_gate_up(const int8_t* q8, const int8_t* w) {
  const bfloat16* s = reinterpret_cast<const bfloat16*>(w + 4 * kH);
  for (int r = 0; r < 2; ++r) {
    gate_acc[gate_n + r] = Dot(q8, kH, w + r * kH, s + r * (kH / 32));
    up_acc[gate_n + r] = Dot(q8, kH, w + (2 + r) * kH, s + (2 + r) * (kH / 32));
  }
  gate_n += 2;
}

// SiLU(gate) * up of the last sixteen channels, as bf16.
void ffn_gate_flush(bfloat16* out) {
  for (int i = 0; i < 16; ++i) {
    const float g = gate_acc[i];
    out[i] = bfloat16(g / (1.0f + Exp(-g)) * up_acc[i]);
  }
  gate_n = 0;
}

// in: bf16 activations (I of them).
void ffn_act_pre(const bfloat16* in, int8_t* q8, float* scratch) {
  bfloat16* qs = reinterpret_cast<bfloat16*>(q8 + kI);
  for (int c = 0; c < kI; c += 512) {
    for (int i = 0; i < 512; i += 16) aie::store_v(scratch + i, Bf16ToFloat(in + c + i));
    Quantize(scratch, 1.0f, nullptr, 512, q8 + c, qs + c / 32);
  }
}

// One down row (I codes, then scales).
void ffn_down(const int8_t* q8, const int8_t* w) {
  down_acc[down_n++] = Dot(q8, kI, w, reinterpret_cast<const bfloat16*>(w + kI));
}

void ffn_down_flush(float* out) {
  aie::store_v(out, aie::load_v<8>(down_acc));
  down_n = 0;
}

// Record r of a phase input: bytes [r * REC, (r + 1) * REC) of x, clipped.
void ffn_copy(const int8_t* rec, int8_t* x, int32_t r) {
  constexpr int kRec = 4 * kH + 4 * (kH / 32) * 2, kIn = 12 * kH;
  const int start = r * kRec, n = (kIn - start) < kRec ? kIn - start : kRec;
  for (int i = 0; i < n; i += 64) aie::store_v(x + start + i, aie::load_v<64>(rec + i));
}

}  // extern "C"
