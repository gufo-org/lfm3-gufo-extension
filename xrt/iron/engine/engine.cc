// Cores of the stack engine (engine.py): a phase interpreter for products and
// row operations, so whole tower segments run in one dispatch. Every phase
// starts with a header record on every stream; the cores derive their loop
// counts from it (eng_head).
//
// Records are 1096 words (4384 bytes).
//  B (weights, 512 codes x 8 columns): codes [block 16][k 32][n 8] int8, int16
//    fractions u [block 16][n 8] of the column's largest block scale, then the
//    8 columns' largest scales / 32767 (f32).
//  A (x, 8 rows x 256 codes, two per 512-code chunk): codes [block 8][k4 8][m 8][k 4] int16.
//  Row packets (row operations): f32 words, a whole number of records.
// Output records are 128 words: a product's 8 x 8 f32 tile (then 64 unused
// words), or 128 words of a row operation's output.
//
// Products: per 32-code block the int32 dot of int16 x and int8 codes times u
// adds into an acc64; the store scales by the columns' largest block scales.
// x's row scales travel as the rows' meta word to the next row operation.
//
// Row operations (two rows per stream; the parity picks the core's): from the
// packet [padding | constants | x row and/or meta | product row o],
// v = x + o r (+ bias) or silu(g r) (u r), then an optional RMS norm, then
// int16 codes and their row scale, or f32. All in f32 vectors.
#include <aie_api/aie.hpp>
#include <stdint.h>

namespace {
constexpr int kRec = 1096, kCodes = 4096, kMaxTiles = 32, kOut = 128;
using Acc = aie::accum<acc64, 16>;
using V = aie::vector<float, 16>;
alignas(128) Acc g_c[kMaxTiles * 4];  // [tile][group][quarter]; row operations: their outputs
alignas(64) float g_top[kMaxTiles * 8];
alignas(64) float g_meta[kOut];

enum Kind { kGemm = 0, kRow = 1 };
enum Flags {
  kO = 1,        // a product row o (times the meta row scale)
  kX = 2,        // an x row (H = K floats) is added (or is the input without o)
  kBias = 4,     // constants hold a bias (K floats, after the norm weight if any)
  kSwiGlu = 8,   // o is [gate | up] (2 K): v = silu(gate) up
  kNorm = 16,    // RMS norm with the constants' weight (K floats)
  kStoreX = 32,  // write v back as the new x row (then its meta)
  kF32 = 64,     // write the normed v as f32 instead of int16 codes
};

struct Head {
  int32_t kind;
  int32_t groups, chunks, segments, passes, tiles, total_chunks, total_tiles;  // products
  int32_t flags, recs, outs, k, consts, xw, ow;      // row operations
  float eps;
} g_h;

int32_t g_parity, g_a, g_rec, g_t, g_kc, g_out, g_in, g_pass, g_seg, g_seg_chunks, g_pass_tiles;
int32_t* g_cnt;

inline V Load(const float* p) { return aie::load_v<16>(p); }
inline V Mul(V a, V b) { return aie::mul(a, b).to_vector<float>(); }
inline V Splat(float x) { return aie::broadcast<float, 16>(x); }

// silu(x) = x / (1 + 2^(-x log2 e)); 2^t from its integer part (exponent bits) and
// a degree-5 polynomial of the fraction (relative error ~2e-7).
V Silu(V x) {
  V t = Mul(x, Splat(-1.4426950408889634f));
  t = aie::min(aie::max(t, Splat(-126.0f)), Splat(126.0f));
  aie::set_rounding(aie::rounding_mode::floor);
  const aie::vector<int32, 16> ti = aie::to_fixed<int32>(t, 0);
  const V f = aie::sub(t, aie::to_float(ti, 0));
  V p = Splat(0.0013333558f);
  p = aie::add(Mul(p, f), Splat(0.0096181291f));
  p = aie::add(Mul(p, f), Splat(0.0555041087f));
  p = aie::add(Mul(p, f), Splat(0.2402265070f));
  p = aie::add(Mul(p, f), Splat(0.6931471806f));
  p = aie::add(Mul(p, f), Splat(1.0f));
  const aie::vector<int32, 16> bits = aie::upshift(aie::add(ti, aie::broadcast<int32, 16>(127)), 23);
  const V e = Mul(p, bits.cast_to<float>());
  return Mul(x, aie::inv(aie::add(e, Splat(1.0f))));
}

float Rsqrt(float v) {
  union {
    float f;
    int32_t i;
  } u{v};
  u.i = 0x5f3759df - (u.i >> 1);
  float y = u.f;
  for (int i = 0; i < 3; ++i) y = y * (1.5f - 0.5f * v * y * y);
  return y;
}
}  // namespace

extern "C" {

// Header record -> loop counts: cnt[0] row input records (both parities),
// cnt[1] row output records, cnt[2] product passes, cnt[3] segments per pass,
// cnt[4] A records of the next segment, cnt[5] its B records (both parities),
// cnt[6] output records of the pass. Segments hold `chunks` chunks and passes
// `tiles` tiles, the last ones the remainder; the kernels update cnt[4..6].
void eng_head(const int32_t* rec, int32_t* cnt, int32_t parity) {
  g_parity = parity;
  g_cnt = cnt;
  g_h.kind = rec[0];
  for (int i = 0; i < 8; ++i) cnt[i] = 0;
  g_a = g_rec = g_t = g_kc = g_out = g_in = g_pass = g_seg = 0;
  if (g_h.kind == kGemm) {
    g_h.groups = rec[1], g_h.chunks = rec[2], g_h.segments = rec[3], g_h.passes = rec[4], g_h.tiles = rec[5];
    g_h.total_chunks = rec[6], g_h.total_tiles = rec[7];
    cnt[2] = g_h.passes, cnt[3] = g_h.segments;
  } else {
    g_h.flags = rec[1], g_h.recs = rec[2], g_h.outs = rec[3], g_h.k = rec[4];
    g_h.consts = rec[5], g_h.xw = rec[6], g_h.ow = rec[7];
    union {
      int32_t i;
      float f;
    } eps{rec[8]};
    g_h.eps = eps.f;
    cnt[0] = 2 * g_h.recs, cnt[1] = g_h.outs;
  }
}

// ---- products ----

namespace {
void StartSegment() {
  g_seg_chunks = g_h.total_chunks - g_seg * g_h.chunks;
  if (g_seg_chunks > g_h.chunks) g_seg_chunks = g_h.chunks;
  g_cnt[4] = 2 * g_h.groups * g_seg_chunks;
  g_cnt[5] = 2 * g_seg_chunks * g_pass_tiles;
  g_a = 0, g_kc = 0, g_t = 0, g_rec = 0;
}
}  // namespace

// A pass starts: zero the accumulators, size the pass and its first segment.
void eng_zero() {
  aie::set_rounding(aie::rounding_mode::floor);
  aie::set_saturation(aie::saturation_mode::none);
  g_pass_tiles = g_h.total_tiles - g_pass * g_h.tiles;
  if (g_pass_tiles > g_h.tiles) g_pass_tiles = g_h.tiles;
  for (int i = 0; i < g_h.groups * g_pass_tiles * 4; ++i) g_c[i] = aie::zeros<acc64, 16>();
  g_cnt[6] = g_h.groups * g_pass_tiles;
  g_seg = 0, g_out = 0;
  StartSegment();
}

// A records of a segment arrive [group][chunk][half].
void eng_acopy(const int8_t* rec, int8_t* a) {
  int8_t* dst = a + g_a * kCodes;
  for (int i = 0; i < kCodes; i += 64) aie::store_v(dst + i, aie::load_v<64>(rec + i));
  ++g_a;
}

void eng_mac(const int8_t* a, const int8_t* b) {
  const int own = (g_rec++ & 1) == g_parity;
  const int last = g_rec == 2 * g_seg_chunks * g_pass_tiles;
  if (!own) {
    if (last) ++g_seg, StartSegment();
    return;
  }
  const int groups = g_h.groups, chunk = g_kc;
  const int16_t* us = reinterpret_cast<const int16_t*>(b + kCodes);
  Acc* ct = g_c + g_t * groups * 4;
  for (int g = 0; g < groups; ++g) {
    const int16_t* ag = reinterpret_cast<const int16_t*>(a + (g * g_seg_chunks + chunk) * 2 * kCodes);
    Acc c0 = ct[4 * g], c1 = ct[4 * g + 1], c2 = ct[4 * g + 2], c3 = ct[4 * g + 3];
    for (int blk = 0; blk < 16; ++blk) {
      const int16_t* at = ag + blk * 256;
      const int8_t* bt = b + blk * 256;
      aie::mmul<8, 4, 8, int16, int8> m;
      m.mul(aie::load_v<32>(at), aie::load_v<32>(bt));
      for (int kb = 1; kb < 8; ++kb) m.mac(aie::load_v<32>(at + 32 * kb), aie::load_v<32>(bt + 32 * kb));
      const aie::vector<int32, 64> v = m.to_vector<int32>();
      const aie::vector<int16, 8> u = aie::load_v<8>(us + blk * 8);
      const aie::vector<int16, 16> s = aie::concat(u, u);
      c0 = aie::mac(c0, v.extract<16>(0), s);
      c1 = aie::mac(c1, v.extract<16>(1), s);
      c2 = aie::mac(c2, v.extract<16>(2), s);
      c3 = aie::mac(c3, v.extract<16>(3), s);
    }
    ct[4 * g] = c0, ct[4 * g + 1] = c1, ct[4 * g + 2] = c2, ct[4 * g + 3] = c3;
  }
  aie::store_v(g_top + 8 * g_t, aie::load_v<8>(reinterpret_cast<const float*>(b + kCodes + 256)));
  if (++g_t == g_pass_tiles) g_t = 0, ++g_kc;
  if (last) ++g_seg, StartSegment();
}

// Output records [tile][group]: the tile in f32, scaled by its columns' top scales.
void eng_store(float* out) {
  const int tile = g_out / g_h.groups;
  const Acc* src = g_c + g_out * 4;
  const aie::vector<float, 8> top8 = aie::load_v<8>(g_top + 8 * tile);
  const V top = aie::concat(top8, top8);
  for (int q = 0; q < 4; ++q) {
    const aie::vector<int32, 16> hi = src[q].to_vector<int32>(24);
    const aie::vector<int32, 16> lo = aie::bit_and(src[q].to_vector<int32>(0), aie::broadcast<int32, 16>(0xFFFFFF));
    const V v = aie::add(Mul(aie::to_float(hi, 0), Splat(16777216.0f)), aie::to_float(lo, 0));
    aie::store_v(out + 16 * q, Mul(v, top));
  }
  if (++g_out == g_h.groups * g_pass_tiles) ++g_pass;
}

// ---- row operations: the core's packet lands in `a`, codes / f32 outputs in g_c ----

void eng_rin(const float* rec, float* a) {
  if (g_in / g_h.recs == g_parity) {
    float* dst = a + (g_in % g_h.recs) * kRec;
    for (int i = 0; i < kRec; i += 8) aie::store_v(dst + i, aie::load_v<8>(rec + i));
  }
  ++g_in;
}

// Computes the row on the first call; every call emits one output record:
// [x region: v (if kStoreX) in 128-word records, then the meta record][codes or f32].
void eng_rout(float* a, float* out) {
  const int fl = g_h.flags, K = g_h.k;
  float* res = reinterpret_cast<float*>(g_c);
  const int nx = (fl & kStoreX) ? K / kOut + 1 : 1;
  float* x = a + g_h.xw;
  float* o = a + g_h.ow;
  const float* w = a + g_h.consts;
  float* v = (fl & kX) ? x : o;
  if (g_out == 0) {
    const V rm = Splat((fl & kO) ? x[(fl & kX) ? K : 0] : 1.0f);
    if (fl & kSwiGlu) {
      for (int i = 0; i < K; i += 16) aie::store_v(o + i, Mul(Silu(Mul(Load(o + i), rm)), Mul(Load(o + K + i), rm)));
    } else if (fl & kO) {
      const float* bias = w + ((fl & kNorm) ? K : 0);
      for (int i = 0; i < K; i += 16) {
        V t = Mul(Load(o + i), rm);
        if (fl & kX) t = aie::add(t, Load(x + i));
        if (fl & kBias) t = aie::add(t, Load(bias + i));
        aie::store_v(v + i, t);
      }
    }
    float scale = 1.0f;
    if (fl & kNorm) {
      aie::accum<accfloat, 16> ss = aie::zeros<accfloat, 16>();
      for (int i = 0; i < K; i += 16) ss = aie::mac(ss, Load(v + i), Load(v + i));
      scale = Rsqrt(aie::reduce_add(ss.to_vector<float>()) / float(K) + g_h.eps);
    }
    // y = v scale (w): f32 into res, or int16 codes into res with their scale in the meta
    V top = aie::zeros<float, 16>();
    for (int i = 0; i < K; i += 16) {
      V y = Mul(Load(v + i), Splat(scale));
      if (fl & kNorm) y = Mul(y, Load(w + i));
      aie::store_v(res + i, y);
      top = aie::max(top, aie::abs(y));
    }
    float r = 1.0f;
    if (!(fl & kF32)) {
      const float t = aie::reduce_max(top);
      r = t > 0 ? t / 32767.0f : 1.0f;
      const V inv = Splat(t > 0 ? 32767.0f / t : 0.0f);
      aie::set_rounding(aie::rounding_mode::symmetric_inf);  // round half away from zero
      int16_t* codes = reinterpret_cast<int16_t*>(res);
      for (int i = 0; i < K; i += 16) aie::store_v(codes + i, aie::to_fixed<int16>(Mul(Load(res + i), inv), 0));
    }
    for (int i = 0; i < kOut; i += 16) aie::store_v(g_meta + i, aie::zeros<float, 16>());
    g_meta[0] = r;
  }
  const float* src = g_out < nx - 1 ? v + g_out * kOut : g_out == nx - 1 ? g_meta : res + (g_out - nx) * kOut;
  for (int i = 0; i < kOut; i += 16) aie::store_v(out + i, Load(src + i));
  ++g_out;
}

}  // extern "C"
