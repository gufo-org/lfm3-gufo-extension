// Q/K preparation and attention kernels (npu/attention.cpp) against host FP32.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "../attention.hpp"
#include "../host_ops.hpp"
#include "bfp.hpp"
#include "npu_runtime.hpp"

using namespace gufo::models::lfm3::npu;

int main() {
  lfm3::npu::Runtime npu;
  AttentionShape s{16, 4, 128, 392, true, 64, 4, 64, 9, false};
  const std::uint32_t D = s.dim, H = s.heads * D, KV = s.kv_heads * D, half = D / 2, R = s.rows;
  const std::uint32_t G = s.qkvg_groups, B = (R + 7) / 8;
  const int prep = npu.Compile(QkPrepProgram(s, "prep"), "prep.loom", "@prep");
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> u(-2, 2);
  std::vector<float> yb(std::size_t(B) * G * 64 + 64), qn(D), kn(D), cosv(s.table_rows * half), sinv(cosv.size());
  for (auto& v : yb) v = u(rng);
  for (auto& v : qn) v = 0.5f + std::abs(u(rng)) / 4;
  for (auto& v : kn) v = 0.5f + std::abs(u(rng)) / 4;
  for (std::uint32_t p = 0; p < s.table_rows; ++p)
    for (std::uint32_t i = 0; i < half; ++i) {
      const float a = p * std::pow(1e6f, -2.0f * i / D);
      cosv[p * half + i] = std::cos(a);
      sinv[p * half + i] = std::sin(a);
    }
  std::vector<std::int32_t> meta(R * 16, 0);
  for (std::uint32_t r = 0; r < R; ++r) {
    meta[r * 16 + 0] = r < 5 ? 1 : int(r % 4);
    meta[r * 16 + 1] = r < 5 ? 40 + int(r) : int(r * 3);
  }
  const int bp = npu.Upload(std::span<const float>(yb)), bq = npu.Upload(std::span<const float>(qn)),
            bk = npu.Upload(std::span<const float>(kn)), bc = npu.Upload(std::span<const float>(cosv)),
            bs = npu.Upload(std::span<const float>(sinv)), bm = npu.Upload(std::span<const std::int32_t>(meta)),
            bqo = npu.Alloc(R * H * 4), bgo = npu.Alloc(R * 16 * 4);
  std::vector<std::uint16_t> zero(std::size_t(s.slots) * s.capacity * KV, 0);
  const int bkp = npu.Upload(std::span<const std::uint16_t>(zero)), bvp = npu.Upload(std::span<const std::uint16_t>(zero));
  const int bind[] = {bp, bq, bk, bc, bs, bm, bqo, bgo, bkp, bvp};
  npu.Dispatch(prep, bind);
  std::vector<float> qo(R * H), go(R * 16);
  std::vector<_Float16> kp(zero.size()), vp(zero.size());
  npu.Read(bqo, 0, qo.data(), qo.size() * 4);
  npu.Read(bgo, 0, go.data(), go.size() * 4);
  npu.Read(bkp, 0, kp.data(), kp.size() * 2);
  npu.Read(bvp, 0, vp.data(), vp.size() * 2);
  double wq = 0, wk = 0, wv = 0, wg = 0;
  {
    std::size_t nz = 0, first = 0;
    for (std::size_t i = 0; i < kp.size(); ++i) if (float(kp[i]) != 0) { if (!nz) first = i; ++nz; }
    std::printf("k pool nonzero %zu first at %zu (row %zu); rows:", nz, first, first / KV);
    for (std::size_t r = 0; r < kp.size() / KV; ++r) if (float(kp[r * KV]) != 0 || float(kp[r * KV + 1]) != 0) std::printf(" %zu", r);
    std::printf("\n");
  }
  for (std::uint32_t r = 0; r < R; ++r) {
    std::vector<float> row(G * 8);
    for (std::uint32_t i = 0; i < G * 8; ++i) row[i] = yb[((std::size_t(r / 8) * G + i / 8) * 8 + r % 8) * 8 + i % 8];
    const int pos = meta[r * 16 + 1], slot = meta[r * 16];
    auto norm_rope = [&](float* x, std::uint32_t heads, const float* w) {
      for (std::uint32_t h = 0; h < heads; ++h) {
        RmsNorm(x + h * D, D, w, 1e-5f);
        for (std::uint32_t i = 0; i < half; ++i) {
          const float a = x[h * D + i], b = x[h * D + i + half];
          const float c = cosv[pos * half + i], sn = sinv[pos * half + i];
          x[h * D + i] = a * c - b * sn;
          x[h * D + i + half] = b * c + a * sn;
        }
      }
    };
    norm_rope(row.data(), s.heads, qn.data());
    norm_rope(row.data() + H, s.kv_heads, kn.data());
    for (std::uint32_t i = 0; i < H; ++i) wq = std::max(wq, double(std::abs(qo[r * H + i] - row[i])));
    const std::size_t at = (std::size_t(slot) * s.capacity + pos % s.capacity) * KV;
    for (std::uint32_t i = 0; i < KV; ++i) {
      wk = std::max(wk, double(std::abs(float(kp[at + i]) - row[H + i])));
      wv = std::max(wv, double(std::abs(float(vp[at + i]) - row[H + KV + i])));
    }
    if (r == 0)
      for (int i = 0; i < 4; ++i)
        std::printf("k[%d] got %g (bits %04x) want %g\n", i, float(kp[at + i]), unsigned(reinterpret_cast<std::uint16_t&>(kp[at + i])), row[H + i]);
    for (std::uint32_t h = 0; h < s.heads; ++h)
      wg = std::max(wg, double(std::abs(go[r * 16 + h] - row[H + 2 * KV + h])));
  }
  std::printf("prep q err %.3g  k err %.3g  v err %.3g  gate err %.3g\n", wq, wk, wv, wg);
  bool ok = wq < 2e-4 && wk < 4e-3 && wv < 4e-3 && wg < 1e-5;
  {  // Attention over random rings: causal, sliding (first > 0), several slots.
    AttentionShape a = s;
    a.capacity = 128;
    const int attend = npu.Compile(AttendProgram(a, "attend"), "attend.loom", "@attend");
    std::uniform_real_distribution<float> uk(-1, 1);
    std::vector<_Float16> kpool(std::size_t(a.slots) * a.capacity * KV), vpool(kpool.size());
    for (auto& v : kpool) v = _Float16(uk(rng));
    for (auto& v : vpool) v = _Float16(uk(rng));
    std::vector<float> q(R * H), gate(R * 16);
    for (auto& v : q) v = uk(rng);
    for (auto& v : gate) v = uk(rng) * 3;
    std::vector<std::int32_t> am(R * 16, 0), groups(64, 0);
    const int lasts[] = {0, 5, 31, 32, 77, 100, 127, 40, 63};
    const int firsts[] = {0, 0, 0, 0, 22, 60, 0, 9, 32};
    for (std::uint32_t r = 0; r < R; ++r) {
      am[r * 16 + 0] = int(r % 4);
      am[r * 16 + 2] = firsts[r];
      am[r * 16 + 3] = lasts[r];
      am[r * 16 + 4] = -1;
    }
    const int bq2 = npu.Upload(std::span<const float>(q)), bg2 = npu.Upload(std::span<const float>(gate)),
              bk2 = npu.Upload(std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(kpool.data()), kpool.size())),
              bv2 = npu.Upload(std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(vpool.data()), vpool.size())),
              bm2 = npu.Upload(std::span<const std::int32_t>(am)), bgr = npu.Upload(std::span<const std::int32_t>(groups));
    const std::uint32_t OG = H / 8, OB = (R + 7) / 8;
    const int bo = npu.Alloc((std::size_t(OB) * OG * 64 + 64) * 2);
    const std::int32_t cn[16] = {std::int32_t(R)};
    const int bcn = npu.Upload(std::span<const std::int32_t>(cn, 16));
    const int b2[] = {bq2, bg2, bk2, bv2, bm2, bgr, bo, bcn};
    npu.Dispatch(attend, b2);
    std::vector<std::uint16_t> ob(std::size_t(OB) * OG * 64 + 64);
    npu.Read(bo, 0, ob.data(), ob.size() * 2);
    double worst = 0;
    for (std::uint32_t r = 0; r < R; ++r)
      for (std::uint32_t h = 0; h < a.heads; ++h) {
        const std::uint32_t kh = h / (a.heads / a.kv_heads);
        std::vector<float> sc;
        float mx = -1e30f;
        for (int t2 = firsts[r]; t2 <= lasts[r]; ++t2) {
          const std::size_t at = (std::size_t(r % 4) * a.capacity + t2 % a.capacity) * KV + kh * D;
          float d = 0;
          for (std::uint32_t i = 0; i < D; ++i) d += q[r * H + h * D + i] * float(kpool[at + i]);
          sc.push_back(d / std::sqrt(float(D)));
          mx = std::max(mx, sc.back());
        }
        float den = 0;
        for (auto& v : sc) { v = std::exp(v - mx); den += v; }
        for (std::uint32_t i = 0; i < D; ++i) {
          float o = 0;
          for (int t2 = firsts[r]; t2 <= lasts[r]; ++t2) {
            const std::size_t at = (std::size_t(r % 4) * a.capacity + t2 % a.capacity) * KV + kh * D;
            o += sc[t2 - firsts[r]] / den * float(vpool[at + i]);
          }
          o *= Sigmoid(gate[r * 16 + h]);
          const std::uint32_t e = h * D + i;
          const float got = lfm3::npu::FromBF16(ob[((std::size_t(r / 8) * OG + e / 8) * 8 + r % 8) * 8 + e % 8]);
          worst = std::max(worst, double(std::abs(got - o)));
        }
      }
    std::printf("attend max err %.3g\n", worst);
    ok = ok && worst < 2e-2;
  }
  return ok ? 0 : 1;
}
