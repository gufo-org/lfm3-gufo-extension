// Q/K preparation and attention kernels (npu/attention.cpp) against host FP32.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "../attention.hpp"
#include "../host_ops.hpp"
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
  return wq < 2e-4 && wk < 4e-3 && wv < 4e-3 && wg < 1e-5 ? 0 : 1;
}
