// Short-convolution kernel (npu/probe/conv.loom) against the host step for
// 8 rows: rows 0-4 one session (slot 1, consecutive), rows 5-7 other slots.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>

#include "npu_runtime.hpp"

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  std::stringstream text;
  text << in.rdbuf();
  lfm3::npu::Runtime npu;
  const int id = npu.Compile(text.str(), argv[1], "@conv8");
  const int H = 2048, R = 8, L = 7, S = 4;
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> proj(R * 3 * H), hist(S * 6 * H), w(L * H), out(R * H);
  for (auto& v : proj) v = u(rng);
  for (auto& v : hist) v = u(rng);
  for (auto& v : w) v = u(rng);
  std::vector<std::int32_t> meta(R * 16 + 16, 0);
  const int slots[R] = {1, 1, 1, 1, 1, 0, 2, 3};
  for (int r = 0; r < R; ++r) meta[r] = slots[r];  // row r's slot at element r
  // The kernel reads 16 i32 from element r; slot is its first.
  auto ref_hist = hist;
  std::vector<float> ref(R * H);
  for (int r = 0; r < R; ++r) {
    float* h = &ref_hist[std::size_t(slots[r]) * 6 * H];
    for (int ch = 0; ch < H; ++ch) {
      const float B = proj[r * 3 * H + ch], C = proj[r * 3 * H + H + ch], x = proj[r * 3 * H + 2 * H + ch];
      const float bx = B * x;
      float acc = w[6 * H + ch] * bx;
      for (int t = 0; t < 6; ++t) acc += w[t * H + ch] * h[t * H + ch];
      ref[r * H + ch] = C * acc;
      for (int t = 0; t < 5; ++t) h[t * H + ch] = h[(t + 1) * H + ch];
      h[5 * H + ch] = bx;
    }
  }
  const int bp = npu.Upload(std::span<const float>(proj)), bh = npu.Upload(std::span<const float>(hist)),
            bw = npu.Upload(std::span<const float>(w)), bm = npu.Upload(std::span<const std::int32_t>(meta)),
            bo = npu.Alloc(out.size() * 4);
  const int bind[] = {bp, bh, bw, bm, bo};
  npu.Dispatch(id, bind);
  npu.Read(bo, 0, out.data(), out.size() * 4);
  std::vector<float> got_hist(hist.size());
  npu.Read(bh, 0, got_hist.data(), got_hist.size() * 4);
  double worst = 0, worst_h = 0;
  for (std::size_t i = 0; i < out.size(); ++i) worst = std::max(worst, double(std::abs(out[i] - ref[i])));
  for (std::size_t i = 0; i < hist.size(); ++i) worst_h = std::max(worst_h, double(std::abs(got_hist[i] - ref_hist[i])));
  std::printf("conv out max err %.3g, history max err %.3g\n", worst, worst_h);
  return worst < 1e-5 && worst_h < 1e-5 ? 0 : 1;
}
