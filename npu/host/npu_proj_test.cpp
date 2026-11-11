// Every LFM3 projection shape on the NPU against a CPU reference over the
// same encoded weights. Usage: npu_proj_test <lfm3_proj.loom> [root ...]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>

#include "bfp.hpp"
#include "npu_runtime.hpp"

using namespace lfm3::npu;

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  std::stringstream text;
  text << in.rdbuf();
  const std::string source = text.str();
  Runtime npu;
  struct Shape { const char* root; std::uint32_t k, m; Split split; };
  std::vector<Shape> shapes = {
      {"@proj_m_k2048_m256", 2048, 256, Split::kRows},
      {"@proj_k_k7680_m2048", 7680, 2048, Split::kColumns},
      {"@proj_m_k2048_m2048", 2048, 2048, Split::kRows},
      {"@proj_m_k2048_m3136", 2048, 3136, Split::kRows},
      {"@proj_m_k2048_m6144", 2048, 6144, Split::kRows},
      {"@proj_m_k2048_m15360", 2048, 15360, Split::kRows},
      {"@proj_m_k2048_m128000", 2048, 128000, Split::kRows},
  };
  if (argc > 2) {
    std::vector<Shape> picked;
    for (int i = 2; i < argc; ++i)
      for (auto& s : shapes) if (std::string(s.root) == argv[i]) picked.push_back(s);
    shapes = picked;
  }
  int failures = 0;
  for (const auto& s : shapes) {
    const auto t0 = std::chrono::steady_clock::now();
    const int kernel = npu.Compile(source, argv[1], s.root);
    const double compile_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::mt19937 rng(s.m + s.k);
    std::uniform_real_distribution<float> u(-1, 1);
    const std::uint32_t B = 8;
    std::vector<float> x(std::size_t(B) * s.k);
    for (auto& v : x) v = u(rng);
    auto weights = PackWeights(s.m, s.k, s.m, s.split, [&](std::uint32_t r, std::span<float> out) {
      std::mt19937 g(r * 7919u + 1);
      std::uniform_real_distribution<float> w(-0.05f, 0.05f);
      for (auto& v : out) v = w(g);
    });
    std::vector<std::uint16_t> xb(std::size_t(s.k) * 8);
    PackActivations(x, B, s.k, xb);
    const std::uint32_t parts = s.split == Split::kColumns ? 8 : 1;
    const int bx = npu.Upload(std::span<const std::uint16_t>(xb));
    const int bw = npu.Upload(std::span<const std::uint8_t>(weights));
    const int by = npu.Alloc(std::uint64_t(parts) * s.m * 8 * 4);
    const int bind[] = {bx, bw, by};
    npu.Dispatch(kernel, bind);
    const int reps = s.m > 20000 ? 3 : 10;
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) npu.Dispatch(kernel, bind);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / reps;
    std::vector<float> blocked(std::size_t(parts) * s.m * 8), y(std::size_t(B) * s.m);
    npu.Read(by, 0, blocked.data(), blocked.size() * 4);
    UnpackOutputs(blocked, B, s.m, y, parts);
    // Reference: decoded weights times BF16 x, in double.
    double worst = 0, scale = 0;
    const std::uint32_t tiles = s.k / 8, groups = s.m / 8;
    for (std::uint32_t r = 0; r < s.m; r += std::max(1u, s.m / 512)) {
      const std::uint32_t g = r / 8, n = r % 8;
      for (std::uint32_t b = 0; b < B; ++b) {
        double ref = 0;
        for (std::uint32_t j = 0; j < tiles; ++j) {
          std::size_t tile_index;
          if (s.split == Split::kRows) {
            const std::uint32_t half = tiles / 2;
            tile_index = (std::size_t(g) * 2 + j / half) * half + j % half;
          } else {
            const std::uint32_t t = tiles / 8;
            tile_index = (std::size_t(j / t) * groups + g) * t + j % t;
          }
          const std::uint8_t* row = weights.data() + tile_index * 72 + n * 9;
          for (int c = 0; c < 8; ++c)
            ref += double(DecodeBfp(row, c)) * FromBF16(ToBF16(x[std::size_t(b) * s.k + j * 8 + c]));
        }
        worst = std::max(worst, std::abs(ref - y[std::size_t(b) * s.m + r]));
        scale = std::max(scale, std::abs(ref));
      }
    }
    const bool ok = worst <= 0.02 * scale + 1e-3;
    failures += !ok;
    const double gb = double(weights.size()) / 1e9;
    std::printf("%-26s compile %5.1f s  %8.3f ms  %6.1f GB/s  max err %.4f of %.3f  %s\n", s.root,
                compile_s, ms, gb / (ms / 1e3), worst, scale, ok ? "ok" : "FAIL");
    npu.Free(bx);
    npu.Free(bw);
    npu.Free(by);
  }
  return failures;
}
