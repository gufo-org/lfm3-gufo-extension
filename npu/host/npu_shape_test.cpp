// Generated projection kernels (npu/program.cpp) for every shape the LFM3
// family needs, against a CPU reference over the same encoded weights.
// Usage: npu_shape_test [k m blocks ...]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "../program.hpp"
#include "bfp.hpp"
#include "npu_runtime.hpp"

using namespace lfm3::npu;
namespace g = gufo::models::lfm3::npu;

int main(int argc, char** argv) {
  Runtime npu;
  struct Case { std::uint32_t k, m, blocks; };
  std::vector<Case> cases = {{2048, 256, 1}, {2048, 256, 4}, {7680, 2048, 2}, {768, 2304, 3},
                             {1792, 768, 2}, {2816, 1024, 2}, {1024, 5120, 2}, {2560, 1024, 2},
                             {4096, 2048, 1}, {768, 1024, 2}, {1024, 24204, 1}};
  if (argc > 3) {
    cases.clear();
    for (int i = 1; i + 2 < argc; i += 3)
      cases.push_back({std::uint32_t(std::stoul(argv[i])), std::uint32_t(std::stoul(argv[i + 1])),
                       std::uint32_t(std::stoul(argv[i + 2]))});
  }
  int failures = 0;
  for (const auto& c : cases) {
    auto shape = g::PlanProjection(c.k, c.m, c.blocks);
    if (std::getenv("FORCEK") && shape.split == g::Split::kRows) {  // K split at any K (multiple of 512)
      shape.split = g::Split::kColumns;
      shape.k = (c.k + 511) / 512 * 512;
      shape.m = (c.m + 7) / 8 * 8;
    }
    const auto t0 = std::chrono::steady_clock::now();
    int kernel;
    try {
      const bool kmulti = std::getenv("KMULTI") && c.blocks > 1 && shape.split == g::Split::kColumns;
      kernel = kmulti ? npu.Compile(g::ProjectionKMultiProgram(shape), "projection.loom", g::ProjectionRoot(shape) + "_kmulti")
                      : npu.Compile(g::ProjectionProgram(shape), "projection.loom", g::ProjectionRoot(shape));
    } catch (const std::exception& e) {
      std::printf("%s: %s\n", g::ProjectionRoot(shape).c_str(), e.what());
      ++failures;
      continue;
    }
    const double compile_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const Split split = shape.split == g::Split::kRows ? Split::kRows : Split::kColumns;
    const std::uint32_t R = 8 * c.blocks;
    std::mt19937 rng(c.k * 31 + c.m);
    std::uniform_real_distribution<float> u(-1, 1);
    std::vector<float> x(std::size_t(R) * shape.k, 0.0f);
    for (std::uint32_t r = 0; r < R; ++r)
      for (std::uint32_t k = 0; k < c.k; ++k) x[std::size_t(r) * shape.k + k] = u(rng);
    auto weights = PackWeights(c.m, shape.k, shape.m, split, [&](std::uint32_t r, std::span<float> out) {
      std::mt19937 gr(r * 7919u + 1);
      std::uniform_real_distribution<float> w(-0.05f, 0.05f);
      for (std::uint32_t k = 0; k < out.size(); ++k) out[k] = k < c.k ? w(gr) : 0.0f;
    });
    std::vector<std::uint16_t> xb(std::size_t(shape.k) * 8 * c.blocks);
    for (std::uint32_t b = 0; b < c.blocks; ++b)
      PackActivations(std::span(x).subspan(std::size_t(b) * 8 * shape.k, 8 * shape.k), 8, shape.k,
                      std::span(xb).subspan(std::size_t(b) * 8 * shape.k, 8 * shape.k));
    const std::uint32_t parts = split == Split::kColumns ? 8 : 1;
    const int bx = npu.Upload(std::span<const std::uint16_t>(xb));
    const int bw = npu.Upload(std::span<const std::uint8_t>(weights));
    const std::size_t block_out = std::size_t(parts) * shape.m * 8;
    const int by = npu.Alloc(block_out * c.blocks * 4);
    const int bind[] = {bx, bw, by};
    npu.Dispatch(kernel, bind);
    const auto t1 = std::chrono::steady_clock::now();
    const int reps = 3;
    for (int i = 0; i < reps; ++i) npu.Dispatch(kernel, bind);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / reps;
    std::vector<float> blocked(block_out * c.blocks), y(std::size_t(R) * c.m);
    npu.Read(by, 0, blocked.data(), blocked.size() * 4);
    for (std::uint32_t b = 0; b < c.blocks; ++b)
      UnpackOutputs(std::span(blocked).subspan(b * block_out, block_out), 8, c.m,
                    std::span(y).subspan(std::size_t(b) * 8 * c.m, 8 * c.m), parts, shape.m);
    const std::uint32_t tiles = shape.k / 8, groups = shape.m / 8;
    double worst = 0, scale = 0;
    for (std::uint32_t r = 0; r < c.m; r += std::max(1u, c.m / 256)) {
      const std::uint32_t gi = r / 8, n = r % 8;
      for (std::uint32_t row = 0; row < R; row += 3) {
        double ref = 0;
        for (std::uint32_t j = 0; j < tiles; ++j) {
          std::size_t ti;
          if (split == Split::kRows) {
            const std::uint32_t half = tiles / 2;
            ti = (std::size_t(gi) * 2 + j / half) * half + j % half;
          } else {
            const std::uint32_t t = tiles / 8;
            ti = (std::size_t(j / t) * groups + gi) * t + j % t;
          }
          const std::uint8_t* wr = weights.data() + ti * 72 + n * 9;
          for (int cc = 0; cc < 8; ++cc)
            ref += double(DecodeBfp(wr, cc)) * FromBF16(ToBF16(x[std::size_t(row) * shape.k + j * 8 + cc]));
        }
        worst = std::max(worst, std::abs(ref - y[std::size_t(row) * c.m + r]));
        scale = std::max(scale, std::abs(ref));
      }
    }
    const bool ok = worst <= 0.02 * scale + 1e-3;
    failures += !ok;
    std::printf("%-30s compile %4.1f s  %8.3f ms  max err %.4f of %.3f  %s\n",
                g::ProjectionRoot(shape).c_str(), compile_s, ms, worst, scale, ok ? "ok" : "FAIL");
    npu.Free(bx);
    npu.Free(bw);
    npu.Free(by);
  }
  return failures;
}
