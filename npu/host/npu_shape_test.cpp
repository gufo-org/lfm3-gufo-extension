// Generated projection kernels (npu/program.cpp) for every shape the LFM3
// family needs, against a CPU reference over the same encoded weights.
// Usage: npu_shape_test [k m blocks ...]
#include <chrono>
#include <fstream>
#include <iterator>
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
  const bool chain = std::getenv("CHAIN") != nullptr;  // K-multi with on-chip column sums
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
      const bool kmulti = (chain || (std::getenv("KMULTI") && c.blocks > 1)) && shape.split == g::Split::kColumns;
      const bool dual = chain && std::getenv("DUAL") && g::DualBlocksPerCore(shape) > 0;
      std::string program = dual ? g::ProjectionDualProgram(shape) : kmulti ? g::ProjectionKMultiProgram(shape, chain) : std::string();
      if (const char* f = std::getenv("KM_FILE")) {  // experiment: a hand-edited program
        std::ifstream in(f);
        program.assign(std::istreambuf_iterator<char>(in), {});
      }
      kernel = kmulti ? npu.Compile(program, "projection.loom", g::ProjectionRoot(shape) + (dual ? "_kdual" : chain ? "_kchain" : "_kmulti"))
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
    const std::uint32_t parts = split == Split::kColumns && !chain ? 8 : 1;
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
    if (chain && std::getenv("CMP")) {  // chain against the eight-part kernel, both on the device
      const int k8 = npu.Compile(c.blocks > 1 ? g::ProjectionKMultiProgram(shape) : g::ProjectionProgram(shape), "projection.loom",
                                 g::ProjectionRoot(shape) + (c.blocks > 1 ? "_kmulti" : ""));
      const std::size_t bo8 = std::size_t(8) * shape.m * 8;
      const int by8 = npu.Alloc(bo8 * c.blocks * 4);
      const int bind8[] = {bx, bw, by8};
      npu.Dispatch(k8, bind8);
      std::vector<float> b8(bo8 * c.blocks), y8(std::size_t(R) * c.m);
      npu.Read(by8, 0, b8.data(), b8.size() * 4);
      for (std::uint32_t b = 0; b < c.blocks; ++b)
        UnpackOutputs(std::span(b8).subspan(b * bo8, bo8), 8, c.m, std::span(y8).subspan(std::size_t(b) * 8 * c.m, 8 * c.m), 8, shape.m);
      double d = 0, mx = 0, rel = 0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        d = std::max(d, double(std::abs(y[i] - y8[i])));
        mx = std::max(mx, double(std::abs(y8[i])));
        if (std::abs(y8[i]) > 1e-3) rel = std::max(rel, double(std::abs(y[i] - y8[i]) / std::abs(y8[i])));
      }
      std::printf("chain vs parts: max abs diff %.3g of %.3g, max rel %.3g\n", d, mx, rel);
    }
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
