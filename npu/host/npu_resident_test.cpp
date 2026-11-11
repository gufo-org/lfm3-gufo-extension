// Resident decode FFN block (npu/resident.cpp) against the same stages as
// separate dispatches (norm, K-split gate/up, SwiGLU on partials, down,
// reduce, norm) at LFM3 1.7B widths, for row 0. Reports timing of both.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "../glue.hpp"
#include "../program.hpp"
#include "../resident.hpp"
#include "../rowops.hpp"
#include "bfp.hpp"
#include "npu_runtime.hpp"

using namespace gufo::models::lfm3::npu;
namespace h = lfm3::npu;

int main() {
  const std::uint32_t H = 2048, I = 7680, kMax = 256;
  const float eps = 1e-5f;
  h::Runtime npu;
  std::mt19937 rng(5);
  auto vec = [&](std::size_t n, float lo, float hi) {
    std::vector<float> v(n);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& x : v) x = d(rng);
    return v;
  };
  auto ksplit = [&](std::uint32_t k, std::uint32_t m) {
    ProjectionShape s = PlanProjection(k, m, 1);
    s.split = Split::kColumns;
    s.k = (k + 511) / 512 * 512;
    s.m = (m + 7) / 8 * 8;
    auto bytes = h::PackWeights(m, s.k, s.m, h::Split::kColumns, [&](std::uint32_t r, std::span<float> out) {
      std::mt19937 g(r * 7919u + k);
      std::uniform_real_distribution<float> d(-0.03f, 0.03f);
      for (auto& v : out) v = d(g);
    });
    return std::make_pair(s, npu.Upload(std::span<const std::uint8_t>(bytes)));
  };
  auto [gu_s, wgu] = ksplit(H, 2 * I);
  auto [d_s, wd] = ksplit(I, H);
  auto dyn = [&](RowKernel k) {
    k.rows = kMax;
    k.dynamic_rows = true;
    k.name += "_dyn";
    return k;
  };
  const RowKernel n_op = dyn(NormPackKernel(H, 8, Residual::kBlocked, true, false, eps));
  const RowKernel swiglu = dyn(SwiGluPackKernel(I, 8, true));
  const RowKernel reduce = dyn(ReducePartialKernel(H, 8));
  const RowKernel n_ffn = dyn(NormPackKernel(H, 8, Residual::kRows, true, false, eps));

  const std::size_t big = std::size_t(kMax) * 2 * I * 8 * 8 + 4096;
  auto x0 = vec(std::size_t(kMax) * H, -2, 2), yout = vec(std::size_t(kMax) * H * 8 / 8 + 4096, -1, 1);
  auto wop = vec(2 * H, 0.5f, 1.5f), wffn = vec(2 * H, 0.5f, 1.5f);
  const int bx = npu.Alloc(big), by = npu.Upload(std::span<const float>(yout)), bwo = npu.Upload(std::span<const float>(wop)),
            bx2 = npu.Alloc(big), bxb = npu.Alloc(big), bygu = npu.Alloc(big), bxf = npu.Alloc(big), byd = npu.Alloc(big),
            bo = npu.Alloc(big), bwf = npu.Upload(std::span<const float>(wffn));
  const std::int32_t count[16] = {1};
  const int bc = npu.Upload(std::span<const std::int32_t>(count, 16));
  auto reset = [&] { npu.Write(bx, 0, x0.data(), x0.size() * 4); };

  struct Stage { std::string program, root; std::vector<int> bind; };
  std::vector<Stage> stages = {
      {RowKernelProgram(n_op), "@" + n_op.name, {bx, by, bwo, bx2, bxb, bc}},
      {ProjectionProgram(gu_s), ProjectionRoot(gu_s), {bxb, wgu, bygu}},
      {RowKernelProgram(swiglu), "@" + swiglu.name, {bygu, bxf, bc}},
      {ProjectionProgram(d_s), ProjectionRoot(d_s), {bxf, wd, byd}},
      {RowKernelProgram(reduce), "@" + reduce.name, {byd, bo, bc}},
      {RowKernelProgram(n_ffn), "@" + n_ffn.name, {bx2, bo, bwf, bx, bxb, bc}},
  };
  std::vector<int> ids;
  for (auto& s : stages) ids.push_back(npu.Compile(s.program, s.root.substr(1) + ".loom", s.root));
  int resident;
  try {
    resident = npu.Compile(ResidentFfnProgram(H, I, eps, "ffn_resident"), "ffn_resident.loom", "@ffn_resident");
  } catch (const std::exception& e) {
    std::printf("resident compile failed: %s\n", e.what());
    FILE* f = std::fopen("ffn_resident.loom", "w");
    std::fputs(ResidentFfnProgram(H, I, eps, "ffn_resident").c_str(), f);
    std::fclose(f);
    return 1;
  }
  auto run_split = [&] {
    for (std::size_t i = 0; i < stages.size(); ++i) npu.Enqueue(ids[i], stages[i].bind);
    npu.Finish();
  };
  const int rb[] = {bx, by, bwo, bx2, bxb, wgu, bygu, bxf, wd, byd, bwf};
  auto run_resident = [&] { npu.Dispatch(resident, rb); };
  auto snap = [&] {
    std::vector<float> x(H), x2(H);
    std::vector<std::uint16_t> xb(std::size_t(H) * 8);
    npu.Read(bx, 0, x.data(), H * 4);
    npu.Read(bx2, 0, x2.data(), H * 4);
    npu.Read(bxb, 0, xb.data(), xb.size() * 2);
    std::vector<float> xb0(H);
    for (std::uint32_t e = 0; e < H; ++e) {
      const std::uint32_t u = std::uint32_t(xb[(e / 8) * 64 + e % 8]) << 16;
      std::memcpy(&xb0[e], &u, 4);
    }
    return std::make_tuple(x, x2, xb0);
  };
  reset();
  run_split();
  auto [ax, ax2, axb] = snap();
  reset();
  run_resident();
  auto [rx, rx2, rxb] = snap();
  auto err = [](const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0, m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, double(std::abs(a[i] - b[i]))), m = std::max(m, double(std::abs(a[i])));
    return std::make_pair(e, m);
  };
  const auto [e1, m1] = err(ax2, rx2);
  const auto [e2, m2] = err(ax, rx);
  const auto [e3, m3] = err(axb, rxb);
  (void)e1; (void)m1;
  std::printf("x err %.3g of %.3g, xb err %.3g of %.3g\n", e2, m2, e3, m3);
  const int n = 20;
  auto time = [&](auto&& f) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
  };
  std::printf("six dispatches %.3f ms, resident %.3f ms\n", time(run_split), time(run_resident));
  return (e2 > 1e-2 * m2 || e3 > 2e-2 * m3) ? 1 : 0;
}
