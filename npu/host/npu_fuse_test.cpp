// One fused FFN block (npu/fuse.cpp) against the same seven kernels
// dispatched one by one, at LFM3 1.7B widths: proj out, norm (op), gate/up,
// SwiGLU, down, reduce, norm (ffn). Outputs must match bit for bit.
// Usage: npu_fuse_test [rows]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

#include "../fuse.hpp"
#include "../glue.hpp"
#include "../program.hpp"
#include "../rowops.hpp"
#include "bfp.hpp"
#include "npu_runtime.hpp"

using namespace gufo::models::lfm3::npu;
namespace h = lfm3::npu;

int main(int argc, char** argv) {
  const std::uint32_t R = argc > 1 ? std::uint32_t(std::atoi(argv[1])) : 1, H = 2048, I = 7680, kMax = 256;
  const std::uint32_t R8 = std::max<std::uint32_t>(R, 8), blocks = (R + 7) / 8;
  h::Runtime npu;
  std::mt19937 rng(5);
  auto vec = [&](std::size_t n, float lo, float hi) {
    std::vector<float> v(n);
    std::uniform_real_distribution<float> d(lo, hi);
    for (auto& x : v) x = d(rng);
    return v;
  };
  auto dyn = [&](RowKernel k) {
    k.rows = kMax;
    k.dynamic_rows = true;
    k.name += "_dyn";
    return k;
  };
  struct Proj { ProjectionShape s; int w; };
  auto proj = [&](std::uint32_t k, std::uint32_t m) {
    Proj p{PlanProjection(k, m, blocks), -1};
    const auto split = p.s.split == Split::kRows ? h::Split::kRows : h::Split::kColumns;
    auto bytes = h::PackWeights(m, p.s.k, p.s.m, split, [&](std::uint32_t r, std::span<float> out) {
      std::mt19937 g(r * 7919u + k);
      std::uniform_real_distribution<float> d(-0.03f, 0.03f);
      for (auto& v : out) v = d(g);
    });
    p.w = npu.Upload(std::span<const std::uint8_t>(bytes));
    return p;
  };
  const Proj out = proj(H, H), gateup = proj(H, 2 * I), down = proj(I, H);
  const RowKernel n_op = dyn(NormPackKernel(H, R8, Residual::kBlocked, true, false, 1e-5f));
  const RowKernel swiglu = dyn(SwiGluPackKernel(I, R8));
  const RowKernel reduce = dyn(ReducePartialKernel(H, R8));
  const RowKernel n_ffn = dyn(NormPackKernel(H, R8, Residual::kRows, true, false, 1e-5f));

  // Buffers, sized for kMax rows.
  const std::size_t big = std::size_t(kMax) * 2 * I * 8 + 4096;
  enum { kXb, kWout, kYout, kX, kWop, kX2, kWgu, kYproj, kXffn, kWd, kYdown, kO, kWffn, kCount, kN };
  int b[kN];
  b[kWout] = out.w, b[kWgu] = gateup.w, b[kWd] = down.w;
  auto xb0 = vec(std::size_t(kMax) * H, -1, 1);
  std::vector<std::uint16_t> xbh(xb0.size());
  for (std::size_t i = 0; i < xb0.size(); ++i) xbh[i] = h::ToBF16(xb0[i]);
  auto x0 = vec(std::size_t(kMax) * H, -2, 2), wop = vec(2 * H, 0.5f, 1.5f), wffn = vec(2 * H, 0.5f, 1.5f);
  const std::int32_t count[16] = {std::int32_t(R)};
  auto reset = [&] {
    npu.Write(b[kXb], 0, xbh.data(), xbh.size() * 2);
    npu.Write(b[kX], 0, x0.data(), x0.size() * 4);
  };
  for (int i : {kXb, kYout, kX, kX2, kYproj, kXffn, kYdown, kO}) b[i] = npu.Alloc(big);
  b[kWop] = npu.Upload(std::span<const float>(wop));
  b[kWffn] = npu.Upload(std::span<const float>(wffn));
  b[kCount] = npu.Upload(std::span<const std::int32_t>(count, 16));

  struct Stage { std::string program, root; std::vector<int> args; };
  std::vector<Stage> stages = {
      {ProjectionProgram(out.s), ProjectionRoot(out.s), {kXb, kWout, kYout}},
      {RowKernelProgram(n_op), "@" + n_op.name, {kX, kYout, kWop, kX2, kXb, kCount}},
      {ProjectionProgram(gateup.s), ProjectionRoot(gateup.s), {kXb, kWgu, kYproj}},
      {RowKernelProgram(swiglu), "@" + swiglu.name, {kYproj, kXffn, kCount}},
      {ProjectionProgram(down.s), ProjectionRoot(down.s), {kXffn, kWd, kYdown}},
      {RowKernelProgram(reduce), "@" + reduce.name, {kYdown, kO, kCount}},
      {RowKernelProgram(n_ffn), "@" + n_ffn.name, {kX2, kO, kWffn, kX, kXb, kCount}},
  };
  auto now = [] { return std::chrono::steady_clock::now(); };
  auto ms = [](auto a, auto z) { return std::chrono::duration<double, std::milli>(z - a).count(); };
  if (argc > 2) {  // comma-separated stage indices
    std::vector<Stage> pick;
    for (const char* p = argv[2]; *p;) {
      pick.push_back(stages[std::size_t(std::strtol(p, const_cast<char**>(&p), 10))]);
      if (*p == ',') ++p;
    }
    stages = pick;
  }
  std::vector<int> ids;
  for (const auto& s : stages) ids.push_back(npu.Compile(s.program, s.root.substr(1) + ".loom", s.root));
  std::vector<FuseStage> fs;
  const std::size_t fuse_n = std::getenv("FUSE_N") ? std::size_t(std::atoi(std::getenv("FUSE_N"))) : stages.size();
  for (std::size_t i = 0; i < std::min(fuse_n, stages.size()); ++i) fs.push_back({stages[i].program, stages[i].root, stages[i].args});
  auto t0 = now();
  const std::string fused_text = FuseProgram(fs, kN, "ffn_fused");
  if (const char* dump = std::getenv("FUSE_DUMP")) {
    FILE* f = std::fopen(dump, "w");
    std::fputs(fused_text.c_str(), f);
    std::fclose(f);
  }
  int fused = -1;
  try {
    fused = npu.Compile(fused_text, "ffn_fused.loom", "@ffn_fused");
  } catch (const std::exception& e) {
    std::printf("fused compile failed (timing stages only): %.4000s\n", e.what());
  }
  std::printf("fused compile %.1f s\n", ms(t0, now()) / 1000);

  auto run_split = [&] {
    for (std::size_t i = 0; i < stages.size(); ++i) {
      std::vector<int> bind;
      for (int a : stages[i].args) bind.push_back(b[a]);
      npu.Enqueue(ids[i], bind);
    }
    npu.Finish();
  };
  auto run_fused = [&] {
    npu.Enqueue(fused, std::span<const int>(b, kN));
    npu.Finish();
  };
  auto snapshot = [&] {
    std::vector<float> x(std::size_t(R) * H);
    std::vector<std::uint16_t> xb(std::size_t(blocks) * 8 * H);
    npu.Read(b[kX], 0, x.data(), x.size() * 4);
    npu.Read(b[kXb], 0, xb.data(), xb.size() * 2);
    return std::make_pair(x, xb);
  };
  const int n = 20;
  auto stage_times = [&] {
    for (std::size_t i = 0; i < stages.size(); ++i) {
      std::vector<int> bind;
      for (int a : stages[i].args) bind.push_back(b[a]);
      npu.Dispatch(ids[i], bind);
      const auto ts = now();
      for (int k = 0; k < n; ++k) npu.Enqueue(ids[i], bind);
      npu.Finish();
      std::printf("  %-28s %.3f ms\n", stages[i].root.c_str(), ms(ts, now()) / n);
    }
  };
  if (fused < 0) {
    stage_times();
    return 0;
  }
  reset();
  run_split();
  const auto a = snapshot();
  reset();
  run_fused();
  const auto f = snapshot();
  std::size_t dx = 0, dxb = 0;
  for (std::size_t i = 0; i < a.first.size(); ++i) dx += std::memcmp(&a.first[i], &f.first[i], 4) != 0;
  for (std::size_t i = 0; i < a.second.size(); ++i) dxb += a.second[i] != f.second[i];
  std::printf("rows %u: x differs at %zu of %zu, xb at %zu of %zu (x[0] %g vs %g)\n", R, dx, a.first.size(), dxb,
              a.second.size(), a.first[0], f.first[0]);
  t0 = now();
  for (int i = 0; i < n; ++i) run_split();
  const double split_ms = ms(t0, now()) / n;
  t0 = now();
  for (int i = 0; i < n; ++i) run_fused();
  const double fused_ms = ms(t0, now()) / n;
  std::printf("seven dispatches %.3f ms, fused %.3f ms\n", split_ms, fused_ms);
  for (std::size_t i = 0; i < stages.size(); ++i) {
    std::vector<int> bind;
    for (int a : stages[i].args) bind.push_back(b[a]);
    npu.Dispatch(ids[i], bind);
    const auto ts = now();
    for (int k = 0; k < n; ++k) npu.Enqueue(ids[i], bind);
    npu.Finish();
    std::printf("  %-28s %.3f ms\n", stages[i].root.c_str(), ms(ts, now()) / n);
  }
  return dx || dxb;
}
