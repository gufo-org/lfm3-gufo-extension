// Q8_0 GEMV through XRT: layout per gemv.py, check against a double
// reference, time synchronous and queued runs.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"
static std::uint16_t tobf(float f) { std::uint32_t b; std::memcpy(&b, &f, 4); return std::uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16); }
static float fbf(std::uint16_t h) { std::uint32_t b = std::uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }
int main(int argc, char** argv) {
  const int M = std::atoi(argv[3]), K = std::atoi(argv[4]), TM = argc > 5 ? std::atoi(argv[5]) : 8, iters = 50;
  const int COLS = 8, ROWS = 4, steps = M / (COLS * ROWS * TM), NB = K / 32;
  const std::size_t rec = std::size_t(TM) * K + std::size_t(TM) * NB * 2, xrec = K + NB * 2;
  std::vector<std::uint32_t> ins;
  { std::ifstream g(argv[2], std::ios::binary | std::ios::ate); auto sz = g.tellg(); g.seekg(0); ins.resize(sz / 4); g.read((char*)ins.data(), sz); }
  xrt::device dev(0);
  xrt::xclbin xb{std::string(argv[1])};
  std::string kname;
  for (auto& k : xb.get_kernels()) if (k.get_name().rfind("MLIR_AIE", 0) == 0) kname = k.get_name();
  dev.register_xclbin(xb);
  xrt::hw_context ctx(dev, xb.get_uuid());
  xrt::kernel kern(ctx, kname);
  auto bo_i = xrt::bo(dev, ins.size() * 4, XCL_BO_FLAGS_CACHEABLE, kern.group_id(1));
  auto bo_x = xrt::bo(dev, xrec, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(3));
  auto bo_w = xrt::bo(dev, rec * COLS * steps * ROWS, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(4));
  auto bo_y = xrt::bo(dev, std::size_t(M) * 4, XRT_BO_FLAGS_HOST_ONLY, kern.group_id(5));
  std::memcpy(bo_i.map<void*>(), ins.data(), ins.size() * 4);
  std::mt19937 rng(3);
  std::uniform_int_distribution<int> q(-127, 127);
  std::uniform_real_distribution<float> sc(0.001f, 0.02f);
  std::vector<std::int8_t> wq(std::size_t(M) * K), xq(K);
  std::vector<float> ws(std::size_t(M) * NB), xs(NB);
  for (auto& v : wq) v = q(rng);
  for (auto& v : xq) v = q(rng);
  for (auto& v : ws) v = fbf(tobf(sc(rng)));
  for (auto& v : xs) v = fbf(tobf(sc(rng) * 10));
  auto* X = bo_x.map<std::int8_t*>();
  std::memcpy(X, xq.data(), K);
  for (int b = 0; b < NB; ++b) { auto h = tobf(xs[b]); std::memcpy(X + K + b * 2, &h, 2); }
  auto* W = bo_w.map<std::int8_t*>();
  // Record of core (c, j) at step t holds rows ((t * COLS + c) * ROWS + j) * TM ...
  for (int c = 0; c < COLS; ++c)
    for (int t = 0; t < steps; ++t)
      for (int j = 0; j < ROWS; ++j) {
        std::int8_t* r = W + ((std::size_t(c) * steps + t) * ROWS + j) * rec;
        const int row0 = ((t * COLS + c) * ROWS + j) * TM;
        for (int i = 0; i < TM; ++i) {
          std::memcpy(r + std::size_t(i) * K, wq.data() + std::size_t(row0 + i) * K, K);
          for (int b = 0; b < NB; ++b) { auto h = tobf(ws[std::size_t(row0 + i) * NB + b]); std::memcpy(r + std::size_t(TM) * K + (std::size_t(i) * NB + b) * 2, &h, 2); }
        }
      }
  bo_i.sync(XCL_BO_SYNC_BO_TO_DEVICE); bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE); bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  auto run = [&] { auto r = kern(3, bo_i, unsigned(ins.size()), bo_x, bo_w, bo_y); r.wait(); };
  run();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) run();
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
  std::vector<xrt::run> runs;
  for (int i = 0; i < 8; ++i) { runs.emplace_back(kern); auto& r = runs.back(); r.set_arg(0, 3); r.set_arg(1, bo_i); r.set_arg(2, unsigned(ins.size())); r.set_arg(3, bo_x); r.set_arg(4, bo_w); r.set_arg(5, bo_y); }
  t0 = std::chrono::steady_clock::now();
  for (int rep = 0; rep < 8; ++rep) { for (auto& r : runs) r.start(); for (auto& r : runs) r.wait(); }
  const double qus = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 64;
  bo_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  const float* Y = bo_y.map<float*>();
  double maxrel = 0;
  for (int c = 0; c < COLS; ++c)
    for (int t = 0; t < steps; ++t)
      for (int j = 0; j < ROWS; ++j)
        for (int i = 0; i < TM; ++i) {
          const int row = ((t * COLS + c) * ROWS + j) * TM + i;
          double ref = 0;
          for (int b = 0; b < NB; ++b) {
            long s = 0;
            for (int e = 0; e < 32; ++e) s += long(wq[std::size_t(row) * K + b * 32 + e]) * xq[b * 32 + e];
            ref += double(s) * ws[std::size_t(row) * NB + b] * xs[b];
          }
          const double got = Y[((std::size_t(c) * steps + t) * ROWS + j) * TM + i];
          maxrel = std::max(maxrel, std::fabs(got - ref) / (std::fabs(ref) + 1e-3));
        }
  const double bytes = double(rec) * COLS * steps * ROWS;
  std::printf("gemv %dx%d: sync %.1f us (%.1f GB/s), queued %.1f us (%.1f GB/s), max rel err %.2e\n", M, K, us, bytes / us / 1e3, qus, bytes / qus / 1e3, maxrel);
}
