// gufo_xrt_gemm against a double-precision reference on random Q8_0 matrices,
// with timings. Usage: gemm_check <dir with q8gemm.xclbin> [n k rows]...
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../../gufo_xrt_gemm.h"

int main(int argc, char** argv) {
  if (argc < 2)
    return 2;
  gufo_xrt_gemm* g = gufo_xrt_gemm_create(argv[1]);
  if (!g)
    return std::fprintf(stderr, "%s\n", gufo_xrt_gemm_last_error()), 1;
  std::vector<int> shapes;
  for (int i = 2; i < argc; ++i)
    shapes.push_back(std::atoi(argv[i]));
  if (shapes.empty())
    shapes = {1312, 1024, 12, 1024,  1024, 12, 5632, 1024, 12, 1024, 2816, 12,
              2017, 1024, 1,  12288, 2048, 1,  1792, 768,  50, 768,  1792, 50};
  std::mt19937 rng(1);
  std::normal_distribution<float> nd;
  int bad = 0;
  for (size_t s = 0; s + 2 < shapes.size(); s += 3) {
    const int n = shapes[s], k = shapes[s + 1], rows = shapes[s + 2];
    std::vector<int8_t> codes(size_t(n) * k);
    std::vector<float> scales(size_t(n) * k / 32), x(size_t(rows) * k),
        y(size_t(rows) * n);
    for (auto& c : codes)
      c = int8_t(std::lround(std::clamp(nd(rng) * 40.0f, -127.0f, 127.0f)));
    for (auto& d : scales)
      d = float(_Float16(0.001f * (1.0f + std::fabs(nd(rng)))));
    for (auto& v : x)
      v = nd(rng);
    const int id = gufo_xrt_gemm_upload(g, n, k, codes.data(), scales.data());
    if (id < 0)
      return std::fprintf(stderr, "%s\n", gufo_xrt_gemm_last_error()), 1;
    double worst = 0, top = 0;
    for (int it = 0; it < 6; ++it) {
      const auto t0 = std::chrono::steady_clock::now();
      if (gufo_xrt_gemm_run(g, id, x.data(), rows, y.data()))
        return std::fprintf(stderr, "%s\n", gufo_xrt_gemm_last_error()), 1;
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
      if (it == 5)
        std::printf("n %5d k %5d rows %3d: %.3f ms", n, k, rows, ms);
    }
    for (int r = 0; r < rows; ++r)
      for (int j = 0; j < n; ++j) {
        double ref = 0;
        for (int c = 0; c < k; ++c)
          ref += double(x[size_t(r) * k + c]) * codes[size_t(j) * k + c] *
                 scales[size_t(j) * (k / 32) + c / 32];
        worst = std::max(worst, std::fabs(ref - y[size_t(r) * n + j]));
        top = std::max(top, std::fabs(ref));
      }
    std::printf(", max error %.2e of %.2e (%.1e)\n", worst, top, worst / top);
    bad += worst / top > 2e-4;
  }
  gufo_xrt_gemm_destroy(g);
  return bad;
}
