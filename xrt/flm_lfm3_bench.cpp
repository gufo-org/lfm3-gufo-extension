// Standalone check of the shim: prefill token ids from a file, greedy decode,
// print ids and timings. Usage: flm_lfm3_bench <model_dir> <asset_root> <ids.txt> [tokens]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "flm_lfm3_shim.h"
int main(int argc, char** argv) {
  std::ifstream f(argv[3]);
  std::vector<int32_t> ids;
  for (int v; f >> v;) ids.push_back(v);
  const int n = argc > 4 ? std::atoi(argv[4]) : 32;
  auto t0 = std::chrono::steady_clock::now();
  flm_lfm3* m = flm_lfm3_create(argv[1], argv[2], 8192);
  if (!m) { std::fprintf(stderr, "create: %s\n", flm_lfm3_last_error()); return 1; }
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  auto t1 = std::chrono::steady_clock::now();
  std::vector<float> lg(flm_lfm3_vocab(m));
  for (int rep = 0; rep < 2; ++rep) {
    flm_lfm3_clear(m);
    auto p0 = std::chrono::steady_clock::now();
    if (flm_lfm3_prefill(m, ids.data(), int(ids.size()), lg.data())) { std::fprintf(stderr, "prefill: %s\n", flm_lfm3_last_error()); return 1; }
    auto p1 = std::chrono::steady_clock::now();
    std::printf("load %.0f ms, prefill %zu tokens %.1f ms (%.0f tok/s)\nids:", ms(t0, t1), ids.size(), ms(p0, p1), ids.size() / ms(p0, p1) * 1e3);
    int tok = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
    auto d0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
      std::printf(" %d", tok);
      flm_lfm3_decode(m, tok, lg.data());
      tok = int(std::max_element(lg.begin(), lg.end()) - lg.begin());
    }
    auto d1 = std::chrono::steady_clock::now();
    std::printf("\ndecode %d tokens %.1f ms (%.1f tok/s)\n", n, ms(d0, d1), n / ms(d0, d1) * 1e3);
  }
  flm_lfm3_destroy(m);
}
