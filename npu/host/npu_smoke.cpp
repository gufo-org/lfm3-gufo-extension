// Compiles the eight-column BF16 GEMV, runs it on the NPU and compares with a
// CPU reference. Usage: npu_smoke <npu_gemv.loom>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

#include "npu_runtime.hpp"

static std::uint16_t ToBF16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
static float FromBF16(std::uint16_t h) {
  std::uint32_t u = std::uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  std::stringstream text;
  text << in.rdbuf();
  const std::string source = text.str();
  lfm3::npu::Runtime npu;
  const int kernel = npu.Compile(source, argv[1], "@gemv8_k2048_m256");
  const int K = 2048, M = 256;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<std::uint16_t> x(K), w(std::size_t(M) * K);
  for (auto& v : x) v = ToBF16(u(rng));
  for (auto& v : w) v = ToBF16(u(rng));
  const int bx = npu.Upload(std::span<const std::uint16_t>(x));
  const int bw = npu.Upload(std::span<const std::uint16_t>(w));
  const int by = npu.Alloc(M * 4);
  const int bindings[] = {bx, bw, by};
  npu.Dispatch(kernel, bindings);
  std::vector<float> y(M);
  npu.Read(by, 0, y.data(), M * 4);
  double worst = 0;
  for (int m = 0; m < M; ++m) {
    double ref = 0;
    for (int k = 0; k < K; ++k) ref += double(FromBF16(x[k])) * FromBF16(w[std::size_t(m) * K + k]);
    worst = std::max(worst, std::abs(ref - y[m]));
  }
  std::printf("max abs error %.6f (y[0] %.4f)\n", worst, y[0]);
  return worst < 0.05 ? 0 : 1;
}
