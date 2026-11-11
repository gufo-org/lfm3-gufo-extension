// Cross-column barrier latency inside one invocation (npu/probe/barrier.loom).
// Usage: npu_barrier_bench npu/probe/barrier.loom
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "npu_runtime.hpp"
int main(int argc, char** argv) {
  lfm3::npu::Runtime npu;
  std::ifstream in(argv[1]);
  std::stringstream ss;
  ss << in.rdbuf();
  std::string src = ss.str();
  std::vector<std::int32_t> out(8 * 16, -1);
  const int b = npu.Upload(std::span<const std::int32_t>(out));
  for (const char* r : {"1", "10000"}) {
    std::string t = src;
    t.replace(t.find("@ROUNDS"), 7, r);
    const int k = npu.Compile(t, "barrier.loom", "@barrier");
    const int bind[] = {b};
    npu.Dispatch(k, bind);
    const auto t0 = std::chrono::steady_clock::now();
    npu.Dispatch(k, bind);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    npu.Read(b, 0, out.data(), out.size() * 4);
    std::printf("rounds %s: %.1f us; counts", r, us);
    for (int c = 0; c < 8; ++c) std::printf(" %d", out[c * 16]);
    std::printf("\n");
  }
}
