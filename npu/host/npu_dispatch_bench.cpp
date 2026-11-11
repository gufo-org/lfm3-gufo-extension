// Dispatch latency on the NPU: one kernel repeated, and two kernels
// alternating (executable switch). Usage: npu_dispatch_bench <npu_gemv.loom>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include "npu_runtime.hpp"

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  std::stringstream text;
  text << in.rdbuf();
  const std::string source = text.str();
  lfm3::npu::Runtime npu;
  const int a = npu.Compile(source, argv[1], "@gemv8_k2048_m256");
  const int b = npu.Compile(source, argv[1], "@gemv8_stream_k2048_m8192");
  std::vector<std::uint16_t> x(2048), w(std::size_t(8192) * 2048);
  const int bx = npu.Upload(std::span<const std::uint16_t>(x));
  const int bw = npu.Upload(std::span<const std::uint16_t>(w));
  const int by = npu.Alloc(8192 * 4);
  const int bind[] = {bx, bw, by};
  auto time = [&](auto&& f, int n) {
    f();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) f();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
  };
  std::printf("small kernel repeated:   %8.1f us\n", time([&] { npu.Dispatch(a, bind); }, 200));
  std::printf("stream kernel repeated:  %8.1f us\n", time([&] { npu.Dispatch(b, bind); }, 50));
  std::printf("alternating small/stream:%8.1f us per pair\n", time([&] { npu.Dispatch(a, bind); npu.Dispatch(b, bind); }, 50));
  std::printf("small enqueued x50:       %8.1f us each\n", time([&] { for (int i = 0; i < 50; ++i) npu.Enqueue(a, bind); npu.Finish(); }, 4) / 50);
  std::vector<float> y(256);
  std::printf("read 1 KiB:              %8.1f us\n", time([&] { npu.Read(by, 0, y.data(), 1024); }, 200));
  std::printf("write 4 KiB:             %8.1f us\n", time([&] { npu.Write(bx, 0, x.data(), 4096); }, 200));
}
