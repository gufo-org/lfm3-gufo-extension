// Times AttendMmaProgram at LFM3 1.7B widths: one session's `rows` rows
// ending at position `keys` (prefill), or `rows` sessions of one row
// (decode). Usage: npu_attention_bench rows keys rows_per_block [scalar]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "../attention.hpp"
#include "npu_runtime.hpp"

using namespace gufo::models::lfm3::npu;

int main(int argc, char** argv) {
  const std::uint32_t rows = std::atoi(argv[1]), keys = std::atoi(argv[2]), rpb = std::atoi(argv[3]);
  const bool scalar = argc > 4;
  lfm3::npu::Runtime npu;
  AttentionShape s{16, 4, 128, 392, true, 4096, 8, 64, 256, false, !scalar};
  const std::uint32_t D = s.dim, Q = s.heads * D, KV = s.kv_heads * D;
  const std::string program = scalar ? AttendProgram(s, "attend") : AttendMmaProgram(s, "attend", rpb);
  if (const char* dump = std::getenv("DUMP")) {  // write the program instead of running it
    FILE* f = std::fopen(dump, "w");
    std::fputs(program.c_str(), f);
    std::fclose(f);
    return 0;
  }
  const int k = npu.Compile(program, "attend.loom", "@attend");
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<_Float16> pool(std::size_t(s.slots) * s.capacity * KV);
  for (auto& v : pool) v = _Float16(u(rng));
  std::vector<float> q(std::size_t(256) * Q), g(256 * 16);
  for (auto& v : q) v = u(rng) * 0.1f;
  std::vector<std::int32_t> meta(256 * 16, 0), groups(4096, 0);
  for (std::uint32_t r = 0; r < rows; ++r) {
    const bool decode = rpb == 1 && rows > 1 && !scalar && std::getenv("DECODE");
    meta[r * 16 + 0] = decode ? int(r % s.slots) : 0;
    meta[r * 16 + 2] = 0;
    meta[r * 16 + 3] = decode ? int(keys) : int(keys - rows + r);
    meta[r * 16 + 4] = -1;
  }
  const int bq = npu.Upload(std::span<const float>(q)), bg = npu.Upload(std::span<const float>(g)),
            bk = npu.Upload(std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(pool.data()), pool.size())),
            bv = npu.Upload(std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(pool.data()), pool.size())),
            bm = npu.Upload(std::span<const std::int32_t>(meta)), bgr = npu.Upload(std::span<const std::int32_t>(groups)),
            bo = npu.Alloc((std::size_t(32) * Q / 8 * 64 + 64) * 2);
  const std::int32_t cn[16] = {std::int32_t(rows)};
  const int bc = npu.Upload(std::span<const std::int32_t>(cn, 16));
  const int bind[] = {bq, bg, bk, bv, bm, bgr, bo, bc};
  npu.Dispatch(k, bind);
  const int n = 5;
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) npu.Dispatch(k, bind);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / n;
  std::printf("%s rows %u keys %u rpb %u: %.3f ms\n", scalar ? "scalar" : "mma", rows, keys, rpb, ms);
}
