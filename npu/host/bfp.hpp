// Host packing for the NPU projections (layouts in npu/gen_proj.py).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <thread>
#include <vector>

namespace lfm3::npu {

inline std::uint16_t ToBF16(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7FFFFFFF) > 0x7F800000) return 0x7FC0;
  return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}
inline float FromBF16(std::uint16_t h) {
  std::uint32_t u = std::uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// bfp16ebs8 row of eight values: one exponent byte e and eight signed
// mantissas m, value = m * 2^(e - 133).
inline void EncodeRow8(const float* v, std::uint8_t* out) {
  float peak = 0;
  for (int i = 0; i < 8; ++i) peak = std::max(peak, std::abs(v[i]));
  int e = 0;
  if (peak > 0) {
    int exp2;
    std::frexp(peak / 127.0f, &exp2);  // peak / 127 < 2^exp2
    e = std::clamp(exp2 + 133, 0, 255);
  }
  out[0] = static_cast<std::uint8_t>(e);
  const float scale = std::ldexp(1.0f, 133 - e);
  for (int i = 0; i < 8; ++i) {
    const long q = std::lround(v[i] * scale);
    out[1 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(std::clamp(q, -127L, 127L)));
  }
}
inline float DecodeBfp(const std::uint8_t* row, int i) {
  return std::ldexp(static_cast<float>(static_cast<std::int8_t>(row[1 + i])), int(row[0]) - 133);
}

// Calls fn(begin, end) over [0, n) on all hardware threads.
inline void Parallel(std::size_t n, const std::function<void(std::size_t, std::size_t)>& fn) {
  const std::size_t threads = std::max(1u, std::thread::hardware_concurrency());
  const std::size_t step = (n + threads - 1) / threads;
  std::vector<std::thread> pool;
  for (std::size_t b = 0; b < n; b += step) pool.emplace_back(fn, b, std::min(n, b + step));
  for (auto& t : pool) t.join();
}

// Packs rows x cols weights; row(r, out) writes cols floats of row r.
// Row split: two records per group of eight rows (cols / 16 tiles each).
// K split: record (column, group) of cols / 64 tiles.
enum class Split { kRows, kColumns };
inline std::vector<std::uint8_t> PackWeights(
    std::uint32_t rows, std::uint32_t cols, std::uint32_t padded_rows, Split split,
    const std::function<void(std::uint32_t, std::span<float>)>& row) {
  const std::uint32_t tiles = cols / 8, groups = padded_rows / 8;
  std::vector<std::uint8_t> out(std::size_t(groups) * tiles * 72);
  Parallel(groups, [&](std::size_t g0, std::size_t g1) {
    std::vector<float> block(std::size_t(8) * cols);
    for (std::size_t g = g0; g < g1; ++g) {
      for (std::uint32_t n = 0; n < 8; ++n) {
        const std::uint32_t r = std::uint32_t(g) * 8 + n;
        std::span<float> dst(block.data() + std::size_t(n) * cols, cols);
        if (r < rows) row(r, dst);
        else std::fill(dst.begin(), dst.end(), 0.0f);
      }
      for (std::uint32_t j = 0; j < tiles; ++j) {
        std::size_t tile_index;
        if (split == Split::kRows) {
          const std::uint32_t half = tiles / 2, h = j / half, jl = j % half;
          tile_index = (g * 2 + h) * half + jl;
        } else {
          const std::uint32_t t = tiles / 8, c = j / t, jl = j % t;
          tile_index = (std::size_t(c) * groups + g) * t + jl;
        }
        std::uint8_t* tile = out.data() + tile_index * 72;
        for (std::uint32_t n = 0; n < 8; ++n)
          EncodeRow8(block.data() + std::size_t(n) * cols + j * 8, tile + n * 9);
      }
    }
  });
  return out;
}

// x for up to eight sequences, x[b][k], to blocked BF16 Xb[j][b][c].
inline void PackActivations(std::span<const float> x, std::uint32_t batch, std::uint32_t cols,
                            std::span<std::uint16_t> out) {
  std::fill(out.begin(), out.end(), 0);
  for (std::uint32_t b = 0; b < batch; ++b)
    for (std::uint32_t k = 0; k < cols; ++k)
      out[(std::size_t(k / 8) * 8 + b) * 8 + k % 8] = ToBF16(x[std::size_t(b) * cols + k]);
}

// Blocked F32 Yb[g][b][n] to y[b][m] for the first `rows` rows.
inline void UnpackOutputs(std::span<const float> blocked, std::uint32_t batch, std::uint32_t rows,
                          std::span<float> y, std::uint32_t parts = 1, std::uint32_t padded_rows = 0) {
  if (!padded_rows) padded_rows = rows;
  const std::size_t part = std::size_t(padded_rows) * 8;
  for (std::uint32_t b = 0; b < batch; ++b)
    for (std::uint32_t m = 0; m < rows; ++m) {
      float sum = 0;
      for (std::uint32_t p = 0; p < parts; ++p)
        sum += blocked[p * part + (std::size_t(m / 8) * 8 + b) * 8 + m % 8];
      y[std::size_t(b) * rows + m] = sum;
    }
}

}  // namespace lfm3::npu
