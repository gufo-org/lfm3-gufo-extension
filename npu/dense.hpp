#pragma once
// Matrix products of the LFM3-Audio towers on the NPU: Q8_0 weights through
// gufo::npu::Q8Gemm, one process-wide program loaded from the NPU bundle's
// gemm/ directory (npu/iron/build.sh).

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>

#include "../weights.hpp"

namespace gufo::models::lfm3::npu {

class Dense {
public:
  // A weight matrix y = W x, W m rows x k columns, resident on the NPU.
  struct Matrix {
    int weights{-1};
    std::uint32_t k{}, m{};
    // 2: non-Q8_0 weights as two Q8_0 planes (codes, then the residual), K
    // doubled and x repeated, about 2^-15 of each block's largest weight.
    std::uint32_t planes{1};
  };
  using RowFn = std::function<void(std::uint32_t row, std::span<float> out)>;
  // Opens the program in `bundle`/gemm once; later calls keep the first.
  static void Open(const std::filesystem::path& bundle);
  // The process-wide engine; throws before Open.
  static Dense& Shared();
  // rows x cols weights given row by row (cols floats each), as two Q8_0
  // planes.
  Matrix Upload(std::uint32_t rows, std::uint32_t cols, const RowFn& row);
  // Stacks tensors' rows (same column count). Q8_0 tensors keep their codes.
  Matrix Upload(std::span<const Tensor* const> tensors);
  // y[r][m] = W x[r] for `rows` rows of x (k floats each).
  void Run(const Matrix& w, std::span<const float> x, std::uint32_t rows,
           std::span<float> y);
  ~Dense();

private:
  Dense();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::lfm3::npu
