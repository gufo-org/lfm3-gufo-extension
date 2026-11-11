// Dense projections on the XDNA NPU shared by the LFM3 backbone and towers:
// one process-wide NPU context, weights encoded once, kernels generated and
// compiled per shape on first use (npu/program.hpp).
#ifndef GUFO_MODELS_LFM3_NPU_DENSE_HPP_
#define GUFO_MODELS_LFM3_NPU_DENSE_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <span>

#include "../weights.hpp"
#include "program.hpp"

namespace lfm3::npu {
class Runtime;
}

namespace gufo::models::lfm3::npu {

class Dense {
 public:
  // A weight matrix y = W x, W rows x cols, resident on the NPU.
  struct Matrix {
    int weights{-1};
    std::uint32_t k{}, m{};
    ProjectionShape shape;  // padded, one block
  };
  using RowFn = std::function<void(std::uint32_t row, std::span<float> out)>;

  // The process-wide engine; throws when the NPU or liblfm3_npu.so is missing.
  static Dense& Shared();
  // Encodes rows x cols weights given row by row (cols floats each).
  // `k_split` forces the K split (eight partial sums) at any K.
  Matrix Upload(std::uint32_t rows, std::uint32_t cols, const RowFn& row, bool k_split = false);
  // Stacks tensors' rows (same column count) into one matrix.
  Matrix Upload(std::span<const Tensor* const> tensors, bool k_split = false);
  // y[r][m] = W x[r] for `rows` rows of x (k floats each); y is rows x m.
  void Run(const Matrix& w, std::span<const float> x, std::uint32_t rows, std::span<float> y);

  // Device-resident use: the runtime, and y = W x for `blocks` blocks of
  // eight rows already in the projection layouts (xb blocked BF16, y Yb/Yp).
  ::lfm3::npu::Runtime& runtime();
  void Enqueue(const Matrix& w, int xb, int y, std::uint32_t blocks);
  // Allocates a zeroed device buffer.
  int Zeros(std::uint64_t bytes);

  ~Dense();

 private:
  Dense();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::lfm3::npu

#endif
