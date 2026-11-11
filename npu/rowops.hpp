// Row kernels on the XDNA NPU: elementwise and per-row operators between the
// projections (normalization, activations, gates, residuals, layout packing).
// Column c of the array handles rows c, c + 8, ... Each row is processed in
// `segments` pieces; operands of a piece live in the column's compute tile
// while the body runs. AIE2P has no scalar FPU and multiplies BF16 pairs into
// F32, so bodies use the compensated F32 vector helpers of kMathLibrary
// (@fmul, @rsqrt, @recip, @exp, @sigmoid) on vector<16xf32>.
#ifndef GUFO_MODELS_LFM3_NPU_ROWOPS_HPP_
#define GUFO_MODELS_LFM3_NPU_ROWOPS_HPP_

#include <cstdint>
#include <string>
#include <vector>

namespace gufo::models::lfm3::npu {

extern const char* const kMathLibrary;

struct RowOperand {
  std::string name;         // the body sees view %<name> of view<floats x f32>
  std::uint32_t binding{};  // kernel buffer argument
  std::uint64_t stride{};   // floats between rows; 0 broadcasts one row
  std::uint64_t offset{};   // floats before the first row's first piece
  std::uint32_t floats{};   // floats per piece (multiple of 16)
  // Element type in memory and in the tile (the body converts).
  bool bf16{};
  // kRows: piece p of row r at r * stride + offset + p * floats.
  // kBlocked: the projection layouts Xb/Yb, rows in blocks of eight, `groups`
  //   runs of eight elements per row: element (g, n) of row r at
  //   offset + (((r / 8) * groups + g) * 8 + r % 8) * 8 + n; a piece covers
  //   floats / 8 groups. The tile holds the row's elements densely.
  // kPartial: K-split outputs Yp, eight parts of a kBlocked layout per block:
  //   (((r / 8) * 8 + c) * groups + g) * 64 + (r % 8) * 8 + n; the tile is
  //   view<8 x floats> (part-major) and the body sums the parts.
  enum class Layout { kRows, kBlocked, kPartial } layout{Layout::kRows};
  std::uint32_t groups{};
};

struct RowKernel {
  std::string name;  // public kernel symbol, without '@'
  std::uint32_t bindings{};
  std::vector<RowOperand> inputs, outputs;
  // Loom statements for one piece. Available: %c0 %c1 %c16 (index),
  // %piece (index of the piece within the row), %row (absolute row).
  std::string body;
  std::uint32_t rows{8}, segments{1};
};

std::string RowKernelProgram(const RowKernel& kernel);

}  // namespace gufo::models::lfm3::npu

#endif
