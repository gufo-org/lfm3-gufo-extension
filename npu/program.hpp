// Loom source of the NPU projection kernels, generated per shape at run time
// (compiling one takes about 0.1 s). Layouts are those of npu/host/bfp.hpp.
//
// A projection y_b = W x_b covers `blocks` blocks of eight rows b.
//   x   blocked BF16 Xb[block][j][b][c] = x_{8 block + b}[8j + c]
//   W   pre-encoded bfp16ebs8 8 x 8 tiles (72 bytes each)
//   Row split (K <= 2048): column c owns groups [c G, (c + 1) G) of eight
//     output rows; record 2g + h holds half h of K for group g.
//     y Yb[block][g][b][n] = y_{8 block + b}[8g + n] (F32)
//   K split: column c owns K / 8 of every group's tiles, one record per
//     (column, group); y Yp[block][c][g][b][n], summed over c by the host.
#ifndef GUFO_MODELS_LFM3_NPU_PROGRAM_HPP_
#define GUFO_MODELS_LFM3_NPU_PROGRAM_HPP_

#include <cstdint>
#include <string>

namespace gufo::models::lfm3::npu {

enum class Split { kRows, kColumns };

struct ProjectionShape {
  std::uint32_t k{}, m{}, blocks{1};
  Split split{Split::kRows};
  bool operator==(const ProjectionShape&) const = default;
};

// Pads K and M to what a kernel accepts and picks the split.
ProjectionShape PlanProjection(std::uint32_t k, std::uint32_t m, std::uint32_t blocks);
// Public kernel name and its Loom module.
std::string ProjectionRoot(const ProjectionShape& shape);
std::string ProjectionProgram(const ProjectionShape& shape);
// K-split shapes over several blocks: each weight record is applied to
// KSplitBlocksPerPass blocks held at once. Root is ProjectionRoot + "_kmulti".
std::uint32_t KSplitBlocksPerPass(const ProjectionShape& shape);
std::string ProjectionKMultiProgram(const ProjectionShape& shape);

constexpr std::uint32_t kColumns = 8;

}  // namespace gufo::models::lfm3::npu

#endif
