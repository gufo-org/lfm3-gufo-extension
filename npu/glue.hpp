// Kernels between the LFM3 projections on the NPU (device-resident forward):
// RMSNorm with residuals and packing into the projections' blocked BF16
// input, partial-sum reduction, SwiGLU and the gated short convolution.
#ifndef GUFO_MODELS_LFM3_NPU_GLUE_HPP_
#define GUFO_MODELS_LFM3_NPU_GLUE_HPP_

#include <cstdint>
#include <string>

#include "rowops.hpp"

namespace gufo::models::lfm3::npu {

// y source of a residual step: a projection's blocked output (Yb), or rows.
enum class Residual { kNone, kBlocked, kRows };

// x' = x + post_norm(y) (post norm optional), xb = pack(norm(x')); with
// Residual::kNone, xb = pack(norm(x)). Bindings: x, [y], [post], norm, x', xb[, z]
// where z (optional) receives norm(x') in F32 rows.
RowKernel NormPackKernel(std::uint32_t hidden, std::uint32_t rows, Residual residual, bool post_norm,
                         bool rows_out, float eps);
// Sum of a K-split output's eight parts: Yp (groups = m / 8) -> rows.
RowKernel ReducePartialKernel(std::uint32_t m, std::uint32_t rows);
// Copies blocked rows (ProjectionProgram outputs, one part) to row-major F32.
RowKernel UnblockKernel(std::uint32_t m, std::uint32_t rows);
// SwiGLU of a fused gate/up Yb (2 I) into the blocked BF16 input (I).
RowKernel SwiGluPackKernel(std::uint32_t intermediate, std::uint32_t rows, bool partial = false);
// Gated short convolution step for `rows` rows in order: proj Yb (3 H),
// history pool [slot][width - 1][H] (tap-major), weights [width][H],
// meta (slot at element 16 r), out blocked BF16 (H).
std::string ConvProgram(std::uint32_t hidden, std::uint32_t width, std::uint32_t rows, std::uint32_t slots,
                        const std::string& name, bool partial = false);

}  // namespace gufo::models::lfm3::npu

#endif
