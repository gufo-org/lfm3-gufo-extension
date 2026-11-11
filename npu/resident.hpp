// Resident NPU programs: several dependent stages in one invocation, with
// cross-column barriers between them instead of separate dispatches (each
// dispatch re-establishes the whole array, about 90-155 us on HRX).
#ifndef GUFO_MODELS_LFM3_NPU_RESIDENT_HPP_
#define GUFO_MODELS_LFM3_NPU_RESIDENT_HPP_

#include <cstdint>
#include <string>

namespace gufo::models::lfm3::npu {

// The decode (one row) feed-forward block of an LFM3 layer:
//   x2 = x + post_norm(y_out) * w_op[0:H]     (out-projection residual)
//   xb = pack(norm(x2) * w_op[H:2H])
//   g|u = W_gu xb       (K split: eight partial sums per element)
//   xb_ffn = pack(silu(g) * u)
//   d = W_d xb_ffn      (K split)
//   x = x2 + post_norm(sum d) * w_ffn[0:H];  xb = pack(norm(x) * w_ffn[H:2H])
// Bindings: x, y_out (Yb), w_op, x2, xb, w_gu, y_gu (Yp), xb_ffn, w_d,
// y_d (Yp), w_ffn. Layouts as npu/program.hpp; weights packed for the K split.
std::string ResidentFfnProgram(std::uint32_t hidden, std::uint32_t intermediate, float eps,
                               const std::string& name);

}  // namespace gufo::models::lfm3::npu

#endif
