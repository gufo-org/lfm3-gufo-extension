// Attention on the XDNA NPU: Q/K preparation (per-head RMSNorm, rotary from
// tables, sigmoid gate, K/V written into a session's F16 ring) and the
// attention itself (online softmax over 32-key chunks of the ring), writing
// the gated output directly as the next projection's blocked BF16 input.
//
// Per-row metadata (meta, 16 x i32 per row): [0] KV slot, [1] position,
// [2] first visible key, [3] last visible key, [4] query group (-1: state),
// with key groups in a separate i32 buffer when masking.
#ifndef GUFO_MODELS_LFM3_NPU_ATTENTION_HPP_
#define GUFO_MODELS_LFM3_NPU_ATTENTION_HPP_

#include <cstdint>
#include <string>

namespace gufo::models::lfm3::npu {

struct AttentionShape {
  std::uint32_t heads{}, kv_heads{}, dim{};  // dim 128 or 64 or 32
  std::uint32_t qkvg_groups{};  // groups of eight in a qkvg projection row
  bool gate{};                  // a per-head gate follows q, k, v
  std::uint32_t capacity{};     // ring rows per slot (power of two)
  std::uint32_t slots{};        // ring slots (power of two)
  std::uint32_t table_rows{};   // rotary table rows (power of two)
  std::uint32_t rows{};         // rows per dispatch
  bool groups{};                // d1 group masks
};

// prep(qkvg Yb, q_norm, k_norm, cos, sin, meta, q out, gate out, k pool, v pool)
std::string QkPrepProgram(const AttentionShape& s, const std::string& name);
// attend(q, gate, k pool, v pool, meta, key groups, out Xb)
std::string AttendProgram(const AttentionShape& s, const std::string& name);
// The same bindings with BFP16 matrix units, `rows_per_block` rows (1, or
// 4 consecutive rows of one session in order) per block of eight query
// vectors; no d1 group masks. Throws std::invalid_argument otherwise.
// Its queries must arrive multiplied by MmaQueryScale(dim) = log2(e) / sqrt(dim).
std::string AttendMmaProgram(const AttentionShape& s, const std::string& name, std::uint32_t rows_per_block);
double MmaQueryScale(std::uint32_t dim);

}  // namespace gufo::models::lfm3::npu

#endif
