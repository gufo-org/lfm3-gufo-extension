// LFM3 backbone on the XDNA NPU. Projections run as Loom programs over
// pre-encoded bfp16ebs8 weights (npu/gen_proj.py); normalization, the gated
// convolution, rotary embedding and attention run on the host in FP32, the
// same arithmetic as reference.cpp. Rows of up to eight sessions or prompt
// positions share every weight pass.
#ifndef GUFO_MODELS_LFM3_NPU_BACKBONE_HPP_
#define GUFO_MODELS_LFM3_NPU_BACKBONE_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../weights.hpp"

namespace gufo::models::lfm3::npu {

// Recurrent state of one sequence, laid out like the HIP engine's: FP32
// convolution history [channel][width - 1] and F16 K/V rings
// [capacity][kv heads][head dim].
struct State {
  struct Layer {
    std::vector<float> conv;
    std::vector<_Float16> k, v;
    std::uint32_t capacity{};
  };
  std::vector<Layer> layers;
  std::uint32_t position{};
  void Reset();
};

// One row of a forward pass.
struct Row {
  State* state{};
  // Absolute position; rows of one state must be consecutive and ascending.
  std::uint32_t position{};
  bool logits{};
};

class Backbone {
 public:
  ~Backbone();
  // Encodes every projection and uploads it to the NPU. |weights| must point
  // at host data (the GGUF mapping) and outlive the backbone.
  static std::unique_ptr<Backbone> Load(const Weights& weights, std::string* error);
  State NewState(std::uint32_t context) const;
  void Embed(std::uint32_t token, std::span<float> row) const;
  // Advances rows of input embeddings (hidden floats each). |hidden| receives
  // each row's final-normed hidden state; |logits| receives vocab floats for
  // each row with Row::logits set, in row order.
  // decision: d1 prompt in one pass; full attention is bidirectional, with
  // groups (one per row; -1 state, else a question ID) masking questions
  // from each other. Convolution and sliding attention remain causal.
  void Forward(std::span<const float> inputs, std::span<const Row> rows,
               std::span<float> hidden, std::span<float> logits, bool decision = false,
               std::span<const std::int32_t> groups = {});
  // Vocabulary logits of final-normed hidden rows (n <= kRows).
  void Head(std::span<const float> hidden, std::uint32_t n, std::span<float> logits);
  const Config& config() const;

  // Rows per pass: they share each weight read.
  static constexpr std::uint32_t kRows = 64;

 private:
  Backbone();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::lfm3::npu

#endif
