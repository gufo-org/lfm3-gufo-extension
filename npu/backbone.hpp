#pragma once
// The LFM3 backbone on the XDNA NPU from the GGUF alone: every matrix product
// runs on Gufo's installed NPU programs (npu::Dense), the row operations
// (norms, short convolutions, rotary embeddings, attention) on the host.
//
// - Few rows (decode, short continuations) go through Q8Gemm with the GGUF's
//   Q8_0 codes: one dispatch per projection (the attention block's q/k/v/gate
//   and the FFN's gate/up are one stacked matrix each).
// - Many rows (prefill) go through BfpGemm on bfp16 copies of the same
//   matrices, packed at load time.
// - The head (tied to the embedding table) runs on Q8Gemm for the rows that
//   need logits.
//
// The state is one sequence: convolution histories and F32 K/V for every
// position held.

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "../weights.hpp"
#include "dense.hpp"

namespace gufo::models::lfm3::npu {

class Backbone {
public:
  // Uploads the weights (borrowed; they must outlive the backbone). Throws
  // gufo::npu::Error or std::runtime_error.
  Backbone(const Weights& weights, int max_length);
  ~Backbone();
  Backbone(const Backbone&) = delete;
  Backbone& operator=(const Backbone&) = delete;

  const Config& config() const { return w_.config; }
  int max_length() const { return max_length_; }
  int length() const { return length_; }

  // Token embedding row (hidden floats).
  void Embed(std::int32_t id, float* out) const;
  // Runs n rows of x (hidden floats each, at positions length() ...) through
  // the layers in place: x leaves as the residual stream. With `groups` (n
  // entries; d1 prompts from an empty state), full attention is
  // bidirectional: a row of group -1 sees and is seen by every row, other
  // rows see the -1 rows and their own group. Convolutions and sliding
  // attention stay causal.
  void Forward(float* x, int n, std::span<const std::int32_t> groups = {});
  // Final norm of `rows` residual rows into out.
  void FinalNorm(const float* x, int rows, float* out) const;
  // Logits [rows][vocab] of final-normed rows.
  void Head(const float* normed, int rows, float* logits);
  void Clear();

private:
  struct Layer;
  struct Sync;
  struct Times {
    double products{}, attention{}, convolutions{};
  };
  // Rows [start, start + n) of a forward pass; with `sync`, chunk `chunk`
  // of a pipelined prompt.
  Times Rows(float* x, int n, int start, std::span<const std::int32_t> groups,
             Sync* sync, int chunk);
  void Project(const Layer& l, int which, const float* x, int n, float* y);
  const Weights& w_;
  int max_length_{}, length_{};
  std::vector<Layer> layers_;
  Dense::Matrix head_;
  std::size_t max_kv_{};
  std::vector<float> inv_full_, inv_swa_;
};

}  // namespace gufo::models::lfm3::npu
