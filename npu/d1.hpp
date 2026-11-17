#pragma once
// d1 decision prompts on the XDNA NPU: one bidirectional forward pass per
// prompt through npu::Backbone (full attention sees the whole prompt, with
// question groups; convolutions and sliding attention stay causal), then the
// LM head at the answer positions.

#include <cstdint>
#include <memory>
#include <span>

#include "../weights.hpp"

namespace gufo::models::lfm3::npu {

class Decider {
public:
  // Borrows `weights`. Throws gufo::npu::Error or std::runtime_error.
  explicit Decider(const Weights& weights);
  ~Decider();
  // The longest prompt (rows of one forward pass).
  std::uint32_t max_rows() const;
  // One prompt from position 0: rows [n][hidden] (token embeddings or media
  // rows), groups (n entries, -1 for state rows; empty: one bidirectional
  // prompt), and the logits [positions][vocab] at `positions`.
  void Decide(std::span<const float> rows, std::uint32_t n,
              std::span<const std::int32_t> groups,
              std::span<const std::uint32_t> positions,
              std::span<float> logits);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::lfm3::npu
