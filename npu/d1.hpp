#pragma once
// d1 decision prompts on the XDNA NPU: one bidirectional prefill per prompt
// (npu/iron/prefill, kinds "d1": full attention sees the whole prompt, with
// question groups; convolutions and sliding attention stay causal), then the
// LM head at the answer positions through npu::Dense. The bundle is the one
// `npu/iron/build.sh` writes for a d1 GGUF.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

#include "../weights.hpp"

namespace gufo::models::lfm3::npu {

class Decider {
public:
  // Throws gufo::npu::Error or std::runtime_error.
  Decider(const std::filesystem::path& bundle, const Weights& weights);
  ~Decider();
  // The longest prompt (rows of one prefill).
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
