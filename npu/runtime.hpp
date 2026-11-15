#pragma once
// The LFM3 backbone on the XDNA NPU (npu::Backbone): prompts of any length
// prefill as one forward pass, at any position; single rows decode. Embedding
// rows (audio, vision) go in place of token ids.

#include <cstdint>
#include <memory>
#include <span>

#include "../weights.hpp"

namespace gufo::models::lfm3::npu {

class Runtime {
public:
  // Borrows `weights`. Throws gufo::npu::Error or std::runtime_error.
  Runtime(const Weights& weights, int max_length);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  int vocab() const;
  int hidden() const;
  int length() const;  // positions held
  // Appends inputs and writes the last row's logits (vocab floats) when
  // `logits` is not null. An id < 0 takes embedding row -id - 1 of `rows`.
  void Prefill(std::span<const std::int32_t> ids, const float* rows,
               float* logits);
  // Starts decoding `id` (a token, or -1 for `row`) and returns at once;
  // Finish (or any other call) waits for it.
  void DecodeStart(std::int32_t id, const float* row);
  void Finish(float* logits);
  // Final-normed hidden state of the last row (hidden floats).
  void Hidden(float* out);
  void Clear();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gufo::models::lfm3::npu
