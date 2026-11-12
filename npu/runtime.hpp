#pragma once
// The LFM3 backbone on the XDNA NPU through gufo::npu: our IRON kernels
// (npu/iron) in one prefill program per row count (M = 128, 512, 1024) and one
// decode program per token. A bundle directory written by npu/iron/bundle.py
// holds the programs, their instruction streams and the repacked weights.
//
// A fresh prompt of up to M tokens prefills, then its KV caches and
// convolution states are copied into the decode state; anything else decodes
// token by token. Embedding rows (audio, vision) go in place of token ids.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>

namespace gufo::models::lfm3::npu {

class Runtime {
public:
  // Throws gufo::npu::Error or std::runtime_error.
  Runtime(const std::filesystem::path& bundle, int max_length);
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
