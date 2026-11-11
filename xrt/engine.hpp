#pragma once
// LFM3 on the XDNA NPU through XRT: FastFlowLM's prebuilt LFM3 engine behind
// the C shim in flm_lfm3_shim.h, loaded with dlopen. One engine holds one
// context; sessions share it and the owner of the next pass replays its
// tokens when another session used the engine since.

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::lfm3::xrt {

class Engine {
public:
  // Reads LFM3_XRT_LIBRARY (the shim), LFM3_XRT_MODEL (directory with
  // config.json and model.q4nx) and LFM3_XRT_ASSETS (directory with
  // xclbins/<model>). Returns nullptr with *error set on failure.
  static std::unique_ptr<Engine> Open(std::uint32_t context, std::uint32_t vocab, std::string* error);
  ~Engine();
  // Appends `tokens` to `history` (the caller's tokens so far) on the
  // engine and writes the last row's logits when `logits` is not empty.
  void Run(const void* owner, std::span<const std::uint32_t> history, std::span<const std::uint32_t> tokens,
           std::span<float> logits);
  // Forgets `owner` so its next pass replays from the start.
  void Release(const void* owner);
  std::uint32_t vocab() const { return vocab_; }

private:
  Engine() = default;
  struct Api;
  std::unique_ptr<Api> api_;
  void* handle_{};
  void* engine_{};
  std::uint32_t vocab_{}, max_prefill_{4096};
  std::mutex mutex_;
  const void* owner_{};
  std::vector<std::uint32_t> resident_;  // tokens the engine context holds
  void Prefill(std::span<const std::uint32_t> tokens, std::span<float> logits);
};

}  // namespace gufo::models::lfm3::xrt
