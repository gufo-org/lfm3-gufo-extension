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
  // As Run, with embedding rows among the inputs: an input id i >= kRowBase stands for
  // row i - kRowBase of `rows` (hidden floats each, owned by the caller and kept for
  // its history). Needs liblfm3_xrt.so's mixed entry points.
  static constexpr std::uint32_t kRowBase = 0x80000000u;
  void RunMixed(const void* owner, std::span<const std::uint32_t> history, std::span<const std::uint32_t> inputs,
                const float* rows, std::span<float> logits);
  bool SupportsRows() const;
  // Final-normed hidden state of `owner`'s last row (hidden floats).
  void Hidden(const void* owner, std::span<float> out);
  // Forgets `owner` so its next pass replays from the start.
  void Release(const void* owner);
  // Starts decoding `token` after `history` without waiting, when the library
  // supports it; the next Run of exactly that token takes its logits. The host
  // samples, scores and streams while the NPU decodes.
  void Speculate(const void* owner, std::span<const std::uint32_t> history,
                 std::uint32_t token);
  std::uint32_t vocab() const { return vocab_; }

private:
  Engine() = default;
  struct Api;
  std::unique_ptr<Api> api_;
  void* handle_{};
  void* engine_{};
  std::uint32_t vocab_{}, max_prefill_{4096}, hidden_{2048};
  std::mutex mutex_;
  const void* owner_{};
  std::vector<std::uint32_t> resident_;  // tokens the engine context holds
  bool speculating_{};                   // resident_.back() is a started decode
  std::size_t pending_{};                // the last pending_ of resident_ wait for a pass
  static constexpr std::size_t kFreshPrefill = 1024;  // rows of the library's largest prefill
  void Flush(std::span<const std::uint32_t> tokens, std::span<float> logits);
  void Prefill(std::span<const std::uint32_t> tokens, std::span<float> logits);
  const float* rows_{};       // rows of the current RunMixed
  const float* last_rows_{};  // the latest rows a session passed (deferred rows use them)
};

}  // namespace gufo::models::lfm3::xrt
