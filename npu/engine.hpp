#pragma once
// Sessions over the one NPU runtime (npu/runtime.hpp). The runtime holds one
// context; the owner of the next pass replays its history when another
// session used it since. Inputs that need no logits wait for the next pass
// that does, and a long continuation prefills the whole history again when
// that is cheaper than decoding it row by row.

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::lfm3::npu {

class Runtime;

class Engine {
public:
  // Opens the bundle directory written by npu/iron/bundle.py. Returns nullptr
  // with *error set on failure.
  static std::unique_ptr<Engine> Open(const std::filesystem::path& bundle,
                                      std::uint32_t context,
                                      std::uint32_t vocab, std::string* error);
  ~Engine();
  // Appends `tokens` to `history` (the caller's inputs so far) and writes the
  // last row's logits when `logits` is not empty.
  void Run(const void* owner, std::span<const std::uint32_t> history,
           std::span<const std::uint32_t> tokens, std::span<float> logits);
  // As Run, with embedding rows among the inputs: an input id i >= kRowBase
  // stands for row i - kRowBase of `rows` (hidden floats each, owned by the
  // caller and kept for its history).
  static constexpr std::uint32_t kRowBase = 0x80000000u;
  void RunMixed(const void* owner, std::span<const std::uint32_t> history,
                std::span<const std::uint32_t> inputs, const float* rows,
                std::span<float> logits);
  // Final-normed hidden state of `owner`'s last row (hidden floats).
  void Hidden(const void* owner, std::span<float> out);
  // Forgets `owner` so its next pass replays from the start.
  void Release(const void* owner);
  // Starts decoding `token` after `history` without waiting; the next Run of
  // exactly that token takes its logits, so the host samples, scores and
  // streams while the NPU decodes.
  void Speculate(const void* owner, std::span<const std::uint32_t> history,
                 std::uint32_t token);
  std::uint32_t vocab() const { return vocab_; }

private:
  Engine() = default;
  std::unique_ptr<Runtime> npu_;
  std::uint32_t vocab_{}, max_prefill_{4096}, hidden_{2048};
  std::mutex mutex_;
  const void* owner_{};
  std::vector<std::uint32_t> resident_;  // inputs the runtime holds
  bool speculating_{};                   // resident_.back() is a started decode
  std::size_t pending_{};  // the last pending_ of resident_ wait for a pass
  static constexpr std::size_t kFreshPrefill =
      1024;  // rows of the largest prefill
  void Flush(std::span<const std::uint32_t> tokens, std::span<float> logits);
  void Prefill(std::span<const std::uint32_t> tokens, std::span<float> logits);
  const float* rows_{};       // rows of the current RunMixed
  const float* last_rows_{};  // the latest rows a session passed (deferred rows
                              // use them)
};

}  // namespace gufo::models::lfm3::npu
