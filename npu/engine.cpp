#include "npu/engine.hpp"

#include <algorithm>
#include <stdexcept>

#include "npu/runtime.hpp"

namespace gufo::models::lfm3::npu {

std::unique_ptr<Engine> Engine::Open(const Weights& weights,
                                     std::uint32_t context,
                                     std::string* error) {
  try {
    std::unique_ptr<Engine> e(new Engine);
    e->npu_ = std::make_unique<Runtime>(weights, int(context));
    e->vocab_ = std::uint32_t(e->npu_->vocab());
    e->hidden_ = std::uint32_t(e->npu_->hidden());
    return e;
  } catch (const std::exception& ex) {
    if (error)
      *error = std::string("LFM3 NPU: ") + ex.what();
    return nullptr;
  }
}

Engine::~Engine() = default;

void Engine::Prefill(std::span<const std::uint32_t> tokens,
                     std::span<float> logits) {
  // A single row after context decodes; longer runs prefill in chunks.
  const float* rows = rows_ ? rows_ : last_rows_;  // deferred rows flush later
  for (std::size_t off = 0; off < tokens.size();) {
    const bool last_chunk = tokens.size() - off <= max_prefill_;
    const std::size_t n = last_chunk ? tokens.size() - off : max_prefill_;
    float* out = last_chunk && !logits.empty() ? logits.data() : nullptr;
    // the runtime's form: id < 0 takes row -id - 1
    std::vector<std::int32_t> ids(n);
    for (std::size_t i = 0; i < n; ++i)
      ids[i] = tokens[off + i] >= kRowBase
                   ? -std::int32_t(tokens[off + i] - kRowBase) - 1
                   : std::int32_t(tokens[off + i]);
    npu_->Prefill(ids, rows, out);
    resident_.insert(resident_.end(), tokens.begin() + std::ptrdiff_t(off),
                     tokens.begin() + std::ptrdiff_t(off + n));
    off += n;
  }
}

void Engine::Speculate(const void* owner,
                       std::span<const std::uint32_t> history,
                       std::uint32_t token) {
  std::lock_guard lock(mutex_);
  if (speculating_ || owner != owner_ || resident_.size() != history.size() ||
      !std::equal(history.begin(), history.end(), resident_.begin()) ||
      resident_.empty() || token >= kRowBase)
    return;
  if (pending_)
    Flush({}, {});
  try {
    npu_->DecodeStart(std::int32_t(token), nullptr);
  } catch (const std::exception&) {
    return;  // a later pass replays (resident_ unchanged)
  }
  resident_.push_back(token);
  speculating_ = true;
}

void Engine::Run(const void* owner, std::span<const std::uint32_t> history,
                 std::span<const std::uint32_t> tokens,
                 std::span<float> logits) {
  std::lock_guard lock(mutex_);
  if (speculating_) {
    speculating_ = false;
    // The decode started by Speculate is this exact step: wait for its logits.
    if (owner == owner_ && tokens.size() == 1 &&
        resident_.size() == history.size() + 1 &&
        resident_.back() == tokens[0] &&
        std::equal(history.begin(), history.end(), resident_.begin())) {
      npu_->Finish(logits.empty() ? nullptr : logits.data());
      return;
    }
    // Otherwise the started token is not part of this history: replay below.
    npu_->Finish(nullptr);
  }
  const bool current =
      owner == owner_ && resident_.size() == history.size() &&
      std::equal(history.begin(), history.end(), resident_.begin());
  if (!current) {
    // Another history: nothing of it is on the device yet.
    npu_->Clear();
    resident_.assign(history.begin(), history.end());
    pending_ = resident_.size();
    owner_ = owner;
  }
  // Inputs that need no logits wait: the next pass that does runs them all at
  // once.
  if (logits.empty() && resident_.size() + tokens.size() <= kFreshPrefill) {
    resident_.insert(resident_.end(), tokens.begin(), tokens.end());
    pending_ += tokens.size();
    return;
  }
  Flush(tokens, logits);
}

// Runs the pending inputs and `tokens` from the first pending position.
void Engine::Flush(std::span<const std::uint32_t> tokens,
                   std::span<float> logits) {
  const std::size_t base = resident_.size() - pending_;
  std::vector<std::uint32_t> work(resident_.begin() + std::ptrdiff_t(base),
                                  resident_.end());
  work.insert(work.end(), tokens.begin(), tokens.end());
  pending_ = 0;
  if (work.empty())
    return;
  resident_.resize(base);
  Prefill(work, logits);
}

void Engine::Hidden(const void* owner, std::span<float> out) {
  std::lock_guard lock(mutex_);
  if (owner != owner_ || speculating_ || resident_.empty())
    throw std::runtime_error("LFM3 NPU hidden state of another pass");
  if (pending_)
    Flush({}, {});
  npu_->Hidden(out.data());
}

void Engine::RunMixed(const void* owner, std::span<const std::uint32_t> history,
                      std::span<const std::uint32_t> inputs, const float* rows,
                      std::span<float> logits) {
  rows_ = last_rows_ = rows;
  Run(owner, history, inputs, logits);
  rows_ = nullptr;
}

void Engine::Release(const void* owner) {
  std::lock_guard lock(mutex_);
  if (owner_ == owner)
    owner_ = nullptr;
}

}  // namespace gufo::models::lfm3::npu
