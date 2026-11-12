#include "engine.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace gufo::models::lfm3::xrt {

struct Engine::Api {
  void* (*create)(const char*, const char*, int);
  void (*destroy)(void*);
  const char* (*last_error)();
  int (*vocab)(const void*);
  int (*prefill)(void*, const std::int32_t*, int, float*);
  int (*decode)(void*, std::int32_t, float*);
  void (*clear)(void*);
  // optional (liblfm3_xrt.so): a decode that returns at once, then its wait
  int (*decode_start)(void*, std::int32_t) = nullptr;
  int (*decode_finish)(void*, float*) = nullptr;
};

std::unique_ptr<Engine> Engine::Open(std::uint32_t context, std::uint32_t vocab, std::string* error) {
  auto fail = [&](std::string message) -> std::unique_ptr<Engine> {
    if (error)
      *error = "LFM3 XRT: " + message;
    return nullptr;
  };
  const char* library = std::getenv("LFM3_XRT_LIBRARY");
  const char* model = std::getenv("LFM3_XRT_MODEL");
  const char* assets = std::getenv("LFM3_XRT_ASSETS");
  if (!library || !model || !assets)
    return fail("set LFM3_XRT_LIBRARY, LFM3_XRT_MODEL and LFM3_XRT_ASSETS");
  std::unique_ptr<Engine> e(new Engine);
  e->handle_ = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!e->handle_)
    return fail(std::string("dlopen: ") + dlerror());
  e->api_ = std::make_unique<Api>();
  auto bind = [&](auto& fn, const char* name) {
    fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(dlsym(e->handle_, name));
    return fn != nullptr;
  };
  auto& a = *e->api_;
  if (!bind(a.create, "flm_lfm3_create") || !bind(a.destroy, "flm_lfm3_destroy") ||
      !bind(a.last_error, "flm_lfm3_last_error") || !bind(a.vocab, "flm_lfm3_vocab") ||
      !bind(a.prefill, "flm_lfm3_prefill") || !bind(a.decode, "flm_lfm3_decode") || !bind(a.clear, "flm_lfm3_clear"))
    return fail("the library lacks the flm_lfm3 entry points");
  if (!bind(a.decode_start, "flm_lfm3_decode_start") ||
      !bind(a.decode_finish, "flm_lfm3_decode_finish"))
    a.decode_start = nullptr, a.decode_finish = nullptr;
  e->engine_ = a.create(model, assets, int(context));
  if (!e->engine_)
    return fail(a.last_error());
  e->vocab_ = std::uint32_t(a.vocab(e->engine_));
  if (e->vocab_ != vocab)
    return fail("engine vocabulary " + std::to_string(e->vocab_) + " differs from the model's " + std::to_string(vocab));
  return e;
}

Engine::~Engine() {
  if (engine_)
    api_->destroy(engine_);
  if (handle_)
    dlclose(handle_);
}

void Engine::Prefill(std::span<const std::uint32_t> tokens, std::span<float> logits) {
  // A single row after context decodes; longer runs prefill in chunks the
  // engine admits.
  for (std::size_t off = 0; off < tokens.size();) {
    const bool last_chunk = tokens.size() - off <= max_prefill_;
    const std::size_t n = last_chunk ? tokens.size() - off : max_prefill_;
    float* out = last_chunk && !logits.empty() ? logits.data() : nullptr;
    int rc;
    if (n == 1 && !resident_.empty())
      rc = api_->decode(engine_, std::int32_t(tokens[off]), out);
    else
      rc = api_->prefill(engine_, reinterpret_cast<const std::int32_t*>(tokens.data() + off), int(n), out);
    if (rc)
      throw std::runtime_error(std::string("LFM3 XRT engine: ") + api_->last_error());
    resident_.insert(resident_.end(), tokens.begin() + off, tokens.begin() + off + n);
    off += n;
  }
}

void Engine::Speculate(const void* owner,
                       std::span<const std::uint32_t> history,
                       std::uint32_t token) {
  std::lock_guard lock(mutex_);
  if (!api_->decode_start || speculating_ || owner != owner_ ||
      resident_.size() != history.size() ||
      !std::equal(history.begin(), history.end(), resident_.begin()) ||
      resident_.empty())
    return;
  if (api_->decode_start(engine_, std::int32_t(token)))
    return;  // a later pass replays (resident_ unchanged)
  resident_.push_back(token);
  speculating_ = true;
}

void Engine::Run(const void* owner, std::span<const std::uint32_t> history, std::span<const std::uint32_t> tokens,
                 std::span<float> logits) {
  std::lock_guard lock(mutex_);
  if (speculating_) {
    speculating_ = false;
    // The decode started by Speculate is this exact step: wait for its logits.
    if (owner == owner_ && tokens.size() == 1 &&
        resident_.size() == history.size() + 1 &&
        resident_.back() == tokens[0] &&
        std::equal(history.begin(), history.end(), resident_.begin())) {
      if (api_->decode_finish(engine_,
                              logits.empty() ? nullptr : logits.data()))
        throw std::runtime_error(std::string("LFM3 XRT engine: ") +
                                 api_->last_error());
      return;
    }
    // Otherwise the started token is not part of this history: the checks below
    // replay.
    if (api_->decode_finish(engine_, nullptr))
      throw std::runtime_error(std::string("LFM3 XRT engine: ") +
                               api_->last_error());
  }
  const bool current = owner == owner_ && resident_.size() == history.size() &&
                       std::equal(history.begin(), history.end(), resident_.begin());
  if (!current) {
    api_->clear(engine_);
    resident_.clear();
    owner_ = owner;
    if (!history.empty()) {
      std::vector<std::uint32_t> all(history.begin(), history.end());
      all.insert(all.end(), tokens.begin(), tokens.end());
      Prefill(all, logits);
      return;
    }
  }
  Prefill(tokens, logits);
}

void Engine::Release(const void* owner) {
  std::lock_guard lock(mutex_);
  if (owner_ == owner)
    owner_ = nullptr;
}

}  // namespace gufo::models::lfm3::xrt
