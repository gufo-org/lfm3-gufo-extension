#include "npu/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <vector>

#include "npu/backbone.hpp"

namespace gufo::models::lfm3::npu {

struct Runtime::Impl {
  Backbone net;
  int vocab{}, hidden{};
  std::vector<float> last;    // residual stream of the last row
  std::vector<float> normed;  // final-normed last row
  std::vector<float> logits;  // of the started decode
  std::future<void> pending;

  Impl(const Weights& w, int max_length)
      : net(w, max_length),
        vocab(int(w.config.vocab)),
        hidden(int(w.config.hidden)),
        last(std::size_t(hidden)),
        normed(std::size_t(hidden)),
        logits(std::size_t(vocab)) {}

  void Wait() {
    if (pending.valid())
      pending.get();  // rethrows the decode's error
  }

  // Runs ids through the backbone; writes the last row's logits when set.
  void Run(std::span<const std::int32_t> ids, const float* rows, float* out) {
    const int n = int(ids.size());
    std::vector<float> x(std::size_t(n) * hidden);
    for (int i = 0; i < n; ++i) {
      float* row = x.data() + std::size_t(i) * hidden;
      if (ids[i] >= 0)
        net.Embed(ids[i], row);
      else if (rows)
        std::memcpy(row, rows + std::size_t(-ids[i] - 1) * hidden,
                    std::size_t(hidden) * 4);
      else
        throw std::invalid_argument("NPU embedding row without rows");
    }
    net.Forward(x.data(), n);
    std::copy_n(x.data() + std::size_t(n - 1) * hidden, hidden, last.data());
    net.FinalNorm(last.data(), 1, normed.data());
    if (out)
      net.Head(normed.data(), 1, out);
  }
};

Runtime::Runtime(const Weights& weights, int max_length)
    : impl_(std::make_unique<Impl>(weights, max_length)) {}
Runtime::~Runtime() {
  try {
    impl_->Wait();
  } catch (...) {
  }
}
int Runtime::vocab() const {
  return impl_->vocab;
}
int Runtime::hidden() const {
  return impl_->hidden;
}
int Runtime::length() const {
  impl_->Wait();
  return impl_->net.length();
}

void Runtime::Prefill(std::span<const std::int32_t> ids, const float* rows,
                      float* logits) {
  impl_->Wait();
  if (ids.empty())
    return;
  impl_->Run(ids, rows, logits);
}

void Runtime::DecodeStart(std::int32_t id, const float* row) {
  auto& m = *impl_;
  m.Wait();
  std::vector<float> copy;
  if (id < 0) {
    if (!row)
      throw std::invalid_argument("NPU embedding row without rows");
    copy.assign(row, row + m.hidden);
  }
  m.pending = std::async(std::launch::async, [&m, id, copy = std::move(copy)] {
    const std::int32_t ids[] = {id < 0 ? -1 : id};
    m.Run(ids, copy.data(), m.logits.data());
  });
}

void Runtime::Finish(float* logits) {
  auto& m = *impl_;
  const bool had = m.pending.valid();
  m.Wait();
  if (had && logits)
    std::copy(m.logits.begin(), m.logits.end(), logits);
}

void Runtime::Hidden(float* out) {
  impl_->Wait();
  if (!impl_->net.length())
    throw std::runtime_error("no rows yet");
  std::copy(impl_->normed.begin(), impl_->normed.end(), out);
}

void Runtime::Clear() {
  impl_->Wait();
  impl_->net.Clear();
}

}  // namespace gufo::models::lfm3::npu
