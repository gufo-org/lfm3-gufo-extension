#include "npu/d1.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "npu/backbone.hpp"

namespace gufo::models::lfm3::npu {
namespace {
constexpr int kMaxRows = 4096;
}  // namespace

struct Decider::Impl {
  Backbone net;
  explicit Impl(const Weights& w) : net(w, kMaxRows) {}
};

Decider::Decider(const Weights& w) : impl_(std::make_unique<Impl>(w)) {}
Decider::~Decider() = default;

std::uint32_t Decider::max_rows() const {
  return std::uint32_t(impl_->net.max_length());
}

void Decider::Decide(std::span<const float> rows, std::uint32_t n,
                     std::span<const std::int32_t> groups,
                     std::span<const std::uint32_t> positions,
                     std::span<float> logits) {
  auto& net = impl_->net;
  const auto& c = net.config();
  const std::size_t H = c.hidden;
  if (n == 0 || n > max_rows() || rows.size() < n * H)
    throw std::invalid_argument("d1 on the NPU takes 1 to " +
                                std::to_string(max_rows()) + " rows");
  if (!groups.empty() && groups.size() != n)
    throw std::invalid_argument("d1 groups differ from the rows");
  if (logits.size() < positions.size() * c.vocab)
    throw std::invalid_argument("d1 logits too small");
  // No groups: every row sees every row (all state rows).
  std::vector<std::int32_t> all;
  if (groups.empty()) {
    all.assign(n, -1);
    groups = all;
  }
  std::vector<float> x(rows.begin(), rows.begin() + std::ptrdiff_t(n * H));
  net.Clear();
  net.Forward(x.data(), int(n), groups);
  std::vector<float> picked(positions.size() * H), normed(picked.size());
  for (std::size_t p = 0; p < positions.size(); ++p) {
    if (positions[p] >= n)
      throw std::out_of_range("d1 answer position");
    std::copy_n(x.data() + std::size_t(positions[p]) * H, H,
                picked.data() + p * H);
  }
  net.FinalNorm(picked.data(), int(positions.size()), normed.data());
  net.Head(normed.data(), int(positions.size()), logits.data());
  net.Clear();
}

}  // namespace gufo::models::lfm3::npu
