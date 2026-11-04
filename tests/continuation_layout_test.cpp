#include "../continuation_layout.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace l = gufo::models::lfm3;
namespace cache = gufo::cache;
namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}
l::Config Geometry() {
  l::Config c;
  c.hidden = 2048;
  c.vocab = 65536;
  c.context = 131072;
  c.head_dim = 64;
  c.window = 512;
  c.conv_width = 4;
  using L = l::LayerType;
  c.layers = {L::kConvolution, L::kSlidingAttention, L::kFullAttention,
              L::kConvolution};
  c.kv_heads = {0, 4, 8, 0};
  return c;
}
void TestInventory() {
  const l::ContinuationLayout layout(Geometry(), 8192);
  const auto count = [&](l::ContinuationPart part) {
    return std::ranges::count(layout.Components(), part,
                              &l::ContinuationComponent::part);
  };
  Require(count(l::ContinuationPart::kConvolution) == 2, "convolution layers");
  Require(count(l::ContinuationPart::kSlidingWindow) == 1, "sliding layers");
  Require(count(l::ContinuationPart::kKey) == 1 &&
              count(l::ContinuationPart::kValue) == 1,
          "full attention shares keys and values");
  std::uint32_t id = 0;
  for (const auto& component : layout.Components())
    Require(component.descriptor.id.value == ++id, "IDs are dense from one");
}
void TestBytes() {
  const l::ContinuationLayout layout(Geometry(), 8192);
  const std::size_t expected = 65536 * 4 + 2048 * 4 +
                               l::kContinuationMetadataBytes +
                               2 * 2048 * 3 * 4 + 2 * 512 * 4 * 64 * 4;
  Require(layout.PrivateBytes() == expected, "private bytes");
  Require(layout.RowBytes(1000) == 2 * 1000 * 8 * 64 * 4, "row bytes");
  const l::ContinuationLayout small(Geometry(), 100);
  Require(small.PrivateBytes() == expected - 2 * (512 - 100) * 4 * 64 * 4,
          "a ring holds at most the context");
}
void TestPositions() {
  const l::ContinuationLayout layout(Geometry(), 8192);
  for (const auto& position : layout.Positions(1031))
    Require(position.valid_rows == 1031, "components share the token frontier");
  bool rejected = false;
  try {
    (void)layout.Positions(8193);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Require(rejected, "a frontier beyond context is rejected");
}
void TestBacking() {
  const l::ContinuationLayout layout(Geometry(), 8192);
  Require(layout.BackingPlan(1 << 20).empty(), "tiny budgets commit nothing");
  const auto plan = layout.BackingPlan(1ULL << 30);
  Require(plan.size() == 1 && plan[0].block_bytes == 65536, "one page class");
  Require(plan[0].blocks * plan[0].block_bytes + plan[0].metadata_bytes <=
              1ULL << 30,
          "committed backing stays inside the budget");
}
}  // namespace
int main() {
  try {
    TestInventory();
    TestBytes();
    TestPositions();
    TestBacking();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
