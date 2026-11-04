#ifndef GUFO_MODELS_LFM3_CONTINUATION_LAYOUT_HPP_
#define GUFO_MODELS_LFM3_CONTINUATION_LAYOUT_HPP_
#include <cstdint>
#include <gufo/ext/adapter.hpp>
#include <gufo/ext/types.hpp>
#include <span>
#include <vector>

#include "config.hpp"
namespace gufo::models::lfm3 {
// State inventory shared by the adapter and its model-local checks. IDs refer
// to this representation, never to an allocation or serving slot.
enum class ContinuationPart {
  kLogits,
  kFrontier,
  kMetadata,
  kConvolution,
  // Keys then values of a sliding ring. Slots are overwritten once it wraps,
  // so the ring cannot be shared between checkpoints.
  kSlidingWindow,
  // Full attention rows are final once written.
  kKey,
  kValue,
};
struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
};
inline constexpr std::size_t kContinuationMetadataBytes = 16;

class ContinuationLayout {
public:
  ContinuationLayout(const Config&, std::uint32_t context);
  [[nodiscard]] std::span<const ContinuationComponent> Components() const {
    return components_;
  }
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      std::uint32_t position) const;
  [[nodiscard]] std::size_t PrivateBytes() const noexcept;
  // Exact row bytes at a frontier; a checkpoint adds PrivateBytes() to them.
  [[nodiscard]] std::size_t RowBytes(std::uint32_t position) const;
  // Rows, tails and private state share the same committed pages.
  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const;

private:
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
};
}  // namespace gufo::models::lfm3
#endif
