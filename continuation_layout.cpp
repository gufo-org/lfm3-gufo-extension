#include "continuation_layout.hpp"

#include <algorithm>
#include <stdexcept>
namespace gufo::models::lfm3 {
ContinuationLayout::ContinuationLayout(const Config& c, std::uint32_t context)
    : context_(context) {
  if (!context || context > c.context || c.layers.empty() || !c.hidden ||
      !c.vocab || !c.head_dim || !c.conv_width ||
      c.kv_heads.size() != c.layers.size())
    throw std::invalid_argument("invalid LFM3 continuation geometry");
  const auto add = [&](ContinuationPart part, std::size_t row,
                       std::size_t state, std::uint32_t layer = 0) {
    const auto id = static_cast<std::uint32_t>(components_.size() + 1);
    components_.push_back({{{id},
                            1,
                            row ? cache::ComponentKind::kAppendRows
                                : cache::ComponentKind::kPrivateState,
                            row,
                            row ? cache::Rows{256} : 0,
                            state},
                           part,
                           layer});
  };
  add(ContinuationPart::kLogits, 0, std::size_t{c.vocab} * sizeof(float));
  add(ContinuationPart::kFrontier, 0, std::size_t{c.hidden} * sizeof(float));
  add(ContinuationPart::kMetadata, 0, kContinuationMetadataBytes);
  for (std::uint32_t layer = 0; layer < c.layers.size(); ++layer) {
    const std::size_t row =
        std::size_t{c.kv_heads[layer]} * c.head_dim * sizeof(float);
    switch (c.layers[layer]) {
      case LayerType::kConvolution:
        add(ContinuationPart::kConvolution, 0,
            std::size_t{c.hidden} * (c.conv_width - 1) * sizeof(float), layer);
        break;
      case LayerType::kSlidingAttention:
        add(ContinuationPart::kSlidingWindow, 0,
            2 * row * std::min(context, c.window), layer);
        break;
      case LayerType::kFullAttention:
        add(ContinuationPart::kKey, row, 0, layer);
        add(ContinuationPart::kValue, row, 0, layer);
        break;
    }
  }
}
std::vector<cache::ComponentPosition> ContinuationLayout::Positions(
    std::uint32_t position) const {
  if (position > context_)
    throw std::invalid_argument("invalid LFM3 continuation frontier");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(components_.size());
  for (const auto& component : components_)
    positions.push_back({component.descriptor.id, position});
  return positions;
}
std::size_t ContinuationLayout::PrivateBytes() const noexcept {
  std::size_t bytes = 0;
  for (const auto& component : components_)
    bytes += component.descriptor.state_bytes;
  return bytes;
}
std::size_t ContinuationLayout::RowBytes(std::uint32_t position) const {
  const auto positions = Positions(position);
  std::size_t bytes = 0;
  for (std::size_t i = 0; i < components_.size(); ++i)
    bytes += components_[i].descriptor.row_bytes * positions[i].valid_rows;
  return bytes;
}
std::vector<cache::BackingClass> ContinuationLayout::BackingPlan(
    std::size_t ram) const {
  // Contiguous page groups retain one payload owner and charge all slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header)
    return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks)
    return {};
  return {{page, blocks, header + blocks * per_block}};
}
}  // namespace gufo::models::lfm3
