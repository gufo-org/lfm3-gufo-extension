#ifndef GUFO_MODELS_LFM3_CONTINUATION_ADAPTER_HPP_
#define GUFO_MODELS_LFM3_CONTINUATION_ADAPTER_HPP_

#include <functional>
#include <gufo/ext/adapter.hpp>
#include <gufo/ext/transfer_pool.hpp>

#include "continuation_layout.hpp"
#include "engine.hpp"

namespace gufo::models::lfm3 {

// The caller supplies the complete artifact/tokenizer/template identity. The
// adapter adds its layout and context. Slots own public model sessions, so the
// runner drives them through the ordinary Forward API.
class ContinuationAdapter final : public cache::Adapter {
public:
  ContinuationAdapter(std::shared_ptr<Model>, std::uint32_t context,
                      cache::Identity);
  ~ContinuationAdapter() override;
  [[nodiscard]] Session& GetSession(cache::Slot&);
  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(
      std::size_t ram_bytes) const override {
    return layout_.BackingPlan(ram_bytes);
  }
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components()
      const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(
      cache::MutationGuard&) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(
      const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(
      const cache::Slot&, cache::Rows first, cache::Rows end) const override;
  void BeginRestore(cache::Slot&,
                    std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&,
                                                 cache::ComponentId,
                                                 std::span<std::byte>,
                                                 cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&,
                                                      cache::ComponentId,
                                                      std::size_t,
                                                      std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>,
                                              cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&,
                                                   cache::ComponentId,
                                                   std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(
      const cache::Slot&, cache::ComponentId, cache::Rows first,
      cache::Rows end, std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId,
                                             cache::Rows first, cache::Rows end,
                                             std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] bool Validate(
      cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

private:
  class State;
  struct Region;
  [[nodiscard]] State& As(cache::Slot&) const;
  [[nodiscard]] const State& As(const cache::Slot&) const;
  [[nodiscard]] const ContinuationComponent& Component(
      cache::ComponentId) const;
  [[nodiscard]] Region PrivateRegion(State&, const ContinuationComponent&,
                                     std::uint32_t position) const;
  [[nodiscard]] void* RowData(State&, const ContinuationComponent&) const;
  [[nodiscard]] cache::Completion Transfer(
      State&, bool load, cache::Stream&,
      const std::function<void(gufo::hip::TransferStream&)>&);
  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  ContinuationLayout layout_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
};
}  // namespace gufo::models::lfm3
#endif
