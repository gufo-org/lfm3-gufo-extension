#include "continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <gufo/ext/snapshot_copy.hpp>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "continuation_hooks.hpp"

namespace gufo::models::lfm3 {
namespace {
using cache::Rows;
using cache::TransferResult;
void Check(hipError_t result) {
  if (result != hipSuccess)
    throw std::runtime_error(hipGetErrorString(result));
}
struct Metadata {
  std::uint32_t version{1};
  std::uint32_t position{};
};
static_assert(sizeof(Metadata) <= kContinuationMetadataBytes);
using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;
bool Covered(Coverage ranges, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = 0;
  for (const auto& [first, last] : ranges) {
    if (first > frontier)
      return false;
    frontier = std::max(frontier, last);
  }
  return frontier == end;
}
}  // namespace

class ContinuationAdapter::State final : public cache::Slot,
                                         public ContinuationHooks {
public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard,
        std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    this->session->SetContinuationHooks(this);
  }
  struct Job {
    State* owner;
    bool load;
    std::optional<cache::Completion> completion;
    std::optional<TransferResult> result;
    mutable std::mutex mutex;
    bool Settled() const noexcept {
      const std::lock_guard lock(mutex);
      return result.has_value();
    }
    bool Ready() const noexcept {
      const std::lock_guard lock(mutex);
      return result || completion->Ready();
    }
    TransferResult Wait() noexcept {
      const std::lock_guard lock(mutex);
      if (!result) {
        result = completion->Wait();
        completion.reset();
        if (load && *result == TransferResult::kFailed)
          owner->failed = true;
        --owner->pending;
      }
      return *result;
    }
  };
  class Signal final : public cache::CompletionSignal {
  public:
    explicit Signal(std::shared_ptr<Job> job) : job_(std::move(job)) {}
    bool Ready() const noexcept override { return job_->Ready(); }
    TransferResult Wait() noexcept override { return job_->Wait(); }

  private:
    std::shared_ptr<Job> job_;
  };
  ~State() override {
    for (const auto& job : jobs)
      (void)job->Wait();
    Release();
    session->SetContinuationHooks(nullptr);
  }
  bool IsValid() const noexcept override {
    return !restoring && !failed && session->IsCoherent();
  }
  void RequireIdle() const {
    if (restoring || failed || pending)
      throw std::logic_error("LFM3 slot is not ready for execution");
  }
  void BeforeWrite(std::uint32_t first, std::uint32_t end) override {
    RequireIdle();
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(component.descriptor.id, first, end);
  }
  void BeforeReset() override {
    if (invalidating)
      return;
    RequireIdle();
    GuardAll();
  }
  void BeforeRestore() override {
    throw std::logic_error("LFM3 slots restore through the adapter");
  }
  void GuardAll() {
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeOverwrite(component.descriptor.id, 0, adapter.context_);
  }
  void Release() noexcept {
    for (const auto& component : adapter.layout_.Components())
      if (component.descriptor.kind == cache::ComponentKind::kAppendRows)
        guard.BeforeRelease(component.descriptor.id, 0, adapter.context_);
  }
  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  bool restoring{false}, invalidating{false};
  std::atomic<bool> failed{false};
  std::atomic<std::size_t> pending{0};
  std::mutex jobs_mutex;
  std::vector<std::shared_ptr<Job>> jobs;
  std::vector<cache::ComponentPosition> expected;
  std::vector<Coverage> coverage;
  std::array<std::byte, kContinuationMetadataBytes> metadata{};
};

struct ContinuationAdapter::Region {
  struct Piece {
    void* pointer;
    std::size_t offset, bytes;
    bool host{false};
  };
  std::vector<Piece> pieces;
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model,
                                         std::uint32_t context,
                                         cache::Identity identity)
    : model_(std::move(model)),
      context_(context),
      layout_(model_ ? model_->config() : Config{}, context),
      identity_(std::move(identity)) {
  if (identity_.empty())
    throw std::invalid_argument("invalid LFM3 adapter configuration");
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i)
      identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(2);  // Component layout version: F16 K/V rows.
  put(context);
  for (const auto& component : layout_.Components())
    descriptors_.push_back(component.descriptor);
}
ContinuationAdapter::~ContinuationAdapter() = default;
ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this)
    throw std::invalid_argument("slot belongs to another LFM3 adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(
    const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  state.RequireIdle();
  return *state.session;
}
cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components()
    const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const {
  return identity_;
}
std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(
    cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(context_, &error);
  if (!session)
    throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}
std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(
    const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid())
    throw std::logic_error("cannot query an invalid LFM3 slot");
  return layout_.Positions(state.session->Position());
}
std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot,
                                                   Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_)
    throw std::invalid_argument("invalid LFM3 prefill plan");
  std::vector<Rows> result;
  constexpr Rows capacity = Session::kPrefillChunk;
  while (first < end) {
    first += std::min(end - first, capacity);
    result.push_back(first);
  }
  return result;
}
const ContinuationComponent& ContinuationAdapter::Component(
    cache::ComponentId id) const {
  if (!id.value || id.value > layout_.Components().size())
    throw std::invalid_argument("invalid LFM3 component ID");
  return layout_.Components()[id.value - 1];
}
void* ContinuationAdapter::RowData(
    State& state, const ContinuationComponent& component) const {
  switch (component.part) {
    case ContinuationPart::kKey:
      return state.session->ContinuationData(ContinuationTensor::kKey,
                                             component.layer);
    case ContinuationPart::kValue:
      return state.session->ContinuationData(ContinuationTensor::kValue,
                                             component.layer);
    default:
      throw std::invalid_argument("component has no append rows");
  }
}

void ContinuationAdapter::BeginRestore(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  state.RequireIdle();
  if (positions.size() != descriptors_.size())
    throw std::invalid_argument("incomplete LFM3 restore positions");
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& descriptor : descriptors_) {
    if (std::ranges::count(positions, descriptor.id,
                           &cache::ComponentPosition::id) != 1)
      throw std::invalid_argument("missing or duplicate LFM3 frontier");
    ordered.push_back(*std::ranges::find(positions, descriptor.id,
                                         &cache::ComponentPosition::id));
  }
  if (ordered.front().valid_rows > context_)
    throw std::invalid_argument("LFM3 frontier exceeds context");
  const auto target = static_cast<std::uint32_t>(ordered.front().valid_rows);
  if (ordered != layout_.Positions(target))
    throw std::invalid_argument("inconsistent LFM3 component frontiers");
  // Reserve first; any refusal leaves execution valid. Guard all ranges while
  // the original state is still readable.
  std::vector<Coverage> coverage(descriptors_.size());
  state.GuardAll();
  state.session->BeginRestore();
  state.expected = std::move(ordered);
  state.coverage = std::move(coverage);
  state.metadata.fill(std::byte{});
  state.restoring = true;
}

cache::Completion ContinuationAdapter::Transfer(
    State& state, bool load, cache::Stream& stream,
    const std::function<void(gufo::hip::TransferStream&)>& submit) {
  auto& hip_stream = dynamic_cast<gufo::hip::TransferStream&>(stream);
  auto job = std::make_shared<State::Job>();
  job->owner = &state;
  job->load = load;
  auto signal = std::make_unique<State::Signal>(job);
  {
    const std::lock_guard lock(state.jobs_mutex);
    std::erase_if(state.jobs,
                  [](const auto& previous) { return previous->Settled(); });
    state.jobs.push_back(job);
  }
  ++state.pending;
  try {
    submit(hip_stream);
    job->completion.emplace(hip_stream.Complete());
  } catch (...) {
    (void)hip_stream.Synchronize();
    job->result = TransferResult::kFailed;
    --state.pending;
    if (load)
      state.failed = true;
    throw;
  }
  return cache::Completion(std::move(signal));
}

// Logical byte layout of one private component at a frontier. A load may
// arrive before the metadata, so regions follow from the frontier alone. Rings
// are copied in physical order, which keeps the absolute-position mapping.
ContinuationAdapter::Region ContinuationAdapter::PrivateRegion(
    State& state, const ContinuationComponent& component,
    std::uint32_t position) const {
  Region result;
  const bool host_state = state.session->HostState();
  const auto add = [&](void* pointer, std::size_t offset, std::size_t bytes,
                       bool host = false) {
    if (bytes)
      result.pieces.push_back({pointer, offset, bytes, host || host_state});
  };
  const auto bytes = component.descriptor.state_bytes;
  auto& session = *state.session;
  switch (component.part) {
    case ContinuationPart::kLogits:
      add(session.ContinuationData(ContinuationTensor::kLogits), 0, bytes);
      break;
    case ContinuationPart::kFrontier:
      add(session.ContinuationData(ContinuationTensor::kFrontier), 0, bytes);
      break;
    case ContinuationPart::kMetadata:
      add(state.metadata.data(), 0, bytes, true);
      break;
    case ContinuationPart::kConvolution:
      add(session.ContinuationData(ContinuationTensor::kConvolution,
                                   component.layer),
          0, bytes);
      break;
    case ContinuationPart::kSlidingWindow: {
      const auto capacity = session.RingCapacity(component.layer);
      const auto row = bytes / 2 / capacity;
      const auto live = std::size_t{std::min(position, capacity)} * row;
      add(session.ContinuationData(ContinuationTensor::kKey, component.layer),
          0, live);
      add(session.ContinuationData(ContinuationTensor::kValue, component.layer),
          bytes / 2, live);
      break;
    }
    default:
      throw std::invalid_argument("component is not private state");
  }
  return result;
}

cache::Completion ContinuationAdapter::CapturePrivate(
    const cache::Slot& slot, cache::ComponentId id, std::span<std::byte> bytes,
    cache::Stream& stream) {
  if (bytes.size() != Component(id).descriptor.state_bytes)
    throw std::invalid_argument("incorrect LFM3 private buffer size");
  return CapturePrivatePiece(slot, id, 0, bytes, stream);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(
    const cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<std::byte> bytes, cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  if (!state.IsValid())
    throw std::logic_error("cannot capture an invalid LFM3 frontier");
  const auto& component = Component(id);
  const auto total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
      offset > total || bytes.size() > total - offset)
    throw std::invalid_argument("invalid LFM3 private piece");
  const auto position = state.session->Position();
  std::array<std::byte, kContinuationMetadataBytes> host_metadata{};
  auto region = [&] {
    if (component.part == ContinuationPart::kMetadata) {
      const Metadata metadata{.position = position};
      std::memcpy(host_metadata.data(), &metadata, sizeof(metadata));
      return Region{{{host_metadata.data(), 0, host_metadata.size(), true}}};
    }
    // Logits are computed lazily from the frontier.
    if (component.part == ContinuationPart::kLogits)
      state.session->PrepareLogits();
    return PrivateRegion(state, component, position);
  }();
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        // Pieces overwrite their ranges. Clear only uncovered bytes: clearing
        // the whole recycled copy on the host dominates capture time.
        std::vector<std::pair<std::size_t, std::size_t>> covered;
        covered.reserve(region.pieces.size());
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first < end)
            covered.emplace_back(first - offset, end - offset);
        }
        std::ranges::sort(covered);
        std::size_t cleared = 0;
        for (const auto& [first, end] : covered) {
          if (first > cleared)
            std::fill(bytes.begin() + cleared, bytes.begin() + first,
                      std::byte{});
          cleared = std::max(cleared, end);
        }
        std::fill(bytes.begin() + std::min(cleared, bytes.size()), bytes.end(),
                  std::byte{});
        for (const auto& piece : region.pieces) {
          const auto first = std::max(offset, piece.offset);
          const auto end =
              std::min(offset + bytes.size(), piece.offset + piece.bytes);
          if (first >= end)
            continue;
          auto* source =
              static_cast<std::byte*>(piece.pointer) + first - piece.offset;
          auto* destination = bytes.data() + first - offset;
          if (piece.host)
            std::memcpy(destination, source, end - first);
          else
            Check(hip::CopySnapshotToHost(destination, source, end - first,
                                          transfer.Native()));
        }
      });
}
cache::Completion ContinuationAdapter::LoadPrivate(
    cache::Slot& slot, cache::ComponentId id, std::span<const std::byte> bytes,
    cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("LFM3 load requires an unfailed restore");
  try {
    if (bytes.size() != Component(id).descriptor.state_bytes)
      throw std::invalid_argument("incorrect LFM3 private buffer size");
    return LoadPrivatePiece(slot, id, 0, bytes, stream);
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(
    cache::Slot& slot, cache::ComponentId id, std::size_t offset,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("LFM3 load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState ||
        offset > total || bytes.size() > total - offset)
      throw std::invalid_argument("invalid LFM3 private piece");
    const auto region = PrivateRegion(
        state, component,
        static_cast<std::uint32_t>(state.expected.front().valid_rows));
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          for (const auto& piece : region.pieces) {
            const auto first = std::max(offset, piece.offset);
            const auto end =
                std::min(offset + bytes.size(), piece.offset + piece.bytes);
            if (first >= end)
              continue;
            auto* destination =
                static_cast<std::byte*>(piece.pointer) + first - piece.offset;
            const auto* source = bytes.data() + first - offset;
            if (piece.host)
              std::memcpy(destination, source, end - first);
            else
              Check(hip::CopySnapshotToDevice(destination, source, end - first,
                                              transfer.Native()));
          }
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}
cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot,
                                                   cache::ComponentId id,
                                                   Rows first, Rows end,
                                                   std::span<std::byte> bytes,
                                                   cache::Stream& stream) {
  auto& state = As(const_cast<cache::Slot&>(slot));
  const auto& component = Component(id);
  if (!state.IsValid())
    throw std::logic_error("cannot read an invalid LFM3 slot");
  const auto positions = Positions(slot);
  const auto row = component.descriptor.row_bytes;
  if (!row || first > end || end > positions[id.value - 1].valid_rows ||
      bytes.size() != (end - first) * row)
    throw std::invalid_argument("invalid LFM3 row capture");
  auto* source =
      bytes.empty()
          ? nullptr
          : static_cast<std::byte*>(RowData(state, component)) + first * row;
  return Transfer(
      state, false, stream, [&](gufo::hip::TransferStream& transfer) {
        if (!bytes.empty() && state.session->HostState())
          std::memcpy(bytes.data(), source, bytes.size());
        else if (!bytes.empty())
          Check(hip::CopySnapshotToHost(bytes.data(), source, bytes.size(),
                                        transfer.Native()));
      });
}
cache::Completion ContinuationAdapter::CopyRowsIn(
    cache::Slot& slot, cache::ComponentId id, Rows first, Rows end,
    std::span<const std::byte> bytes, cache::Stream& stream) {
  auto& state = As(slot);
  if (!state.restoring || state.failed)
    throw std::logic_error("LFM3 load requires an unfailed restore");
  try {
    const auto& component = Component(id);
    const auto row = component.descriptor.row_bytes;
    if (!row || first > end || end > state.expected[id.value - 1].valid_rows ||
        bytes.size() != (end - first) * row)
      throw std::invalid_argument("invalid LFM3 row restore");
    auto* destination =
        bytes.empty()
            ? nullptr
            : static_cast<std::byte*>(RowData(state, component)) + first * row;
    {
      const std::lock_guard lock(state.jobs_mutex);
      state.coverage[id.value - 1].emplace_back(first * row, end * row);
    }
    return Transfer(
        state, true, stream, [&](gufo::hip::TransferStream& transfer) {
          if (!bytes.empty() && state.session->HostState())
            std::memcpy(destination, bytes.data(), bytes.size());
          else if (!bytes.empty())
            Check(hip::CopySnapshotToDevice(destination, bytes.data(),
                                            bytes.size(), transfer.Native()));
        });
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(
    cache::Slot& slot, std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring)
    throw std::logic_error("LFM3 validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || state.pending ||
      positions.size() != state.expected.size())
    return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1)
      return refuse();
    const auto& descriptor = descriptors_[i];
    const auto total = descriptor.row_bytes
                           ? expected.valid_rows * descriptor.row_bytes
                           : descriptor.state_bytes;
    if (!Covered(state.coverage[i], total))
      return refuse();
  }
  Metadata metadata{};
  std::memcpy(&metadata, state.metadata.data(), sizeof(metadata));
  if (metadata.version != 1 ||
      metadata.position != state.expected.front().valid_rows)
    return refuse();
  state.session->FinishRestore(metadata.position);
  state.restoring = false;
  state.coverage.clear();
  state.expected.clear();
  return true;
}
bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  try {
    auto& state = As(slot);
    if (state.pending)
      return false;
    state.Release();
    struct ResetScope {
      bool& active;
      ~ResetScope() { active = false; }
    } reset_scope{state.invalidating};
    state.invalidating = true;
    if (!state.session->Reset())
      return false;
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.guard.AfterReset();
    return true;
  } catch (...) {
    return false;
  }
}
}  // namespace gufo::models::lfm3
