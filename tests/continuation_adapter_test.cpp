#include "../continuation_adapter.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>


namespace l = gufo::models::lfm3;
namespace cache = gufo::cache;

namespace {
std::size_t kVocabulary = 0;
void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<class F>
void Reject(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid adapter operation was accepted");
}
struct Guard final : cache::MutationGuard {
  void BeforeOverwrite(cache::ComponentId, cache::Rows first,
                       cache::Rows end) override {
    overwrites += first < end;
  }
  void BeforeRelease(cache::ComponentId, cache::Rows,
                     cache::Rows) noexcept override {}
  void AfterReset() noexcept override { ++resets; }
  std::size_t overwrites{}, resets{};
};
struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<std::vector<std::byte>> bytes;
};
void Wait(cache::Completion done) {
  Require(done.Wait() == cache::TransferResult::kSucceeded, "transfer failed");
}
// Odd sizes make every tensor and ring boundary cross a piece boundary.
constexpr std::size_t kPiece = (1 << 18) + 13;
constexpr cache::Rows kRowPiece = 97;

auto Acquire(gufo::hip::TransferPool& streams) {
  auto stream = streams.TryAcquire();
  Require(bool(stream), "stream exhausted");
  return stream;
}

Checkpoint Capture(l::ContinuationAdapter& adapter, cache::Slot& slot,
                   gufo::hip::TransferPool& streams) {
  Checkpoint result{adapter.Positions(slot), {}};
  for (const auto& component : adapter.Components()) {
    const auto rows = result.positions[component.id.value - 1].valid_rows;
    auto& bytes = result.bytes.emplace_back(component.row_bytes
                                                ? rows * component.row_bytes
                                                : component.state_bytes);
    if (component.row_bytes) {
      for (cache::Rows first = 0; first < rows;) {
        const auto end = std::min(rows, first + kRowPiece);
        Wait(adapter.CopyRowsOut(
            slot, component.id, first, end,
            std::span(bytes).subspan(first * component.row_bytes,
                                     (end - first) * component.row_bytes),
            *Acquire(streams)));
        first = end;
      }
    } else {
      for (std::size_t offset = 0; offset < bytes.size();) {
        const auto count = std::min(kPiece, bytes.size() - offset);
        Wait(adapter.CapturePrivatePiece(
            slot, component.id, offset, std::span(bytes).subspan(offset, count),
            *Acquire(streams)));
        offset += count;
      }
    }
  }
  return result;
}

// Reverse order: metadata arrives after the device regions, and the first
// piece of each tensor arrives last.
void Load(l::ContinuationAdapter& adapter, cache::Slot& slot,
          const Checkpoint& checkpoint, gufo::hip::TransferPool& streams,
          bool omit_last = false) {
  adapter.BeginRestore(slot, checkpoint.positions);
  Require(!slot.IsValid(), "BeginRestore exposed executable state");
  Reject([&] { (void)adapter.GetSession(slot); });
  for (std::size_t i = adapter.Components().size(); i-- > 0;) {
    const auto& component = adapter.Components()[i];
    const auto bytes = std::span<const std::byte>(checkpoint.bytes[i]);
    if (component.row_bytes) {
      for (cache::Rows end = checkpoint.positions[i].valid_rows; end;) {
        const auto first = end > kRowPiece ? end - kRowPiece : 0;
        Wait(adapter.CopyRowsIn(
            slot, component.id, first, end,
            bytes.subspan(first * component.row_bytes,
                          (end - first) * component.row_bytes),
            *Acquire(streams)));
        end = first;
      }
    } else {
      for (std::size_t end = bytes.size(); end;) {
        const auto first = end > kPiece ? end - kPiece : 0;
        auto count = end - first;
        if (omit_last && component.id.value == 2 && !first)
          --count;
        Wait(adapter.LoadPrivatePiece(slot, component.id, first,
                                      bytes.subspan(first, count),
                                      *Acquire(streams)));
        end = first;
      }
    }
  }
}

std::vector<std::uint32_t> Prompt(std::size_t tokens) {
  std::vector<std::uint32_t> prompt(tokens);
  std::uint32_t state = 0x9e3779b9U;
  for (auto& token : prompt) {
    state = state * 1664525U + 1013904223U;
    token = 1000 + (state >> 8) % 60000;
  }
  return prompt;
}

std::vector<float> Logits(l::Session& session) {
  std::vector<float> logits(kVocabulary);
  std::string error;
  Require(session.ReadLogits(logits, &error), error);
  return logits;
}

// Greedy decoding through the single-token path.
std::vector<std::uint32_t> Continue(l::Session& session, std::size_t count) {
  std::vector<std::uint32_t> emitted;
  std::string error;
  while (emitted.size() < count) {
    const auto logits = Logits(session);
    const auto token = static_cast<std::uint32_t>(
        std::ranges::max_element(logits) - logits.begin());
    Require(session.Forward(std::span(&token, 1), {}, &error), error);
    emitted.push_back(token);
  }
  return emitted;
}

void Sync(l::Session& session, std::span<const std::uint32_t> prompt) {
  std::string error;
  Require(session.Forward(prompt, {}, &error), error);
}

// Prompt lengths exercise a partial and a wrapped sliding ring, and a partial
// and whole row chunks.
void CheckRoundTrip(l::ContinuationAdapter& adapter,
                    gufo::hip::TransferPool& streams, std::size_t tokens) {
  Guard source_guard, target_guard;
  auto source = adapter.CreateSlot(source_guard);
  auto target = adapter.CreateSlot(target_guard);
  const auto prompt = Prompt(tokens);
  Sync(adapter.GetSession(*source), prompt);
  Require(source_guard.overwrites > 0, "prefill bypassed the write guard");

  // A longer unrelated frontier must be fully replaced by a shorter restore.
  const auto other = Prompt(tokens + 300);
  Sync(adapter.GetSession(*target), std::span(other).subspan(7, tokens + 200));

  const auto checkpoint = Capture(adapter, *source, streams);
  auto& original = adapter.GetSession(*source);
  const auto expected_bytes = original.SaveSnapshot();
  Load(adapter, *target, checkpoint, streams);
  Require(adapter.Validate(*target, checkpoint.positions),
          "complete restore failed validation");
  Reject([&] { (void)adapter.Validate(*target, checkpoint.positions); });
  auto& restored = adapter.GetSession(*target);
  Require(restored.SaveSnapshot() == expected_bytes,
          "restored state differs from the source session");
  Require(Logits(restored) == Logits(original), "restored logits differ");
  Require(Continue(restored, 24) == Continue(original, 24),
          "restored decode diverged from an uninterrupted run");
  Require(restored.SaveSnapshot() == original.SaveSnapshot(),
          "state diverged after decoding from a restore");

  const auto later = Capture(adapter, *source, streams);
  Require(adapter.Invalidate(*target), "invalidate failed");
  Require(target_guard.resets == 1, "invalidate skipped AfterReset");
  Require(adapter.Positions(*target).front().valid_rows == 0,
          "invalidate left a frontier");
  Load(adapter, *target, later, streams);
  Require(adapter.Validate(*target, later.positions), "later restore failed");
  Require(adapter.GetSession(*target).SaveSnapshot() ==
              adapter.GetSession(*source).SaveSnapshot(),
          "later restore differs");
}

void CheckRefusals(l::ContinuationAdapter& adapter,
                   gufo::hip::TransferPool& streams) {
  Guard guard;
  auto slot = adapter.CreateSlot(guard);
  const auto prompt = Prompt(300);
  auto& session = adapter.GetSession(*slot);
  Sync(session, prompt);
  const auto checkpoint = Capture(adapter, *slot, streams);

  std::string error;
  Require(!session.RestoreSnapshot(session.SaveSnapshot(), &error),
          "an adapter slot accepted a snapshot restore");

  auto inconsistent = checkpoint.positions;
  inconsistent.back().valid_rows += 1;
  Reject([&] { adapter.BeginRestore(*slot, inconsistent); });
  Require(adapter.Positions(*slot) == checkpoint.positions,
          "refused restore changed the slot");

  // A missing byte latches failure until the slot is invalidated.
  Load(adapter, *slot, checkpoint, streams, true);
  Require(!adapter.Validate(*slot, checkpoint.positions),
          "incomplete restore passed validation");
  Require(!slot->IsValid(), "failed restore left an executable slot");
  Require(!adapter.Validate(*slot, checkpoint.positions),
          "a false validation must stay false");
  Reject([&] { (void)adapter.GetSession(*slot); });
  Require(adapter.Invalidate(*slot), "failed restore cannot be invalidated");
  Require(slot->IsValid(), "invalidate must leave a usable empty slot");

  // A cold prefill after failure reproduces the original state.
  Sync(adapter.GetSession(*slot), prompt);
  Require(adapter.Positions(*slot) == checkpoint.positions,
          "cold prefill after failure lost its frontier");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: lfm3_continuation_adapter_test MODEL.gguf\n";
    return 2;
  }
  try {
    std::string error;
    auto model = l::Model::Load(argv[1], &error);
    Require(bool(model), error);
    kVocabulary = model->config().vocab;
    l::ContinuationAdapter adapter(model, 8192, {1, 2, 3});
    Require(adapter.GetCapabilities().persistent_encoding,
            "adapter is not persistent");
    gufo::hip::TransferPool streams(2);
    CheckRefusals(adapter, streams);
    for (const std::size_t tokens : {3U, 100U, 600U, 1031U, 2500U})
      CheckRoundTrip(adapter, streams, tokens);
    std::cout << "continuation checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
