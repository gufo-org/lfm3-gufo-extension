#include "runner.hpp"

#include <algorithm>
#include <cstring>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/host_budget.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "chat_template.hpp"

namespace gufo::models::lfm3 {
namespace {
using namespace gufo::ext;

class State final : public TextRunnerState {
public:
  std::unique_ptr<Session> session;
  std::vector<float> logits;
  bool logits_ready{};
  void Invalidate() noexcept override {
    (void)session->Reset();
    logits_ready = false;
  }
};

State& Get(TextRunnerState& state) {
  return dynamic_cast<State&>(state);
}
const State& Get(const TextRunnerState& state) {
  return dynamic_cast<const State&>(state);
}
void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}

class Lfm3Snapshot final : public TextRunnerSnapshot {
public:
  Lfm3Snapshot(std::shared_ptr<Model> owner, std::vector<std::uint8_t> payload)
      : owner(std::move(owner)), payload(std::move(payload)) {}
  std::size_t PayloadBytes() const noexcept override { return payload.size(); }
  std::shared_ptr<Model> owner;
  std::vector<std::uint8_t> payload;
};

class Runner final : public HipTextModelRunner {
public:
  Runner(std::shared_ptr<Model> model, std::uint32_t context,
         std::string artifact_fingerprint)
      : model_(std::move(model)), context_(context) {
    if (!artifact_fingerprint.empty()) {
      if (artifact_fingerprint.size() != 64 ||
          artifact_fingerprint.find_first_not_of("0123456789abcdef") !=
              std::string::npos)
        throw std::invalid_argument("invalid LFM3 artifact fingerprint");
      const auto identity =
          "schema=gufo-text-continuation-v2\nmodel_kind=lfm3\nartifact_id=" +
          std::string(core::kGgufIdentityScheme) + ':' + artifact_fingerprint +
          "\nstate_abi=lfm3-fp32-v2\npayload_layout=lfm3-fp32-compact-v1\n" +
          "tokenizer=lfm2-byte-bpe-v2-gguf-controls\nchat_template=lfm3-native-"
          "v2\n" +
          "numerics=fixed-point-three-digit-v1-wave32-attention-v1\ncontext_"
          "tokens=" +
          std::to_string(context_) +
          "\nposition_policy=absolute-ring-v1\nadapters=none\n";
      persistence_ = TextRunnerPersistenceDescriptor{
          {identity.begin(), identity.end()}, 1};
    }
  }
  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = "LFM3-1.7B",
            .state_abi = "lfm3-fp32-v2",
            .max_context = context_,
            .capabilities = {.incremental_prefill = true,
                             .snapshot = true,
                             .fork = true,
                             .final_token_advance_required = true,
                             .incremental_text_is_exact = true,
                             .prefix_reuse = true,
                             .partial_prefill_retention = true},
            .persistence = persistence_};
  }
  TextRunnerResourceClaim ResourceClaim() const override {
    return {.resident_weights_bytes = 0,
            .state_capacity_bytes = 0,
            .per_request_state_bytes = 0,
            .temporary_scratch_bytes = 0,
            .retained_snapshot_capacity_bytes = HostSnapshotBudgetBytes(),
            .retained_snapshot_ceiling_bytes = HostSnapshotCeilingBytes()};
  }
  std::shared_ptr<const sampling::ConstraintVocabulary>
  BuildConstraintVocabulary() const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        VocabularySize(), [this](std::uint32_t id) {
          return sampling::ConstraintVocabulary::Piece{
              std::string(model_->tokenizer().Piece(id)),
              model_->tokenizer().IsStop(id)};
        });
  }
  sampling::JsonConstraint::ToolFormat ToolFormat() const override {
    return sampling::JsonConstraint::ToolFormat::kQwen;
  }
  ReasoningTags ReasoningBudgetTags() const override {
    return {"<think>", "</think>", {"</think>", "<tool_call>"}};
  }
  std::vector<TextExecutionPlan> SupportedPlans() const override {
    std::vector<TextExecutionPlan> plans{{TextExecutionPlanKind::kSerial, 1}};
    for (std::size_t width = 2; width <= Session::kMaxBatch; ++width)
      plans.push_back({TextExecutionPlanKind::kBatched, width});
    return plans;
  }
  std::uint32_t VocabularySize() const override {
    return model_->config().vocab;
  }
  std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    return model_->tokenizer().Encode(text, false);
  }
  std::vector<TextRunnerToken> TokenizeControl(
      std::string_view text) const override {
    return model_->tokenizer().Encode(text);
  }
  std::optional<std::string> RenderPrompt(
      const ChatRequest& request) const override {
    return RenderChat(request.messages, Tools(request), request.reasoning);
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    return EncodeChat(model_->tokenizer(), request.messages, Tools(request),
                      request.reasoning);
  }
  std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    auto tokens = RenderAndTokenize(request);
    if (!tokens)
      return std::nullopt;
    // Keep the history before the mutable assistant opening. A client that
    // discards a cancelled reply appends a new user turn at this boundary.
    const auto history = EncodeChat(model_->tokenizer(), request.messages,
                                    Tools(request), request.reasoning, false);
    if (history.size() > tokens->size() ||
        !std::equal(history.begin(), history.end(), tokens->begin()))
      throw std::logic_error("LFM3 history is not a prompt prefix");
    return TextPreparedPrompt{std::move(*tokens), {}, history.size()};
  }
  ext::InitialOutputState InitialOutputState(
      const ChatRequest& r) const override {
    return ThinkingEnabled(r.reasoning) ? ext::InitialOutputState::kReasoning
                                        : ext::InitialOutputState::kContent;
  }
  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
  }
  std::unique_ptr<TextRunnerState> CreateState() const override {
    std::string error;
    auto s = std::make_unique<State>();
    s->session = model_->CreateSession(context_, &error);
    Require(bool(s->session), error);
    s->logits.resize(model_->config().vocab);
    return s;
  }
  TextPrefillStep Prefill(TextRunnerState& state,
                          std::span<const TextRunnerToken> prompt,
                          std::size_t offset,
                          std::size_t max_input_tokens) const override {
    auto& s = Get(state);
    if (offset != s.session->Position() || offset >= prompt.size())
      throw std::logic_error("LFM3 prefill frontier mismatch");
    auto count = std::min<std::size_t>(
        {Session::kPrefillChunk, max_input_tokens, prompt.size() - offset});
    if (!count)
      throw std::invalid_argument("LFM3 prefill budget must be positive");
    std::string error;
    Require(
        s.session->Forward(prompt.subspan(offset, count),
                           offset + count == prompt.size() ? std::span(s.logits)
                                                           : std::span<float>{},
                           &error),
        error);
    s.logits_ready = offset + count == prompt.size();
    return {count, s.logits_ready};
  }
  TextDecodeSelection SelectNext(
      TextRunnerState& state, sampling::SamplerState& sampler) const override {
    auto& s = Get(state);
    if (!s.logits_ready) {
      std::string error;
      Require(s.session->ReadLogits(s.logits, &error), error);
      s.logits_ready = true;
    }
    auto token = sampler.Sample(s.logits);
    if (state.stop_at_eos() && model_->tokenizer().IsStop(token))
      return {.stop = true, .token = token, .piece = {}, .logprobs = {}};
    return {.stop = false,
            .token = token,
            .piece = std::string(model_->tokenizer().Piece(token)),
            .logprobs = {}};
  }
  void Advance(TextRunnerState& state, TextRunnerToken token) const override {
    auto& s = Get(state);
    std::string error;
    Require(s.session->Forward(std::span(&token, 1), s.logits, &error), error);
    s.logits_ready = true;
  }
  void AdvanceBatch(
      std::span<const TextRunnerAdvance> advances) const override {
    if (advances.size() < 2) {
      TextModelRunner::AdvanceBatch(advances);
      return;
    }
    std::vector<Session*> sessions;
    std::vector<std::uint32_t> tokens;
    std::vector<std::span<float>> outputs;
    for (const auto& advance : advances) {
      auto& state = Get(advance.state.get());
      sessions.push_back(state.session.get());
      tokens.push_back(advance.token);
      outputs.emplace_back(state.logits);
    }
    std::string error;
    Require(Session::ForwardBatch(sessions, tokens, outputs, &error), error);
    for (const auto& advance : advances)
      Get(advance.state.get()).logits_ready = true;
  }
  std::size_t CheckpointPosition(const TextRunnerState& state) const override {
    return Get(state).session->Position();
  }

  std::size_t SnapshotPayloadBytes(
      const TextRunnerState& state) const override {
    return Get(state).session->SnapshotBytes();
  }
  std::unique_ptr<TextRunnerSnapshot> Snapshot(
      const TextRunnerState& state) const override {
    return std::make_unique<Lfm3Snapshot>(model_,
                                          Get(state).session->SaveSnapshot());
  }
  void RestoreOrFork(TextRunnerState& state,
                     const TextRunnerSnapshot& value) const override {
    const auto& snapshot = CheckedSnapshot(value);
    RestorePersistentSnapshot(state, snapshot.payload);
  }
  std::size_t PersistentSnapshotPayloadBytes(
      const TextRunnerSnapshot& value) const override {
    return CheckedSnapshot(value).payload.size();
  }
  std::size_t SerializePersistentSnapshot(
      const TextRunnerSnapshot& value,
      std::span<std::uint8_t> destination) const override {
    const auto& payload = CheckedSnapshot(value).payload;
    if (payload.size() != destination.size())
      throw std::invalid_argument("invalid LFM3 snapshot destination");
    std::copy(payload.begin(), payload.end(), destination.begin());
    return payload.size();
  }
  void StreamPersistentSnapshot(const TextRunnerSnapshot& value,
                                const SnapshotSink& sink) const override {
    sink(CheckedSnapshot(value).payload);
  }
  void RestorePersistentSnapshot(
      TextRunnerState& state,
      std::span<const std::uint8_t> payload) const override {
    auto& s = Get(state);
    std::string error;
    Require(s.session->RestoreSnapshot(payload, &error), error);
    s.logits_ready = s.session->Position() != 0;
    if (s.logits_ready)
      Require(s.session->ReadLogits(s.logits, &error), error);
  }

private:
  const Lfm3Snapshot& CheckedSnapshot(
      const TextRunnerSnapshot& snapshot) const {
    const auto* value = dynamic_cast<const Lfm3Snapshot*>(&snapshot);
    if (!value || value->owner != model_)
      throw std::invalid_argument("LFM3 snapshot belongs to a different model");
    return *value;
  }
  static std::span<const ChatTool> Tools(const ChatRequest& r) {
    return r.tool_choice == ChatRequest::ToolChoice::kNone
               ? std::span<const ChatTool>{}
               : r.tools;
  }

  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  std::optional<TextRunnerPersistenceDescriptor> persistence_;
};
}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateLfm3Runner(
    std::shared_ptr<Model> model, std::uint32_t context,
    std::string artifact_fingerprint) {
  return std::make_shared<Runner>(std::move(model), context,
                                  std::move(artifact_fingerprint));
}

std::shared_ptr<ext::TextModelRunner> CreateLfm3TextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  if (!options.vision_model_path.empty()) {
    if (error)
      *error = "LFM3 supports text input only";
    return nullptr;
  }
  if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled) {
    if (error)
      *error = "LFM3 does not support speculative decoding";
    return nullptr;
  }
  auto model = Model::Load(options.model_path, error);
  if (!model)
    return nullptr;
  std::uint32_t max_context = options.max_context;
  if (max_context == 0)
    max_context = model->config().context;
  if (max_context < 2 || max_context > model->config().context) {
    if (error)
      *error = "HTTP context exceeds the loaded LFM3 model context";
    return nullptr;
  }
  std::string model_fingerprint = options.disk_cache.model_artifact_fingerprint;
  if (!options.disk_cache.directory.empty() && model_fingerprint.empty()) {
    try {
      std::shared_ptr<const core::GgufReader> reader = options.reader;
      if (!reader) {
        auto opened = core::GgufReader::OpenFile(options.model_path, error);
        if (!opened)
          return nullptr;
        reader = std::move(opened);
      }
      model_fingerprint = core::GgufIdentityHex(*reader);
    } catch (const std::exception& e) {
      if (error)
        *error =
            std::string("LFM3 disk cache fingerprinting failed: ") + e.what();
      return nullptr;
    }
  }
  try {
    return CreateLfm3Runner(std::move(model), max_context,
                            std::move(model_fingerprint));
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::lfm3
