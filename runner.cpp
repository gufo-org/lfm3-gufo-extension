#include "runner.hpp"

#include <algorithm>
#include <cstring>
#include <gufo/ext/component_cache_resources.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/host_budget.hpp>
#include <gufo/ext/sampling.hpp>
#include <gufo/ext/sha256.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "chat_template.hpp"
#include "continuation_adapter.hpp"
#include "vl/prompt.hpp"

namespace gufo::models::lfm3 {
namespace {
using namespace gufo::ext;

struct ImageContext final : TextPromptContext {
  lfm3_vl::Prompt prompt;
};

// LiquidAI's general recommendation (chat, tools, agents, coding), as in
// the official generation_config.json, for either thinking mode.
sampling::TextSamplingDefaults SamplingDefaults() {
  const sampling::TextSamplingValues values{
      .temperature = 0.1F, .top_k = 50, .repeat_penalty = 1.1F};
  return {values, values, values};
}

class State final : public TextRunnerState {
public:
  State(Session& session, cache::Adapter& adapter, cache::Slot& slot,
        std::size_t vocabulary)
      : session(&session),
        logits(vocabulary),
        adapter_(&adapter),
        slot_(&slot) {}
  Session* session;
  std::vector<float> logits;
  bool logits_ready{};
  std::shared_ptr<const ImageContext> input;
  std::vector<std::vector<float>> image_rows;
  void Invalidate() noexcept override {
    (void)adapter_->Invalidate(*slot_);
    logits_ready = false;
  }

private:
  cache::Adapter* adapter_;
  cache::Slot* slot_;
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

class Runner final : public HipTextModelRunner {
public:
  Runner(std::shared_ptr<Model> model, std::uint32_t context,
         std::string artifact_fingerprint,
         std::shared_ptr<lfm3_vl::Vision> vision)
      : model_(std::move(model)),
        vision_(std::move(vision)),
        context_(context) {
    if (!artifact_fingerprint.empty()) {
      if (artifact_fingerprint.size() != 64 ||
          artifact_fingerprint.find_first_not_of("0123456789abcdef") !=
              std::string::npos)
        throw std::invalid_argument("invalid LFM3 artifact fingerprint");
      auto identity =
          "schema=gufo-text-continuation-v2\nmodel_kind=lfm3\nartifact_id=" +
          std::string(core::kGgufIdentityScheme) + ':' + artifact_fingerprint +
          "\nstate_abi=lfm3-fp32-v2\npayload_layout=lfm3-fp32-compact-v1\n" +
          "tokenizer=lfm2-byte-bpe-v2-gguf-controls\nchat_template=lfm3-native-"
          "v2\n" +
          "numerics=fixed-point-three-digit-v1-wave32-attention-v1\ncontext_"
          "tokens=" +
          std::to_string(context_) +
          "\nposition_policy=absolute-ring-v1\nadapters=none\n";
      if (vision_)
        identity += "vision=lfm3-vl-image-v1:" + vision_->Identity() + "\n";
      component_identity_.assign(identity.begin(), identity.end());
    } else {
      // Without a file fingerprint the loaded model is the identity; durable
      // caches require the validated artifact identity above.
      std::ostringstream identity;
      identity << "schema=gufo-process-text-continuation-v1\nmodel_instance="
               << model_.get() << "\ncontext_tokens=" << context_ << '\n';
      const auto canonical = identity.str();
      component_identity_.assign(canonical.begin(), canonical.end());
    }
  }
  std::unique_ptr<ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    return hip::CreateHipComponentCacheResources(
        std::make_unique<ContinuationAdapter>(model_, context_,
                                              component_identity_),
        ledger,
        {[] { return HostSnapshotCeilingBytes(); }, [](std::string_view) {},
         [](std::string_view) {}});
  }
  std::unique_ptr<TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& lfm3 = dynamic_cast<ContinuationAdapter&>(adapter);
    return std::make_unique<State>(lfm3.GetSession(slot), adapter, slot,
                                   model_->config().vocab);
  }
  void ReconcileComponentState(TextRunnerState& state,
                               std::size_t) const override {
    // The frontier changed under this state; read its logits on demand.
    Get(state).logits_ready = false;
  }
  TextRunnerDescriptor Descriptor() const override {
    return {.model_id = vision_ ? "LFM3-VL-2B" : "LFM3-1.7B",
            .state_abi = "lfm3-fp32-v2",
            .max_context = context_,
            .capabilities = {.supports_vision = vision_ != nullptr,
                             .incremental_prefill = true,
                             .final_token_advance_required = true,
                             .incremental_text_is_exact = true,
                             .prefix_reuse = true,
                             .partial_prefill_retention = true},
            .sampling_defaults = SamplingDefaults(),
            // Each prefill pass is the engine's 512-row workspace, the
            // host's decode-active budget. A pass that short gets one layer
            // group from the host, so passes run unsplit beside decoders.
            .scheduling = TextSchedulingProfile{
                .pass_tokens = Session::kPrefillChunk,
                .max_layer_groups = 1,
                .max_yield_decode_width = Session::kMaxBatch}};
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
    if (vision_)
      return std::nullopt;
    return RenderChat(request.messages, Tools(request), request.reasoning);
  }
  std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ChatRequest& request) const override {
    if (vision_)
      return lfm3_vl::Prepare(model_->tokenizer(), request.messages,
                              Tools(request), request.reasoning, context_)
          .tokens;
    return EncodeChat(model_->tokenizer(), request.messages, Tools(request),
                      request.reasoning);
  }
  std::optional<TextPreparedPrompt> PreparePrompt(
      const ChatRequest& request) const override {
    if (vision_)
      return PrepareImagePrompt(request);
    auto tokens = RenderAndTokenize(request);
    if (!tokens)
      return std::nullopt;
    // Keep the history before the mutable assistant opening (a client that
    // discards a cancelled reply appends a new user turn there), or before
    // replaceable per-turn agent context.
    const auto history =
        EncodeStableHistory(model_->tokenizer(), request.messages,
                            Tools(request), request.reasoning);
    if (history.size() > tokens->size() ||
        !std::equal(history.begin(), history.end(), tokens->begin()))
      throw std::logic_error("LFM3 history is not a prompt prefix");
    return TextPreparedPrompt{std::move(*tokens), {}, history.size()};
  }
  std::optional<std::size_t> CountPromptTokens(
      const ChatRequest& request) const override {
    if (vision_)
      return lfm3_vl::Prepare(model_->tokenizer(), request.messages,
                              Tools(request), request.reasoning,
                              model_->config().context)
          .tokens.size();
    return TextModelRunner::CountPromptTokens(request);
  }
  void SetPromptContext(
      TextRunnerState& state,
      std::shared_ptr<const TextPromptContext> context) const override {
    auto& s = Get(state);
    s.input = std::dynamic_pointer_cast<const ImageContext>(context);
    if (context && (!vision_ || !s.input))
      throw std::invalid_argument("invalid LFM3-VL prompt context");
    s.image_rows.clear();
    if (s.input)
      s.image_rows.resize(s.input->prompt.images.size());
  }
  ext::InitialOutputState InitialOutputState(
      const ChatRequest& r) const override {
    return ThinkingEnabled(r.reasoning) ? ext::InitialOutputState::kReasoning
                                        : ext::InitialOutputState::kContent;
  }
  std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    return model_->tokenizer().Decode(tokens);
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
    bool visual = false;
    if (s.input)
      for (std::size_t i = 0; i < s.input->prompt.images.size(); ++i) {
        const auto& image = s.input->prompt.images[i];
        const auto rows =
            std::size_t(image.picture.height) * image.picture.width / 4;
        if (offset < image.offset) {
          count = std::min(count, image.offset - offset);
          break;
        }
        if (offset >= image.offset + rows)
          continue;
        if (s.image_rows[i].empty())
          s.image_rows[i] = vision_->Encode(image.picture);
        count = std::min(count, image.offset + rows - offset);
        const auto hidden = model_->config().hidden;
        Require(
            s.session->ForwardEmbeddings(
                std::span(s.image_rows[i])
                    .subspan((offset - image.offset) * hidden, count * hidden),
                false,
                offset + count == prompt.size() ? std::span(s.logits)
                                                : std::span<float>{},
                &error),
            error);
        visual = true;
        break;
      }
    if (!visual)
      Require(s.session->Forward(prompt.subspan(offset, count),
                                 offset + count == prompt.size()
                                     ? std::span(s.logits)
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
    auto logprobs =
        sampling::CollectLogprobs(s.logits, token, sampler.config().logprobs);
    if (state.stop_at_eos() && model_->tokenizer().IsStop(token))
      return {.stop = true,
              .token = token,
              .piece = {},
              .logprobs = std::move(logprobs)};
    return {.stop = false,
            .token = token,
            .piece = std::string(model_->tokenizer().Piece(token)),
            .logprobs = std::move(logprobs)};
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
      sessions.push_back(state.session);
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

private:
  // Images extend the cache identity at their first token, with the
  // projector identity, placement, grid and processed pixels.
  std::optional<TextPreparedPrompt> PrepareImagePrompt(
      const ChatRequest& request) const {
    auto context = std::make_shared<ImageContext>();
    context->prompt =
        lfm3_vl::Prepare(model_->tokenizer(), request.messages, Tools(request),
                         request.reasoning, context_);
    crypto::Sha256Hasher hash;
    const auto& identity = vision_->Identity();
    hash.Update({reinterpret_cast<const std::uint8_t*>(identity.data()),
                 identity.size()});
    for (const auto& image : context->prompt.images) {
      context->cache_prefixes.push_back(
          {image.offset, context->cache_identity});
      const std::uint64_t geometry[] = {image.offset,
                                        std::uint64_t(image.picture.height),
                                        std::uint64_t(image.picture.width)};
      hash.Update(
          {reinterpret_cast<const std::uint8_t*>(geometry), sizeof(geometry)});
      const auto& patches = image.picture.patches;
      hash.Update({reinterpret_cast<const std::uint8_t*>(patches.data()),
                   patches.size() * sizeof(float)});
      const auto digest = hash.Digest();
      context->cache_identity.assign(digest.begin(), digest.end());
    }
    const auto stable = context->prompt.stable_prefix;
    auto tokens = context->prompt.tokens;
    return TextPreparedPrompt{std::move(tokens), std::move(context), stable};
  }
  static std::span<const ChatTool> Tools(const ChatRequest& r) {
    return r.tool_choice == ChatRequest::ToolChoice::kNone
               ? std::span<const ChatTool>{}
               : r.tools;
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<lfm3_vl::Vision> vision_;
  std::uint32_t context_;
  cache::Identity component_identity_;
};
}  // namespace

std::shared_ptr<ext::TextModelRunner> CreateLfm3Runner(
    std::shared_ptr<Model> model, std::uint32_t context,
    std::string artifact_fingerprint, std::shared_ptr<lfm3_vl::Vision> vision) {
  return std::make_shared<Runner>(std::move(model), context,
                                  std::move(artifact_fingerprint),
                                  std::move(vision));
}

std::shared_ptr<ext::TextModelRunner> CreateLfm3TextRunner(
    const ext::TextRunnerOptions& options, std::string* error) {
  if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled) {
    if (error)
      *error = "LFM3 does not support speculative decoding";
    return nullptr;
  }
  auto model = Model::Load(options.model_path, error);
  if (!model)
    return nullptr;
  std::shared_ptr<lfm3_vl::Vision> vision;
  try {
    vision = lfm3_vl::LoadMatchingVision(options.model_path,
                                         options.vision_model_path);
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
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
                            std::move(model_fingerprint), std::move(vision));
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::lfm3
