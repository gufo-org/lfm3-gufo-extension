#include "runner.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <limits>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../probe.hpp"
#include "generation.hpp"
#include "realtime.hpp"

namespace gufo::models::lfm3_audio {
namespace {

std::string Lower(std::string text) {
  for (auto& c : text)
    c = char(std::tolower(static_cast<unsigned char>(c)));
  return text;
}

// English language names for the ASR system prompt.
std::string AsrLanguage(const std::string& language) {
  static const std::unordered_map<std::string, std::string> names{
      {"ar", "Arabic"},   {"da", "Danish"},     {"de", "German"},
      {"el", "Greek"},    {"en", "English"},    {"fi", "Finnish"},
      {"fr", "French"},   {"he", "Hebrew"},     {"it", "Italian"},
      {"ja", "Japanese"}, {"ko", "Korean"},     {"nl", "Dutch"},
      {"pl", "Polish"},   {"pt", "Portuguese"}, {"es", "Spanish"},
      {"sv", "Swedish"},  {"tr", "Turkish"},    {"zh", "Chinese"}};
  if (auto it = names.find(Lower(language)); it != names.end())
    return it->second;
  return language;
}

// Offline transcription windows: the model card qualifies 30 s contexts.
constexpr std::size_t kWindowSamples = 30 * Model::kSampleRate;

std::shared_ptr<Model> LoadModel(const std::string& model_root,
                                 const std::string& mmproj, std::size_t context,
                                 std::string* error) {
  std::string reason;
  const auto artifacts = lfm3::FindLfm3Audio(model_root, mmproj, &reason);
  if (!artifacts) {
    *error = reason.empty()
                 ? "no LFM3 GGUF with an LFM3-Audio mmproj at " + model_root
                 : reason;
    return nullptr;
  }
  if (context == 0 || context > std::numeric_limits<std::uint32_t>::max()) {
    *error = "invalid LFM3-Audio context capacity";
    return nullptr;
  }
  return Model::Load(artifacts->backbone.string(), artifacts->mmproj.string(),
                     error);
}

class Lfm3AsrRunner final : public ext::AsrRunner {
public:
  Lfm3AsrRunner(std::shared_ptr<Model> model,
                std::unique_ptr<Generator> generator)
      : model_(std::move(model)), generator_(std::move(generator)) {}

  std::string ModelId() const override { return "lfm3-audio-2b"; }
  bool SupportsLiveTranscription() const override { return true; }
  std::unique_ptr<ext::LiveTranscription> CreateLiveTranscription(
      std::uint32_t rate, std::string* error) override {
    return CreateLiveStream(model_, sessions_, rate, error);
  }
  bool SupportsVoiceConversation() const override { return true; }
  std::unique_ptr<ext::VoiceConversation> CreateVoiceConversation(
      std::uint32_t rate, std::string* error) override {
    return CreateConversation(model_, sessions_, rate, error);
  }

  bool Transcribe(const ext::TranscriptionRequest& request,
                  const ext::CancellationCheck& cancelled,
                  ext::TranscriptionResult* result,
                  std::string* error) override {
    std::string ignored;
    if (!error)
      error = &ignored;
    std::vector<float> pcm;
    if (!request.pcm16k.empty()) {
      pcm.assign(request.pcm16k.begin(), request.pcm16k.end());
    } else {
      std::uint32_t rate = 0;
      if (!DecodeWav(request.wav, &pcm, &rate, error))
        return false;
      pcm = Resample(pcm, rate, Model::kSampleRate);
    }
    if (pcm.empty()) {
      *error = "empty audio";
      return false;
    }
    *result = {};
    result->sample_rate = Model::kSampleRate;
    result->audio_samples = pcm.size();
    const std::string language =
        request.language ? AsrLanguage(*request.language) : std::string{};
    result->language = language;
    const std::lock_guard lock(mutex_);
    for (std::size_t at = 0; at < pcm.size(); at += kWindowSamples) {
      const auto count = std::min(kWindowSamples, pcm.size() - at);
      std::vector<ChatMessage> messages(1);
      messages[0].role = "user";
      messages[0].parts.push_back({true,
                                   {},
                                   {pcm.begin() + std::ptrdiff_t(at),
                                    pcm.begin() + std::ptrdiff_t(at + count)}});
      GenerateOptions options;
      options.max_new_tokens = request.max_new_tokens;
      options.suppress_mixed = options.suppress_audio_out = true;
      const std::string prefix =
          result->text.empty() ? std::string{} : result->text + " ";
      GenerateCallbacks callbacks;
      std::string window_text;
      bool stopped = false;
      callbacks.cancelled = [&] { return cancelled && cancelled(); };
      if (request.on_text)
        callbacks.on_text = [&](std::string_view piece) {
          window_text += piece;
          if (!request.on_text(prefix + window_text))
            stopped = true;
          return !stopped;
        };
      GenerateResult generated;
      if (!generator_->Generate(
              BuildPrompt(AsrSystemPrompt(language), messages), options,
              callbacks, &generated, error))
        return false;
      if (generated.cancelled || stopped) {
        *error = "transcription cancelled";
        return false;
      }
      if (!generated.text.empty())
        result->text = prefix + generated.text;
      result->generated_ids.insert(result->generated_ids.end(),
                                   generated.tokens.begin(),
                                   generated.tokens.end());
      result->audio_tokens += generated.audio_tokens;
      result->prompt_tokens += generated.prompt_tokens;
      result->mel_frames += MelFrames(count);
      ++result->chunks;
      result->timings.audio_encoder_ms += generated.timings.encode_ms;
      result->timings.text_decoder_ms +=
          generated.timings.total_ms - generated.timings.encode_ms;
      result->timings.total_ms += generated.timings.total_ms;
    }
    result->decoded = result->text;
    return true;
  }

private:
  std::shared_ptr<Model> model_;
  std::unique_ptr<Generator> generator_;
  std::mutex mutex_;
  std::shared_ptr<std::atomic<int>> sessions_ =
      std::make_shared<std::atomic<int>>(0);
};

// Synthesis requests join one batch on the speaker thread, so concurrent
// requests share backbone and depth-decoder steps.
class Lfm3TtsRunner final : public ext::TtsRunner {
public:
  Lfm3TtsRunner(std::shared_ptr<Model> model, std::uint32_t context)
      : model_(std::move(model)), context_(context) {}

  ~Lfm3TtsRunner() override {
    {
      const std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    ready_.notify_all();
    if (speaker_.joinable())
      speaker_.join();
  }

  bool Synthesize(const ext::SynthesisRequest& request,
                  const ext::CancellationCheck& cancelled,
                  ext::SynthesisResult* result, std::string* error) override {
    std::string ignored;
    if (!error)
      error = &ignored;
    if (!request.reference_audio.samples.empty()) {
      *error = "LFM3-Audio does not clone reference voices";
      return false;
    }
    if (request.text.empty()) {
      *error = "empty text";
      return false;
    }
    std::string system;
    if (!TtsSystemPrompt(request.language, request.speaker, &system, error))
      return false;
    std::vector<ChatMessage> messages(1);
    messages[0].role = "user";
    messages[0].parts.push_back({false, request.text, {}});
    auto job = std::make_shared<Job>();
    job->prompt = BuildPrompt(system, messages, "<|audio_start|>");
    job->options.max_new_tokens = request.max_new_tokens;
    job->options.seed = request.sampling.seed;
    if (!request.sampling.sample)
      job->options.audio.temperature = 0;
    job->callbacks.cancelled = [&] { return cancelled && cancelled(); };
    if (request.on_audio)
      job->callbacks.on_audio = request.on_audio;
    auto done = job->done.get_future();
    {
      const std::lock_guard lock(mutex_);
      if (!speaker_.joinable())
        speaker_ = std::thread([this] { Speak(); });
      jobs_.push_back(std::move(job));
    }
    ready_.notify_all();
    auto [failure, generated] = done.get();
    if (!failure.empty()) {
      *error = failure;
      return false;
    }
    if (generated.cancelled) {
      *error = "synthesis cancelled";
      return false;
    }
    *result = {};
    result->sample_rate = Model::kOutputRate;
    result->code_groups = AudioWeights::kCodebooks;
    result->codes = std::move(generated.codes);
    result->sample_count = generated.samples.size();
    if (!request.on_audio)
      result->samples = std::move(generated.samples);
    return true;
  }

  std::vector<std::string> Voices() const override {
    return {"female", "male"};
  }
  ext::TtsVariant Variant() const override {
    return ext::TtsVariant::kCustomVoice;
  }
  std::string ModelId() const override { return "lfm3-audio-2b"; }
  bool Concurrent() const override { return true; }

private:
  struct Job {
    std::vector<PromptPiece> prompt;
    GenerateOptions options;
    GenerateCallbacks callbacks;
    std::promise<std::pair<std::string, GenerateResult>> done;
  };

  void Speak() {
    std::string error;
    auto batch =
        SpeechBatch::Create(model_, context_, lfm3::Session::kMaxBatch, &error);
    const std::string unavailable = error;
    while (true) {
      std::vector<std::shared_ptr<Job>> admitted;
      {
        std::unique_lock lock(mutex_);
        ready_.wait(lock, [&] {
          return stopping_ || !jobs_.empty() || (batch && batch->active());
        });
        if (stopping_)
          break;
        while (!jobs_.empty() && (!batch || admitted.size() + batch->active() <
                                                batch->capacity())) {
          admitted.push_back(std::move(jobs_.front()));
          jobs_.pop_front();
        }
      }
      for (auto& job : admitted) {
        std::string failure = unavailable;
        if (batch &&
            batch->Add(
                job->prompt, job->options, job->callbacks,
                [job](bool ok, GenerateResult result, std::string message) {
                  job->done.set_value(
                      {ok ? std::string() : message, std::move(result)});
                },
                &failure))
          continue;
        job->done.set_value({failure.empty() ? "speech batch failed" : failure,
                             GenerateResult{}});
      }
      if (batch && batch->active())
        (void)batch->Step(&error);
      error.clear();
    }
    batch.reset();  // ends active streams with an error
    const std::lock_guard lock(mutex_);
    for (auto& job : jobs_)
      job->done.set_value({"LFM3-Audio runtime stopped", {}});
    jobs_.clear();
  }

  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::shared_ptr<Job>> jobs_;
  bool stopping_{false};
  std::thread speaker_;
};

}  // namespace

std::unique_ptr<ext::AsrRunner> CreateLfm3AsrRunner(
    const ext::AsrRunnerOptions& options, std::string* error) {
  auto model = LoadModel(options.model_root, options.projector_path,
                         options.maximum_context_tokens, error);
  if (!model)
    return nullptr;
  auto generator = Generator::Create(
      model, std::uint32_t(options.maximum_context_tokens), error);
  if (!generator)
    return nullptr;
  return std::make_unique<Lfm3AsrRunner>(std::move(model),
                                         std::move(generator));
}

std::unique_ptr<ext::TtsRunner> CreateLfm3TtsRunner(
    const ext::TtsRunnerOptions& options, std::string* error) {
  auto model = LoadModel(options.model_root, options.projector_path,
                         options.maximum_context_tokens, error);
  if (!model)
    return nullptr;
  return std::make_unique<Lfm3TtsRunner>(
      std::move(model), std::uint32_t(options.maximum_context_tokens));
}

}  // namespace gufo::models::lfm3_audio
