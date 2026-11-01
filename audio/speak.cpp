// gufo speak for LFM3-Audio, from the in-tree port's command (Gufo lfm3
// branch, src/cli/speak/speak.cpp at 500df164).
#include <chrono>
#include <fstream>
#include <gufo/ext/json.hpp>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "../probe.hpp"
#include "generation.hpp"
#include "runner.hpp"

namespace gufo::models::lfm3_audio {
namespace audio = ::gufo::models::lfm3_audio;

int RunLfm3Speak(const ext::SpeakOptions& o) {
  const bool chat = o.mode == "chat", text_mode = o.mode == "text";
  const bool tts = !chat && !text_mode;
  if ((tts && (o.text.empty() || !o.audio_path.empty() ||
               !o.tools_path.empty() || !o.system.empty())) ||
      (o.reasoning && (!text_mode || !o.tools_path.empty()))) {
    std::cerr << "Error: LFM3-Audio speaks text in tts mode (with --output, "
                 "no --audio, --tools or --system); --reasoning is for text "
                 "mode without --tools\n";
    return 2;
  }
  std::string error;
  const auto artifacts = lfm3::FindLfm3Audio(o.model, o.projector_path, &error);
  if (!artifacts) {
    std::cerr << "Error: "
              << (error.empty() ? "--model is not an LFM3-Audio GGUF" : error)
              << '\n';
    return 2;
  }
  const auto load_begin = std::chrono::steady_clock::now();
  auto model = audio::Model::Load(artifacts->backbone.string(),
                                  artifacts->mmproj.string(), &error);
  std::unique_ptr<audio::Generator> generator;
  if (model)
    generator =
        audio::Generator::Create(model, std::uint32_t(o.context), &error);
  if (!generator) {
    std::cerr << "Error loading LFM3-Audio: " << error << '\n';
    return 1;
  }
  const double load_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - load_begin)
                             .count();
  std::vector<audio::ChatMessage> messages(1);
  messages[0].role = "user";
  if (!o.audio_path.empty()) {
    std::ifstream in(o.audio_path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(in)), {});
    std::vector<float> pcm;
    std::uint32_t rate = 0;
    if (raw.empty() ||
        !audio::DecodeWav(
            {reinterpret_cast<const std::byte*>(raw.data()), raw.size()}, &pcm,
            &rate, &error)) {
      std::cerr << "Error reading " << o.audio_path << ": " << error << '\n';
      return 1;
    }
    messages[0].parts.push_back(
        {true, {}, audio::Resample(pcm, rate, audio::Model::kSampleRate)});
  }
  if (!o.text.empty())
    messages[0].parts.push_back({false, o.text, {}});
  std::vector<std::string> tools;
  if (!o.tools_path.empty()) {
    std::ifstream in(o.tools_path);
    std::string body((std::istreambuf_iterator<char>(in)), {});
    try {
      const auto list = json::parse(body);
      if (!list.is_array() || list.items().empty())
        throw std::invalid_argument("expected a non-empty array");
      for (const auto& tool : list.items())
        tools.push_back(audio::ToolDefinition(tool.dump()));
    } catch (const std::exception& e) {
      std::cerr << "Error reading " << o.tools_path << ": " << e.what() << '\n';
      return 2;
    }
  }
  // Model card prompts: interleaved replies are prefilled with
  // <|mixed_start|> and text replies with <think> or <think></think>; with
  // tools the model chooses between a call and an answer.
  std::string system = o.system, prefix;
  if (tts) {
    prefix = "<|audio_start|>";
    if (!audio::TtsSystemPrompt(o.language,
                                o.voice.empty() ? "female" : o.voice, &system,
                                &error)) {
      std::cerr << "Error: " << error << '\n';
      return 2;
    }
  } else if (system.empty() && !tools.empty())
    system = chat ? audio::kToolsInterleavedSystemPrompt
                  : audio::kToolsTextSystemPrompt;
  else if (system.empty() && chat)
    system = audio::kInterleavedSystemPrompt;
  if (tools.empty() && chat)
    prefix = "<|mixed_start|>";
  else if (tools.empty() && text_mode)
    prefix = o.reasoning ? "<think>" : "<think></think>";
  audio::GenerateOptions options;
  if (text_mode)
    options.suppress_mixed = options.suppress_audio_out = true;
  // Without reasoning the reply never opens or closes it.
  if (!tts && !o.reasoning)
    for (const auto* tag : {"<think>", "</think>"})
      if (const auto id = model->tokenizer().Encode(tag, true); id.size() == 1)
        options.suppress_tokens.push_back(id[0]);
      else if (chat && !tools.empty()) {
        // A spoken reply starts with a call or with interleaved speech.
        const auto call = model->tokenizer().Encode("<tool_call>", true);
        if (call.size() == 1)
          options.first_tokens = {call[0], model->tokens().mixed_start};
      }
  options.max_new_tokens = o.max_frames;
  options.seed = o.seed;
  options.audio.temperature = o.temperature;
  options.audio.top_k = o.top_k;
  audio::GenerateCallbacks callbacks;
  callbacks.cancelled = [&] { return o.cancelled && o.cancelled(); };
  // Reasoning streams to stderr, the answer to stdout; tool calls are
  // reported once the reply ends.
  audio::ReplyParser reply(o.reasoning);
  auto show = [&](const audio::ReplyParser::Delta& delta) {
    if (o.format != "text")
      return;
    if (!delta.reasoning.empty())
      std::cerr << delta.reasoning << std::flush;
    if (!delta.content.empty())
      std::cout << delta.content << std::flush;
  };
  if (!tts)
    callbacks.on_text = [&](std::string_view piece) {
      show(reply.Feed(piece));
      return true;
    };
  audio::GenerateResult result;
  const bool ok =
      generator->Generate(audio::BuildPrompt(system, messages, prefix, tools),
                          options, callbacks, &result, &error);
  if (!ok || result.cancelled) {
    std::cerr << "Error: " << (ok ? "cancelled" : error) << '\n';
    return result.cancelled ? 130 : 1;
  }
  std::vector<audio::ToolCall> calls;
  show(reply.Finish(&calls));
  json::Value tool_calls = json::Value::array();
  for (const auto& call : calls) {
    auto entry = json::Value::object();
    entry["name"] = call.name;
    auto arguments = json::Value::object();
    for (const auto& argument : call.arguments)
      arguments[argument.name] = argument.is_string
                                     ? json::Value(argument.value)
                                     : json::parse(argument.value);
    entry["arguments"] = std::move(arguments);
    tool_calls.push_back(std::move(entry));
  }
  if (!tts && o.format == "text") {
    std::cout << '\n';
    for (const auto& call : tool_calls.items())
      std::cout << "tool call: " << call.dump() << '\n';
  }
  if (!o.output_path.empty()) {
    const auto wav =
        audio::EncodeWav(result.samples, audio::Model::kOutputRate);
    std::ofstream out(o.output_path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(wav.data()),
              std::streamsize(wav.size()));
    if (!out) {
      std::cerr << "Error: cannot write " << o.output_path << '\n';
      return 1;
    }
  }
  const double seconds =
      double(result.samples.size()) / audio::Model::kOutputRate;
  if (o.format == "json") {
    json::Value report = json::Value::object();
    report["schema"] = "gufo.lfm3-audio.speech.v1";
    report["text"] = tts ? result.text : reply.content();
    if (!tts) {
      report["reasoning"] = reply.reasoning();
      report["tool_calls"] = std::move(tool_calls);
    }
    report["frames"] = result.frames;
    report["audio_seconds"] = seconds;
    report["prompt_tokens"] = result.prompt_tokens;
    json::Value t = json::Value::object();
    t["load_ms"] = load_ms;
    t["encode_ms"] = result.timings.encode_ms;
    t["prefill_ms"] = result.timings.prefill_ms;
    t["first_text_ms"] = result.timings.first_text_ms;
    t["first_audio_ms"] = result.timings.first_audio_ms;
    t["backbone_ms"] = result.timings.backbone_ms;
    t["depth_ms"] = result.timings.depth_ms;
    t["total_ms"] = result.timings.total_ms;
    t["real_time_factor"] = result.timings.total_ms > 0
                                ? seconds * 1000 / result.timings.total_ms
                                : 0.0;
    report["timings"] = std::move(t);
    std::cout << report.dump() << '\n';
  } else if (!o.output_path.empty())
    std::cerr << "wrote " << o.output_path << ": " << seconds << " s, "
              << result.frames << " frames in " << result.timings.total_ms
              << " ms (first audio " << result.timings.first_audio_ms
              << " ms)\n";
  return 0;
}

}  // namespace gufo::models::lfm3_audio
