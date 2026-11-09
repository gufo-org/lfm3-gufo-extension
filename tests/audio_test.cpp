// Host-side LFM3-Audio helpers: WAV round trip, resampling, prompt framing
// and task prompts. Expected strings come from the model card and the
// official chat template, not from production constants.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

#include "../audio/generation.hpp"
namespace audio = gufo::models::lfm3_audio;
namespace {
int failures = 0;
void Expect(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL %s\n", what.c_str());
    ++failures;
  }
}
void Push16(std::vector<std::byte>& out, std::uint16_t v) {
  out.push_back(std::byte(v & 255));
  out.push_back(std::byte(v >> 8));
}
void Push32(std::vector<std::byte>& out, std::uint32_t v) {
  Push16(out, std::uint16_t(v & 0xFFFF));
  Push16(out, std::uint16_t(v >> 16));
}
std::vector<std::byte> Bytes(const char* s) {
  std::vector<std::byte> out;
  for (; *s; ++s)
    out.push_back(std::byte(*s));
  return out;
}
// A RIFF/WAVE with one fmt chunk and one data chunk. A zero block_align
// derives the canonical channels * bytes per sample.
std::vector<std::byte> MakeWav(std::uint16_t format, std::uint16_t channels,
                               std::uint32_t rate, std::uint16_t bits,
                               std::vector<std::byte> data,
                               std::uint16_t block_align = 0,
                               std::uint32_t fmt_size = 16,
                               const std::vector<std::byte>& fmt_ext = {}) {
  if (!block_align)
    block_align = std::uint16_t(channels * (bits / 8));
  auto chunk = [&](const char* id, const std::vector<std::byte>& body) {
    auto out = Bytes(id);
    Push32(out, std::uint32_t(body.size()));
    out.insert(out.end(), body.begin(), body.end());
    if (body.size() & 1)
      out.push_back(std::byte(0));
    return out;
  };
  std::vector<std::byte> fmt;
  Push16(fmt, format);
  Push16(fmt, channels);
  Push32(fmt, rate);
  Push32(fmt, rate * block_align);
  Push16(fmt, block_align);
  Push16(fmt, bits);
  fmt.insert(fmt.end(), fmt_ext.begin(), fmt_ext.end());
  fmt.resize(fmt_size);  // Truncates, or zero-pads an extension area.
  auto body = Bytes("WAVE");
  for (const auto& c : {chunk("fmt ", fmt), chunk("data", data)})
    body.insert(body.end(), c.begin(), c.end());
  auto out = Bytes("RIFF");
  Push32(out, std::uint32_t(body.size()));
  out.insert(out.end(), body.begin(), body.end());
  return out;
}
std::vector<std::byte> F32(std::initializer_list<float> values) {
  std::vector<std::byte> out;
  for (const float v : values) {
    std::uint32_t bits{};
    std::memcpy(&bits, &v, 4);
    Push32(out, bits);
  }
  return out;
}
// Decodes and expects failure with this exact message.
void ExpectRejected(const std::vector<std::byte>& wav, const char* message,
                    const std::string& what) {
  std::vector<float> samples;
  std::uint32_t rate = 0;
  std::string error;
  Expect(!audio::DecodeWav(wav, &samples, &rate, &error), what + ": accepted");
  Expect(error == message, what + ": " + error);
}
std::string Render(const std::vector<audio::PromptPiece>& pieces) {
  std::string out;
  for (const auto& p : pieces)
    out += p.kind == audio::PromptPiece::Kind::kAudio ? "[audio]" : p.text;
  return out;
}
}  // namespace
int main() {
  // WAV: PCM16 round trip within one quantization step.
  std::vector<float> tone(2400);
  for (std::size_t i = 0; i < tone.size(); ++i)
    tone[i] = 0.5F * std::sin(2 * std::numbers::pi * 440.0 * i / 24000);
  const auto wav = audio::EncodeWav(tone, 24000);
  std::vector<float> decoded;
  std::uint32_t rate = 0;
  std::string error;
  Expect(audio::DecodeWav(wav, &decoded, &rate, &error), "decode: " + error);
  Expect(rate == 24000 && decoded.size() == tone.size(), "wav shape");
  double worst = 0;
  for (std::size_t i = 0; i < tone.size() && i < decoded.size(); ++i)
    worst = std::max(worst, std::abs(double(decoded[i]) - tone[i]));
  Expect(worst <= 1.0 / 32767, "wav quantization error");
  Expect(!audio::DecodeWav(std::span(wav).first(20), &decoded, &rate, &error),
         "truncated wav rejected");

  // Shared-decoder edges, decided when switching off the local parser:
  // WAVE_FORMAT_EXTENSIBLE is rejected even with a PCM subformat; a
  // truncated fmt chunk is an error instead of being ignored; data that
  // is not a whole number of frames is rejected; float samples clamp to
  // [-1, 1] and reject non-finite values; the limits are 8 channels,
  // 192 kHz, 300 seconds and PCM up to 32 bits.
  {
    const std::vector<std::byte> pcm16{std::byte(0x00), std::byte(0x20),
                                       std::byte(0x00), std::byte(0xC0)};
    // PCM16 subformat of WAVE_FORMAT_EXTENSIBLE (0xFFFE): cbSize 22,
    // valid bits 16, channel mask 0, KSDATAFORMAT_SUBTYPE_PCM.
    std::vector<std::byte> pcm_subformat;
    Push16(pcm_subformat, 22);
    Push16(pcm_subformat, 16);
    Push32(pcm_subformat, 0);
    Push32(pcm_subformat, 0x00000001);  // GUID Data1
    Push32(pcm_subformat, 0x00100000);  // Data2, Data3
    for (const std::uint8_t tail :
         {0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71})
      pcm_subformat.push_back(std::byte(tail));
    ExpectRejected(MakeWav(0xFFFE, 1, 16000, 16, pcm16, 2, 40, pcm_subformat),
                   "WAV format is unsupported", "extensible wav");
    ExpectRejected(MakeWav(1, 1, 16000, 16, std::vector<std::byte>(8), 2, 8),
                   "WAV fmt chunk is invalid", "short fmt chunk");
    ExpectRejected(
        MakeWav(1, 1, 16000, 16, {std::byte(0), std::byte(0x20), std::byte(0)}),
        "WAV must be PCM16/24/32 or float32 interleaved", "partial frame");
    ExpectRejected(MakeWav(1, 1, 16000, 16, {}), "WAV format is unsupported",
                   "empty data");
    ExpectRejected(MakeWav(1, 9, 16000, 16, std::vector<std::byte>(36)),
                   "WAV format is unsupported", "9 channels");
    ExpectRejected(MakeWav(1, 1, 192001, 16, pcm16),
                   "WAV format is unsupported", "sample rate over 192 kHz");
    // 301 seconds at 8 Hz stays small while exceeding the duration limit.
    ExpectRejected(MakeWav(1, 1, 8, 16, std::vector<std::byte>(301 * 8 * 2)),
                   "audio must contain at most 300 seconds", "too long");
    ExpectRejected(MakeWav(1, 1, 16000, 64, std::vector<std::byte>(32)),
                   "WAV must be PCM16/24/32 or float32 interleaved", "pcm64");
    ExpectRejected(MakeWav(1, 2, 16000, 16, pcm16, 3),
                   "WAV must be PCM16/24/32 or float32 interleaved",
                   "block align mismatch");
    // Float32: out-of-range values clamp, non-finite samples reject.
    const auto clamped_wav = MakeWav(3, 1, 16000, 32, F32({2.0F, -3.0F, 0.5F}));
    std::vector<float> clamped;
    Expect(audio::DecodeWav(clamped_wav, &clamped, &rate, &error) &&
               clamped.size() == 3 && clamped[0] == 1.0F &&
               clamped[1] == -1.0F && clamped[2] == 0.5F,
           "float clamp: " + error);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ExpectRejected(MakeWav(3, 1, 16000, 32, F32({0.25F, nan})),
                   "WAV contains non-finite samples", "float nan");
    // Stereo downmix: the mean of the channels.
    const auto stereo = MakeWav(1, 2, 16000, 16, [] {
      std::vector<std::byte> d;
      for (const std::uint16_t v : {std::uint16_t(8192), std::uint16_t(24576)})
        Push16(d, v);
      return d;
    }());
    std::vector<float> mono;
    Expect(audio::DecodeWav(stereo, &mono, &rate, &error) && mono.size() == 1 &&
               std::abs(mono[0] - 0.5F) < 1e-4F,
           "stereo downmix: " + error);
    Expect(rate == 16000, "stereo rate");
  }

  // Resampling keeps an in-band tone: 1 kHz at 24 kHz to 16 kHz.
  std::vector<float> source(24000);
  for (std::size_t i = 0; i < source.size(); ++i)
    source[i] = std::sin(2 * std::numbers::pi * 1000.0 * i / 24000);
  const auto resampled = audio::Resample(source, 24000, 16000);
  Expect(resampled.size() == 16000, "resampled length");
  double error_sum = 0;
  for (std::size_t i = 2000; i < 14000; ++i)
    error_sum += std::abs(resampled[i] -
                          std::sin(2 * std::numbers::pi * 1000.0 * i / 16000));
  Expect(error_sum / 12000 < 1e-3, "resampled tone error");

  // Mel framing of the official extractor: right padding to whole hops.
  // One second: 16080 padded samples, 99 frames (the reference reports 99).
  Expect(audio::MelFrames(16000) == 99 && audio::MelFrames(1) == 1 &&
             audio::MelFrames(560) == 2 && audio::EncoderTokens(99) == 13,
         "mel frame counts");

  // Task prompts from the model card.
  Expect(audio::AsrSystemPrompt("English") == "Perform ASR in English.",
         "asr prompt");
  Expect(audio::AsrSystemPrompt() == "Perform ASR.", "asr auto prompt");
  std::string tts;
  Expect(audio::TtsSystemPrompt("en", "female", &tts, &error) &&
             tts == "Perform TTS in American English using a female voice.",
         "tts en prompt");
  Expect(audio::TtsSystemPrompt("en-gb", "MALE", &tts, &error) &&
             tts == "Perform TTS in British English using a male voice.",
         "tts en-gb prompt");
  Expect(audio::TtsSystemPrompt("French", "", &tts, &error) &&
             tts == "Perform TTS in French using a female voice.",
         "tts language name");
  Expect(!audio::TtsSystemPrompt("klingon", "female", &tts, &error),
         "unsupported language rejected");
  Expect(!audio::TtsSystemPrompt("en", "child", &tts, &error),
         "unsupported voice rejected");

  // Chat framing of the official template with an audio placeholder.
  std::vector<audio::ChatMessage> messages(1);
  messages[0].role = "user";
  messages[0].parts.push_back({true, {}, std::vector<float>(160)});
  messages[0].parts.push_back({false, "What is happening?", {}});
  const auto pieces =
      audio::BuildPrompt("Perform ASR.", messages, "<|mixed_start|>");
  Expect(Render(pieces) ==
             "<|startoftext|><|im_start|>system\nPerform ASR.<|im_end|>\n"
             "<|im_start|>user\n[audio]What is happening?<|im_end|>\n"
             "<|im_start|>assistant\n<|mixed_start|>",
         "chat framing: " + Render(pieces));
  // Client text stays literal: control tokens in it are not template.
  Expect(pieces[1].kind == audio::PromptPiece::Kind::kLiteral &&
             pieces[3].kind == audio::PromptPiece::Kind::kAudio &&
             pieces[4].kind == audio::PromptPiece::Kind::kLiteral,
         "piece kinds");

  // Streaming resampling: arbitrary pieces give the one-call samples.
  {
    std::vector<float> noise(7919);
    std::uint32_t state = 1;
    for (auto& v : noise) {
      state = state * 1664525U + 1013904223U;
      v = float(state >> 8) / float(1U << 24) - 0.5F;
    }
    for (auto [from, to] :
         {std::pair{24000U, 16000U}, {44100U, 16000U}, {16000U, 24000U}}) {
      const auto whole = audio::Resample(noise, from, to);
      audio::Resampler stream(from, to);
      std::vector<float> pieces;
      std::size_t at = 0, step = 1;
      while (at < noise.size()) {
        const auto n = std::min(step, noise.size() - at);
        auto out = stream.Push(std::span(noise).subspan(at, n));
        pieces.insert(pieces.end(), out.begin(), out.end());
        at += n;
        step = step * 3 % 487 + 1;
      }
      auto out = stream.Push({}, true);
      pieces.insert(pieces.end(), out.begin(), out.end());
      Expect(pieces == whole, "streaming resampler " + std::to_string(from) +
                                  " -> " + std::to_string(to));
    }
  }

  // Tools in the system turn and tool results, as the official template
  // renders them (tojson spelling, one definition per line).
  {
    const std::string tool =
        R"({"type":"function","function":{"name":"set_volume","description":"Set the device speaker volume.","parameters":{"type":"object","properties":{"volume":{"type":"integer"}},"required":["volume"]},"defer_loading":false}})";
    const auto definition = audio::ToolDefinition(tool);
    Expect(
        definition ==
            R"({"name": "set_volume", "description": "Set the device speaker volume.", "parameters": {"type": "object", "properties": {"volume": {"type": "integer"}}, "required": ["volume"]}})",
        "tool definition: " + definition);
    const std::vector<std::string> tools{definition};
    std::vector<audio::ChatMessage> turn(2);
    turn[0].role = "user";
    turn[0].parts.push_back({false, "Turn it down.", {}});
    turn[1].role = "tool";
    turn[1].parts.push_back({false, R"([{"volume": 30}])", {}});
    const auto framed = Render(audio::BuildPrompt(
        audio::kToolsInterleavedSystemPrompt, turn, {}, tools));
    Expect(framed ==
               "<|startoftext|><|im_start|>system\nRespond with function "
               "calls, and interleaved text and audio.\n\nList of tools:"
               "\n\n<tools>\n" +
                   definition +
                   "\n</tools><|im_end|>\n<|im_start|>user\nTurn it "
                   "down.<|im_end|>\n<|im_start|>tool\n<tool_response>\n"
                   "[{\"volume\": 30}]\n</tool_response><|im_end|>\n"
                   "<|im_start|>assistant\n",
           "tool framing: " + framed);
  }

  // Reply parsing: reasoning, content and calls, with tags split anywhere.
  {
    const std::string reply =
        "Let me think.</think>\n\nSure.<tool_call>\n<function=set_volume>\n"
        "<parameter=volume>\n30\n</parameter>\n<parameter=room>\nkitchen\n"
        "</parameter>\n</function>\n</tool_call>\n\n<tool_call>\n"
        "<function=mute>\n</function>\n</tool_call>";
    for (std::size_t step : {1U, 2U, 5U, 1000U}) {
      audio::ReplyParser parser(true);
      std::string reasoning, content;
      for (std::size_t at = 0; at < reply.size(); at += step) {
        auto delta = parser.Feed(reply.substr(at, step));
        reasoning += delta.reasoning;
        content += delta.content;
      }
      std::vector<audio::ToolCall> calls;
      auto delta = parser.Finish(&calls);
      reasoning += delta.reasoning;
      content += delta.content;
      Expect(reasoning == "Let me think." && content == "Sure.",
             "reply split at " + std::to_string(step) + ": [" + reasoning +
                 "] [" + content + "]");
      Expect(calls.size() == 2 && calls[0].name == "set_volume" &&
                 calls[0].arguments.size() == 2 &&
                 calls[0].arguments[0].value == "30" &&
                 !calls[0].arguments[0].is_string &&
                 calls[0].arguments[1].value == "kitchen" &&
                 calls[0].arguments[1].is_string && calls[1].name == "mute" &&
                 calls[1].arguments.empty(),
             "tool calls at " + std::to_string(step));
    }
    // A reply may open its own reasoning; plain replies pass through.
    audio::ReplyParser opened;
    auto a = opened.Feed("<think>hm</think>Yes");
    std::vector<audio::ToolCall> calls;
    auto b = opened.Finish(&calls);
    Expect(a.reasoning + b.reasoning == "hm" &&
               a.content + b.content == "Yes" && calls.empty(),
           "self-opened reasoning");
    audio::ReplyParser plain;
    auto c = plain.Feed("Hello <tool");
    auto d = plain.Finish(&calls);
    Expect(c.content == "Hello " && d.content == "<tool" && calls.empty(),
           "partial tag released at the end");
  }
  std::printf(failures ? "FAILED %d\n" : "PASSED\n", failures);
  return failures ? 1 : 0;
}
