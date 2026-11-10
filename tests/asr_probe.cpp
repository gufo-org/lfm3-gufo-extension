// Transcription latency and the effect of the shared prompt pass:
//   lfm3_asr_probe BACKBONE MMPROJ WAV...
// For each clip (its first 30 s), times greedy Generator transcription
// (first text and total, median of five after a warmup) and compares the
// prompt fed in one shared pass (production) with the text, audio and text
// passes it replaced, teacher-forced over the shared pass's greedy
// transcript: KL(split || shared) per step, argmax agreement and the
// largest logit difference.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "../audio/generation.hpp"
namespace audio = gufo::models::lfm3_audio;
namespace lfm3 = gufo::models::lfm3;
namespace {
void Check(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}
std::vector<double> LogSoftmax(std::span<const float> logits) {
  const double top = *std::max_element(logits.begin(), logits.end());
  double total = 0;
  for (auto v : logits)
    total += std::exp(double(v) - top);
  const double norm = top + std::log(total);
  std::vector<double> out(logits.size());
  for (std::size_t i = 0; i < logits.size(); ++i)
    out[i] = double(logits[i]) - norm;
  return out;
}
double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc < 4) {
      std::cerr << "usage: lfm3_asr_probe BACKBONE MMPROJ WAV...\n";
      return 2;
    }
    std::string error;
    auto model = audio::Model::Load(argv[1], argv[2], &error);
    Check(bool(model), error);
    const auto& tok = model->tokenizer();
    const auto& ids = model->tokens();
    const auto vocab = model->backbone().config().vocab;
    const auto H = model->backbone().config().hidden;
    auto generator = audio::Generator::Create(model, 4096, &error);
    Check(bool(generator), error);
    auto shared = model->backbone().CreateSession(4096, &error);
    auto split = model->backbone().CreateSession(4096, &error);
    Check(shared && split, error);
    const auto system = audio::AsrSystemPrompt("English");
    const auto prefix =
        tok.Encode("<|startoftext|><|im_start|>system\n" + system +
                       "<|im_end|>\n<|im_start|>user\n",
                   true);
    const auto suffix = tok.Encode("<|im_end|>\n<|im_start|>assistant\n", true);
    int failures = 0;
    for (int arg = 3; arg < argc; ++arg) {
      std::ifstream in(argv[arg], std::ios::binary);
      std::vector<char> raw((std::istreambuf_iterator<char>(in)), {});
      std::vector<float> pcm;
      std::uint32_t rate = 0;
      Check(
          audio::DecodeWav(std::as_bytes(std::span(raw)), &pcm, &rate, &error),
          error);
      pcm = audio::Resample(pcm, rate, audio::Model::kSampleRate);
      pcm.resize(std::min<std::size_t>(pcm.size(), 30 * 16000));
      // Production transcription and its timings.
      std::vector<audio::ChatMessage> messages(1);
      messages[0].role = "user";
      messages[0].parts.push_back({true, {}, pcm});
      const auto prompt = audio::BuildPrompt(system, messages);
      audio::GenerateOptions options;
      options.max_new_tokens = 256;
      options.suppress_mixed = options.suppress_audio_out = true;
      audio::GenerateResult result;
      std::vector<double> first, total;
      for (int run = 0; run < 6; ++run) {
        Check(generator->Generate(prompt, options, {}, &result, &error), error);
        if (run) {
          first.push_back(result.timings.first_text_ms);
          total.push_back(result.timings.total_ms);
        }
      }
      // Shared pass, greedy, as Generator decodes.
      std::vector<float> rows;
      Check(model->Encode(pcm, &rows, &error), error);
      std::uint64_t hash = 1469598103934665603ULL;  // FNV-1a of the rows
      for (auto byte : std::as_bytes(std::span(rows)))
        hash = (hash ^ std::uint64_t(byte)) * 1099511628211ULL;
      std::vector<float> logits(vocab);
      Check(shared->Reset(&error), error);
      const lfm3::Session::Input inputs[] = {
          {prefix, {}}, {{}, rows}, {suffix, {}}};
      Check(shared->ForwardInputs(inputs, logits, &error), error);
      std::vector<std::uint32_t> tokens;
      std::vector<std::vector<double>> reference;
      const std::uint32_t suppressed[] = {ids.audio, ids.audio_out,
                                          ids.mixed_start, ids.audio_start};
      for (int step = 0; step < 256; ++step) {
        for (auto id : suppressed)
          logits[id] = -INFINITY;
        reference.push_back(LogSoftmax(logits));
        const auto token = std::uint32_t(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        tokens.push_back(token);
        if (token == ids.im_end || token == ids.text_end || tok.IsStop(token))
          break;
        Check(shared->Forward({&token, 1}, logits, &error), error);
      }
      const bool same_tokens = tokens == result.tokens;
      // Split passes, teacher-forced over the same tokens.
      Check(split->Reset(&error), error);
      Check(split->Forward(prefix, {}, &error), error);
      Check(split->ForwardEmbeddings(rows, false, {}, &error), error);
      Check(split->Forward(suffix, {}, &error), error);
      double kl_max = 0, kl_sum = 0, delta = 0;
      std::size_t agree = 0;
      for (std::size_t step = 0; step < tokens.size(); ++step) {
        Check(split->ReadLogits(logits, &error), error);
        for (auto id : suppressed)
          logits[id] = -INFINITY;
        const auto b = LogSoftmax(logits);
        const auto& a = reference[step];
        double kl = 0;
        for (std::size_t i = 0; i < vocab; ++i)
          if (std::isfinite(b[i])) {
            kl += std::exp(b[i]) * (b[i] - a[i]);
            delta = std::max(delta, std::abs(b[i] - a[i]));
          }
        kl_max = std::max(kl_max, kl);
        kl_sum += kl;
        agree += std::max_element(b.begin(), b.end()) - b.begin() ==
                 std::ptrdiff_t(tokens[step]);
        if (step + 1 < tokens.size())
          Check(split->Forward({&tokens[step], 1}, {}, &error), error);
      }
      const bool ok =
          same_tokens && agree == tokens.size() && kl_max < 1e-3 && delta < 0.1;
      failures += !ok;
      std::printf(
          "%s %s rows %zu steps %zu first %.2f ms total %.2f ms kl max %.3g "
          "mean %.3g argmax %zu/%zu dlogp %.3g generator %s encoder %016llx\n  "
          "%s\n",
          ok ? "ok  " : "FAIL", argv[arg], rows.size() / H, tokens.size(),
          Median(first), Median(total), kl_max, kl_sum / tokens.size(), agree,
          tokens.size(), delta, same_tokens ? "same" : "DIFFERENT",
          static_cast<unsigned long long>(hash), result.text.c_str());
    }
    return failures ? 1 : 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
