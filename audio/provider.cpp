#include <gufo/ext/asr.hpp>
#include <gufo/ext/generated/lfm3_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/tts.hpp>

#include "../probe.hpp"
#include "runner.hpp"

void lfm3_asr_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_lfm3::kProvider_lfm3_audio_asr);
  registry.AddAsrProvider({
      .id = "lfm3-audio-asr",
      .probe = &gufo::models::lfm3::ProbeLfm3Audio,
      .resolve = &gufo::models::lfm3::ResolveLfm3Audio,
      .create = &gufo::models::lfm3_audio::CreateLfm3AsrRunner,
  });
}

void lfm3_tts_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(
      gufo::ext::generated::extension_lfm3::kProvider_lfm3_audio_tts);
  registry.AddTtsProvider({
      .id = "lfm3-audio-tts",
      .probe = &gufo::models::lfm3::ProbeLfm3Audio,
      .resolve = &gufo::models::lfm3::ResolveLfm3Audio,
      .create = &gufo::models::lfm3_audio::CreateLfm3TtsRunner,
  });
}
