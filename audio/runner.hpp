#ifndef GUFO_MODELS_LFM3_AUDIO_RUNNER_HPP_
#define GUFO_MODELS_LFM3_AUDIO_RUNNER_HPP_

#include <cstddef>
#include <gufo/ext/asr.hpp>
#include <gufo/ext/tts.hpp>
#include <memory>
#include <string>

namespace gufo::models::lfm3_audio {

// model_root and projector_path resolve as lfm3::FindLfm3Audio does. Each
// runner loads its own model and serializes offline transcription;
// concurrent syntheses join one batch. The ASR runner also serves live
// transcription and spoken conversations, up to kRealtimeSessions at once.
std::unique_ptr<ext::AsrRunner> CreateLfm3AsrRunner(
    const ext::AsrRunnerOptions& options, std::string* error);
std::unique_ptr<ext::TtsRunner> CreateLfm3TtsRunner(
    const ext::TtsRunnerOptions& options, std::string* error);
// gufo speak: synthesis, voice chat and text replies.
int RunLfm3Speak(const ext::SpeakOptions& options);

}  // namespace gufo::models::lfm3_audio

#endif  // GUFO_MODELS_LFM3_AUDIO_RUNNER_HPP_
