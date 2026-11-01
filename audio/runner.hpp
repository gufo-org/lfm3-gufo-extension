#ifndef GUFO_MODELS_LFM3_AUDIO_RUNNER_HPP_
#define GUFO_MODELS_LFM3_AUDIO_RUNNER_HPP_

#include <cstddef>
#include <gufo/ext/asr.hpp>
#include <gufo/ext/tts.hpp>
#include <memory>
#include <string>

namespace gufo::models::lfm3_audio {

// model_root resolves as lfm3::FindLfm3Audio does. Each runner loads its own
// model and serializes transcription; concurrent syntheses join one batch.
std::unique_ptr<ext::AsrRunner> CreateLfm3AsrRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error);
std::unique_ptr<ext::TtsRunner> CreateLfm3TtsRunner(
    const std::string& model_root, std::size_t maximum_context_tokens,
    std::string* error);

}  // namespace gufo::models::lfm3_audio

#endif  // GUFO_MODELS_LFM3_AUDIO_RUNNER_HPP_
