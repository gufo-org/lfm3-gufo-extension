#ifndef GUFO_MODELS_LFM3_AUDIO_REALTIME_HPP_
#define GUFO_MODELS_LFM3_AUDIO_REALTIME_HPP_

#include <atomic>
#include <cstdint>
#include <gufo/ext/asr.hpp>
#include <memory>
#include <string>

#include "generation.hpp"

namespace gufo::models::lfm3_audio {

// Realtime streams have their own contexts; at most kRealtimeSessions exist
// at once per sessions counter. rate is the input sample rate.
inline constexpr int kRealtimeSessions = 4;
std::unique_ptr<ext::LiveTranscription> CreateLiveStream(
    const std::shared_ptr<Model>& model,
    const std::shared_ptr<std::atomic<int>>& sessions, std::uint32_t rate,
    std::string* error);
std::unique_ptr<ext::VoiceConversation> CreateConversation(
    const std::shared_ptr<Model>& model,
    const std::shared_ptr<std::atomic<int>>& sessions, std::uint32_t rate,
    std::string* error);

}  // namespace gufo::models::lfm3_audio

#endif  // GUFO_MODELS_LFM3_AUDIO_REALTIME_HPP_
