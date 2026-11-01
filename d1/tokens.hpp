#ifndef GUFO_MODELS_D1_TOKENS_HPP_
#define GUFO_MODELS_D1_TOKENS_HPP_
#include <string_view>
namespace gufo::models::d1 {
// LFM3 chat and vision framing (Qwen spellings), as the LFM3 chat template.
inline constexpr std::string_view kImStart = "<|im_start|>";
inline constexpr std::string_view kImEnd = "<|im_end|>";
inline constexpr std::string_view kVisionStart = "<|vision_start|>";
inline constexpr std::string_view kVisionEnd = "<|vision_end|>";
}  // namespace gufo::models::d1
#endif
