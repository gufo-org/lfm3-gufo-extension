#ifndef GUFO_MODELS_LFM3_VL_PROMPT_HPP_
#define GUFO_MODELS_LFM3_VL_PROMPT_HPP_
#include "../chat_template.hpp"
#include "../tokenizer.hpp"
#include "vision.hpp"
namespace gufo::models::lfm3_vl {
inline constexpr std::string_view kImageToken = "<|image|>";
struct InputImage {
  std::size_t offset{};
  Picture picture;
};
struct Prompt {
  std::vector<std::uint32_t> tokens;
  std::vector<InputImage> images;
  std::size_t stable_prefix{};
};
Prompt Prepare(const lfm3::Tokenizer& tokenizer,
               std::span<const ext::ChatMessage> messages,
               std::span<const ext::ChatTool> tools,
               const ReasoningOptions& reasoning, std::uint32_t context);
}  // namespace gufo::models::lfm3_vl
#endif
