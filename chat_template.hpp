#ifndef GUFO_MODELS_LFM3_CHAT_TEMPLATE_HPP_
#define GUFO_MODELS_LFM3_CHAT_TEMPLATE_HPP_

#include <cstdint>
#include <gufo/ext/reasoning.hpp>
#include <gufo/ext/text.hpp>
#include <span>
#include <string>
#include <vector>

namespace gufo::models::lfm3 {
class Tokenizer;
// LFM3 has an on/off thinking mode: positive API effort levels map to on.
bool ThinkingEnabled(const ReasoningOptions& reasoning);
std::string RenderChat(std::span<const ext::ChatMessage> messages,
                       std::span<const ext::ChatTool> tools = {},
                       const ReasoningOptions& reasoning = {},
                       bool add_generation_prompt = true);
std::vector<std::uint32_t> EncodeChat(
    const Tokenizer& tokenizer, std::span<const ext::ChatMessage> messages,
    std::span<const ext::ChatTool> tools = {},
    const ReasoningOptions& reasoning = {}, bool add_generation_prompt = true);
}  // namespace gufo::models::lfm3
#endif
