#include "chat_template.hpp"

#include <algorithm>
#include <gufo/ext/json.hpp>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "tokenizer.hpp"

namespace gufo::models::lfm3 {
namespace {
using ext::ChatRole;

constexpr std::string_view kImStart = "<|im_start|>";
constexpr std::string_view kImEnd = "<|im_end|>";
constexpr std::string_view kCallOpen = "<tool_call>";
constexpr std::string_view kCallClose = "</tool_call>";
constexpr std::string_view kFunctionOpen = "<function=";
constexpr std::string_view kFunctionClose = "</function>";
constexpr std::string_view kParameterOpen = "<parameter=";
constexpr std::string_view kParameterClose = "</parameter>";

constexpr std::string_view ToString(ChatRole role) noexcept {
  switch (role) {
    case ChatRole::kSystem:
      return "system";
    case ChatRole::kDeveloper:
      return "developer";
    case ChatRole::kUser:
      return "user";
    case ChatRole::kAssistant:
      return "assistant";
    case ChatRole::kTool:
      return "tool";
  }
  return "user";
}

struct Rendered {
  std::string text{"<|startoftext|>"};
  std::vector<std::pair<std::size_t, std::size_t>> literals;
  void Literal(std::string_view value) {
    if (!value.empty())
      literals.emplace_back(text.size(), value.size());
    text += value;
  }
};
std::string_view Trim(std::string_view s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == s.npos)
    return {};
  return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
std::string ToolDefinition(const ext::ChatTool& tool) {
  auto fn = json::Value::object();
  if (!tool.definition_json.empty()) {
    auto def = json::parse(tool.definition_json);
    fn = def.find("function") ? *def.find("function") : std::move(def);
  } else {
    fn["name"] = tool.name;
    fn["description"] = tool.description;
    fn["parameters"] =
        json::parse(tool.parameters_json.empty() ? "{}" : tool.parameters_json);
  }
  auto filtered = json::Value::object();
  for (const auto& [key, value] : fn.members())
    if (key != "defer_loading")
      filtered[key] = value;
  return filtered.tojson();
}
void Calls(Rendered& out, std::span<const ext::ChatMessage::ToolCall> calls) {
  for (std::size_t i = 0; i < calls.size(); ++i) {
    if (i)
      out.text += "\n\n";
    const auto& call = calls[i];
    out.text += kCallOpen;
    out.text += '\n';
    out.text += kFunctionOpen;
    out.text += call.name + ">\n";
    for (const auto& arg : call.arguments) {
      out.text += kParameterOpen;
      out.text += arg.name + ">\n";
      if (arg.is_string)
        out.Literal(arg.value);
      else
        out.Literal(json::parse(arg.value).tojson());
      out.text += '\n';
      out.text += kParameterClose;
      out.text += '\n';
    }
    out.text += kFunctionClose;
    out.text += '\n';
    out.text += kCallClose;
  }
}
Rendered Render(std::span<const ext::ChatMessage> messages,
                std::span<const ext::ChatTool> tools,
                const ReasoningOptions& reasoning, bool generation) {
  Rendered out;
  std::size_t first = 0;
  std::optional<std::size_t> last_user;
  for (std::size_t i = 0; i < messages.size(); ++i) {
    if (!messages[i].images.empty())
      throw std::invalid_argument("LFM3 supports text input only");
    if (messages[i].role == ChatRole::kUser)
      last_user = i;
  }
  std::string system;
  if (!messages.empty() && messages.front().role == ChatRole::kSystem) {
    system = messages.front().content + messages.front().framing_suffix;
    first = 1;
  }
  if (!system.empty() || !tools.empty()) {
    out.text += kImStart;
    out.text += "system\n";
    out.Literal(system);
    if (!tools.empty()) {
      if (!system.empty())
        out.text += "\n\n";
      out.text += "<tools>\n";
      for (std::size_t i = 0; i < tools.size(); ++i) {
        if (i)
          out.text += '\n';
        out.Literal(ToolDefinition(tools[i]));
      }
      out.text += "\n</tools>";
    }
    out.text += kImEnd;
    out.text += '\n';
  }
  for (std::size_t i = first; i < messages.size(); ++i) {
    const auto& m = messages[i];
    out.text += kImStart;
    out.text += ToString(m.role);
    out.text += '\n';
    if (m.role == ChatRole::kAssistant) {
      auto content = std::string_view(m.content);
      const bool keep = reasoning.preserve_thinking.value_or(false) ||
                        !last_user || i > *last_user;
      if (keep) {
        if (!m.thought.empty()) {
          out.text += "<think>";
          out.Literal(m.thought);
          out.text += "</think>";
        } else if (content.find("</think>") == content.npos)
          out.text += "<think></think>";
      } else {
        auto end = content.rfind("</think>");
        if (end != content.npos)
          content = Trim(content.substr(end + 8));
        out.text += "<think></think>";
      }
      out.Literal(content);
      out.Literal(m.framing_suffix);
      Calls(out, m.tool_calls);
    } else if (m.role == ChatRole::kTool) {
      // Adjacent parts of one call are one native tool response.
      std::string content = m.content;
      while (i + 1 < messages.size() &&
             messages[i + 1].role == ChatRole::kTool &&
             !m.tool_call_id.empty() &&
             messages[i + 1].tool_call_id == m.tool_call_id) {
        if (!content.empty() && !messages[i + 1].content.empty())
          content += '\n';
        content += messages[++i].content;
      }
      if (content.find("<tool_response>") != std::string::npos)
        out.Literal(content);
      else {
        out.text += "<tool_response>\n";
        out.Literal(content);
        out.text += "\n</tool_response>";
      }
    } else {
      out.Literal(m.content);
      out.Literal(m.framing_suffix);
    }
    out.text += kImEnd;
    out.text += '\n';
  }
  if (generation) {
    out.text += kImStart;
    out.text += "assistant\n";
    out.text += ThinkingEnabled(reasoning) ? "<think>" : "<think></think>";
  }
  return out;
}
}  // namespace
bool ThinkingEnabled(const ReasoningOptions& reasoning) {
  return reasoning.enabled.value_or(true);
}
std::string RenderChat(std::span<const ext::ChatMessage> messages,
                       std::span<const ext::ChatTool> tools,
                       const ReasoningOptions& reasoning, bool generation) {
  return Render(messages, tools, reasoning, generation).text;
}
std::vector<std::uint32_t> EncodeChat(
    const Tokenizer& tokenizer, std::span<const ext::ChatMessage> messages,
    std::span<const ext::ChatTool> tools, const ReasoningOptions& reasoning,
    bool generation) {
  const auto rendered = Render(messages, tools, reasoning, generation);
  return tokenizer.EncodeFramed(rendered.text, rendered.literals);
}
}  // namespace gufo::models::lfm3
