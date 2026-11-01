#include <gufo/ext/generated/lfm3_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"
#include "runner.hpp"

namespace gufo::models::lfm3 {

int RunLfm3Bench(const ext::TextBenchOptions& opt,
                 const std::shared_ptr<const core::GgufReader>& reader,
                 std::chrono::steady_clock::time_point load_start);

int RunLfm3Prompt(const ext::TextPromptOptions& opt,
                  const core::GgufReader& reader,
                  std::chrono::steady_clock::time_point load_start);

int RunLfm3Chat(const ext::TextPromptOptions& opt,
                const core::GgufReader& reader,
                std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::models::lfm3

void lfm3_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_lfm3::kProvider_lfm3);
  registry.AddTextProvider({
      .id = "lfm3",
      .probe = &gufo::models::lfm3::ProbeLfm3,
      .probe_gguf = &gufo::models::lfm3::ProbeLfm3Gguf,
      .create = &gufo::models::lfm3::CreateLfm3TextRunner,
      .bench_driver = &gufo::models::lfm3::RunLfm3Bench,
      .prompt_driver = &gufo::models::lfm3::RunLfm3Prompt,
      .chat_driver = &gufo::models::lfm3::RunLfm3Chat,
  });
}
