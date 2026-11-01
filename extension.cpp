#include <gufo/ext/asr.hpp>
#include <gufo/ext/decision.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <gufo/ext/tts.hpp>

#include "probe.hpp"

namespace gufo::models::lfm3 {

#if !defined(ENGINE_ENABLE_HIP)
void RegisterLfm3Cpu(ext::Registry& registry) {
  registry.AddTextProvider({
      .id = "lfm3",
      .probe = &ProbeLfm3,
      .probe_gguf = &ProbeLfm3Gguf,
      .create = nullptr,
      .bench_driver = nullptr,
      .prompt_driver = nullptr,
      .chat_driver = nullptr,
  });
  registry.AddAsrProvider({
      .id = "lfm3-audio-asr",
      .probe = &ProbeLfm3Audio,
      .resolve = &ResolveLfm3Audio,
      .create = nullptr,
  });
  registry.AddTtsProvider({
      .id = "lfm3-audio-tts",
      .probe = &ProbeLfm3Audio,
      .resolve = &ResolveLfm3Audio,
      .create = nullptr,
  });
  registry.AddDecisionProvider({
      .id = "d1",
      .probe = &ProbeD1,
      .resolve = &ResolveD1,
      .create = nullptr,
  });
}
#endif

}  // namespace gufo::models::lfm3

void lfm3_register(gufo::ext::Registry& registry) {
#if !defined(ENGINE_ENABLE_HIP)
  gufo::models::lfm3::RegisterLfm3Cpu(registry);
#else
  (void)registry;
#endif
}
