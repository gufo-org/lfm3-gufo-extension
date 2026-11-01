#include <gufo/ext/decision.hpp>
#include <gufo/ext/generated/lfm3_constants.hpp>
#include <gufo/ext/registry.hpp>

#include "../probe.hpp"
#include "runner.hpp"

void lfm3_d1_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_lfm3::kProvider_d1);
  registry.AddDecisionProvider({
      .id = "d1",
      .probe = &gufo::models::lfm3::ProbeD1,
      .resolve = &gufo::models::lfm3::ResolveD1,
      .create = &gufo::models::d1::CreateD1Runner,
  });
}
