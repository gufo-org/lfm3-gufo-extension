#ifndef GUFO_MODELS_LFM3_RUNNER_HPP_
#define GUFO_MODELS_LFM3_RUNNER_HPP_

#include <cstdint>
#include <gufo/ext/text.hpp>
#include <memory>
#include <string>

#include "engine.hpp"

namespace gufo::models::lfm3 {

std::shared_ptr<ext::TextModelRunner> CreateLfm3Runner(
    std::shared_ptr<Model> model, std::uint32_t context,
    std::string artifact_fingerprint = {});

std::shared_ptr<ext::TextModelRunner> CreateLfm3TextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

}  // namespace gufo::models::lfm3
#endif
