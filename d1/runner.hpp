#ifndef GUFO_MODELS_D1_RUNNER_HPP_
#define GUFO_MODELS_D1_RUNNER_HPP_

#include <gufo/ext/decision.hpp>
#include <memory>
#include <string>

namespace gufo::models::d1 {

// model_root is a d1 GGUF or a directory holding one; projector_path, when
// set, adds image and audio inputs. Contexts above 16384 are rejected.
std::unique_ptr<ext::DecisionRunner> CreateD1Runner(
    const ext::DecisionRunnerOptions& options, std::string* error);

}  // namespace gufo::models::d1

#endif  // GUFO_MODELS_D1_RUNNER_HPP_
