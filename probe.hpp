#ifndef GUFO_MODELS_LFM3_PROBE_HPP_
#define GUFO_MODELS_LFM3_PROBE_HPP_

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::lfm3 {

[[nodiscard]] bool ProbeLfm3(std::string_view model_root);
[[nodiscard]] bool ProbeLfm3Gguf(const core::GgufReader& reader);

}  // namespace gufo::models::lfm3

#endif  // GUFO_MODELS_LFM3_PROBE_HPP_
