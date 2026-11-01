#ifndef GUFO_MODELS_LFM3_PROBE_HPP_
#define GUFO_MODELS_LFM3_PROBE_HPP_

#include <filesystem>
#include <gufo/ext/gguf_reader.hpp>
#include <optional>
#include <string>
#include <string_view>

namespace gufo::models::lfm3 {

[[nodiscard]] bool ProbeLfm3(std::string_view model_root);
[[nodiscard]] bool ProbeLfm3Gguf(const core::GgufReader& reader);

// An LFM3 backbone GGUF with the LFM3-Audio towers of its mmproj GGUF.
struct Lfm3AudioArtifacts {
  std::filesystem::path backbone;
  std::filesystem::path mmproj;
};

// model_root is an LFM3 backbone GGUF, or a directory holding exactly one.
// The mmproj is the single mmproj*.gguf with LFM3-Audio towers beside the
// backbone, or "mmproj-<backbone file name>" among several. An empty result
// with an empty error means the artifacts are not LFM3-Audio.
[[nodiscard]] std::optional<Lfm3AudioArtifacts> FindLfm3Audio(
    std::string_view model_root, std::string* error);
[[nodiscard]] bool ProbeLfm3Audio(std::string_view model_root);
[[nodiscard]] bool ResolveLfm3Audio(std::string_view model_root,
                                    std::string* error);

}  // namespace gufo::models::lfm3

#endif  // GUFO_MODELS_LFM3_PROBE_HPP_
