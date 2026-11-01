#ifndef GUFO_MODELS_LFM3_PROBE_HPP_
#define GUFO_MODELS_LFM3_PROBE_HPP_

#include <filesystem>
#include <gufo/ext/gguf_reader.hpp>
#include <optional>
#include <string>
#include <string_view>

namespace gufo::models::lfm3 {

[[nodiscard]] bool ProbeLfm3(std::string_view model_root);
// Causal LFM3 checkpoints (text, VL, audio backbones).
[[nodiscard]] bool ProbeLfm3Gguf(const core::GgufReader& reader);
// d1 decision checkpoints: bidirectional LFM3 with a decision readout.
[[nodiscard]] bool ProbeD1Gguf(const core::GgufReader& reader);

// An LFM3 backbone GGUF with the LFM3-Audio towers of its mmproj GGUF.
struct Lfm3AudioArtifacts {
  std::filesystem::path backbone;
  std::filesystem::path mmproj;
};

// model_root is an LFM3-Audio backbone GGUF, or a directory holding exactly
// one. mmproj, when given, must hold LFM3-Audio towers; otherwise the single
// mmproj*.gguf with them beside the backbone is used, or
// "mmproj-<backbone file name>" among several. An empty result with an empty
// error means the artifacts are not LFM3-Audio.
[[nodiscard]] std::optional<Lfm3AudioArtifacts> FindLfm3Audio(
    std::string_view model_root, std::string_view mmproj, std::string* error);
// An LFM3-Audio backbone, with or without a projector beside it.
[[nodiscard]] bool ProbeLfm3Audio(std::string_view model_root);
[[nodiscard]] bool ResolveLfm3Audio(std::string_view model_root,
                                    std::string* error);
// A d1 backbone GGUF, or a directory holding exactly one.
[[nodiscard]] std::optional<std::filesystem::path> FindD1(
    std::string_view model_root, std::string* error);
[[nodiscard]] bool ProbeD1(std::string_view model_root);
[[nodiscard]] bool ResolveD1(std::string_view model_root, std::string* error);

}  // namespace gufo::models::lfm3

#endif  // GUFO_MODELS_LFM3_PROBE_HPP_
