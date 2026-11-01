#include "probe.hpp"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace gufo::models::lfm3 {
namespace {

bool IsGguf(const std::filesystem::path& path) {
  std::error_code ec;
  return path.extension() == ".gguf" &&
         std::filesystem::is_regular_file(path, ec);
}

bool IsMmprojName(const std::filesystem::path& path) {
  return path.filename().string().starts_with("mmproj");
}

std::unique_ptr<core::GgufReader> OpenBackbone(
    const std::filesystem::path& path) {
  if (!IsGguf(path) || IsMmprojName(path))
    return nullptr;
  std::string error;
  return core::GgufReader::OpenFile(path.string(), &error);
}

bool IsLfm3AudioBackbone(const std::filesystem::path& path) {
  auto reader = OpenBackbone(path);
  return reader && ProbeLfm3Gguf(*reader) &&
         reader->GetMetadataString("general.basename") == "LFM3-Audio";
}

bool IsD1Backbone(const std::filesystem::path& path) {
  auto reader = OpenBackbone(path);
  return reader && ProbeD1Gguf(*reader);
}

bool IsLfm3AudioMmproj(const std::filesystem::path& path) {
  if (!IsGguf(path) || !IsMmprojName(path))
    return false;
  std::string error;
  auto reader = core::GgufReader::OpenFile(path.string(), &error);
  return reader &&
         reader->GetMetadataString("clip.audio.projector_type") == "lfm3a" &&
         reader->GetMetadataString("clip.gen.audio.projector_type") ==
             "lfm3a_gen";
}

std::vector<std::filesystem::path> Entries(
    const std::filesystem::path& dir,
    bool (*keep)(const std::filesystem::path&)) {
  std::vector<std::filesystem::path> found;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
    if (keep(entry.path()))
      found.push_back(entry.path());
  return found;
}

}  // namespace

bool ProbeLfm3Gguf(const core::GgufReader& reader) {
  // Decision checkpoints share the architecture but not generation.
  return reader.GetMetadataString("general.architecture").value_or("") ==
             "lfm3" &&
         reader.GetMetadataBool("lfm3.attention.causal").value_or(true);
}

bool ProbeD1Gguf(const core::GgufReader& reader) {
  return reader.GetMetadataString("general.architecture").value_or("") ==
             "lfm3" &&
         !reader.GetMetadataBool("lfm3.attention.causal").value_or(true) &&
         reader.GetMetadataString("lfm3.decision.type") == "lfm3-d1-omni";
}
bool ProbeLfm3(std::string_view model_root) {
  try {
    std::filesystem::path p(model_root);
    if (!std::filesystem::exists(p)) {
      return false;
    }
    if (std::filesystem::is_regular_file(p)) {
      std::string err;
      auto reader = core::GgufReader::OpenFile(p.string(), &err);
      if (!reader) {
        return false;
      }
      return ProbeLfm3Gguf(*reader);
    }
    return false;
  } catch (...) {
    return false;
  }
}

namespace {
// model_root is a backbone file, or a directory holding exactly one.
std::optional<std::filesystem::path> FindBackbone(
    std::string_view model_root,
    bool (*is_backbone)(const std::filesystem::path&), std::string_view family,
    std::string* error) {
  std::filesystem::path backbone(model_root);
  std::error_code ec;
  if (!std::filesystem::is_directory(backbone, ec))
    return is_backbone(backbone) ? std::optional(backbone) : std::nullopt;
  const auto found = Entries(backbone, is_backbone);
  if (found.size() > 1)
    *error = "several " + std::string(family) + " GGUF files in " +
             backbone.string() + "; select one by path";
  if (found.size() != 1)
    return std::nullopt;
  return found.front();
}
}  // namespace

std::optional<Lfm3AudioArtifacts> FindLfm3Audio(std::string_view model_root,
                                                std::string_view mmproj,
                                                std::string* error) {
  error->clear();
  try {
    auto backbone =
        FindBackbone(model_root, &IsLfm3AudioBackbone, "LFM3-Audio", error);
    if (!backbone)
      return std::nullopt;
    if (!mmproj.empty()) {
      if (!IsLfm3AudioMmproj(std::filesystem::path(mmproj))) {
        *error =
            "--mmproj is not an LFM3-Audio mmproj GGUF: " + std::string(mmproj);
        return std::nullopt;
      }
      return Lfm3AudioArtifacts{std::move(*backbone),
                                std::filesystem::path(mmproj)};
    }
    const auto dir = backbone->parent_path().empty()
                         ? std::filesystem::path(".")
                         : backbone->parent_path();
    auto found = Entries(dir, &IsLfm3AudioMmproj);
    if (found.size() > 1) {
      const auto paired = dir / ("mmproj-" + backbone->filename().string());
      if (std::find(found.begin(), found.end(), paired) == found.end()) {
        *error = "several LFM3-Audio mmproj files beside " +
                 backbone->string() + " and none named " +
                 paired.filename().string() + "; select one with --mmproj";
        return std::nullopt;
      }
      found = {paired};
    }
    if (found.empty()) {
      *error = "no LFM3-Audio mmproj*.gguf beside " + backbone->string() +
               "; select one with --mmproj";
      return std::nullopt;
    }
    return Lfm3AudioArtifacts{std::move(*backbone), std::move(found.front())};
  } catch (const std::exception& e) {
    *error = e.what();
    return std::nullopt;
  }
}

bool ProbeLfm3Audio(std::string_view model_root) {
  std::string error;
  try {
    return FindBackbone(model_root, &IsLfm3AudioBackbone, "LFM3-Audio", &error)
        .has_value();
  } catch (...) {
    return false;
  }
}

bool ResolveLfm3Audio(std::string_view model_root, std::string* error) {
  std::string reason;
  if (FindLfm3Audio(model_root, {}, &reason))
    return true;
  if (error)
    *error = reason;
  return false;
}

std::optional<std::filesystem::path> FindD1(std::string_view model_root,
                                            std::string* error) {
  error->clear();
  try {
    return FindBackbone(model_root, &IsD1Backbone, "d1", error);
  } catch (const std::exception& e) {
    *error = e.what();
    return std::nullopt;
  }
}

bool ProbeD1(std::string_view model_root) {
  std::string error;
  return FindD1(model_root, &error).has_value();
}

bool ResolveD1(std::string_view model_root, std::string* error) {
  std::string reason;
  if (FindD1(model_root, &reason))
    return true;
  if (error)
    *error =
        reason.empty() ? "no d1 GGUF at " + std::string(model_root) : reason;
  return false;
}

}  // namespace gufo::models::lfm3
