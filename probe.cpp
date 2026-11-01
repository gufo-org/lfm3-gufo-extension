#include "probe.hpp"

#include <algorithm>
#include <filesystem>
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

bool IsLfm3Backbone(const std::filesystem::path& path) {
  if (!IsGguf(path) || IsMmprojName(path))
    return false;
  std::string error;
  auto reader = core::GgufReader::OpenFile(path.string(), &error);
  return reader && ProbeLfm3Gguf(*reader);
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
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  return architecture == "lfm3";
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

std::optional<Lfm3AudioArtifacts> FindLfm3Audio(std::string_view model_root,
                                                std::string* error) {
  error->clear();
  try {
    std::filesystem::path backbone(model_root);
    std::error_code ec;
    if (std::filesystem::is_directory(backbone, ec)) {
      const auto found = Entries(backbone, &IsLfm3Backbone);
      if (found.empty())
        return std::nullopt;
      if (found.size() > 1) {
        // Only LFM3-Audio directories are ambiguous; text ones are not ours.
        if (Entries(backbone, &IsLfm3AudioMmproj).empty())
          return std::nullopt;
        *error = "several LFM3 GGUF files in " + backbone.string() +
                 "; select one by path";
        return std::nullopt;
      }
      backbone = found.front();
    } else if (!IsLfm3Backbone(backbone)) {
      return std::nullopt;
    }
    const auto dir = backbone.parent_path().empty() ? std::filesystem::path(".")
                                                    : backbone.parent_path();
    auto mmproj = Entries(dir, &IsLfm3AudioMmproj);
    if (mmproj.size() > 1) {
      const auto paired = dir / ("mmproj-" + backbone.filename().string());
      if (std::find(mmproj.begin(), mmproj.end(), paired) == mmproj.end()) {
        *error = "several LFM3-Audio mmproj files beside " + backbone.string() +
                 " and none named " + paired.filename().string();
        return std::nullopt;
      }
      mmproj = {paired};
    }
    if (mmproj.empty())
      return std::nullopt;
    return Lfm3AudioArtifacts{std::move(backbone), std::move(mmproj.front())};
  } catch (const std::exception& e) {
    *error = e.what();
    return std::nullopt;
  }
}

bool ProbeLfm3Audio(std::string_view model_root) {
  std::string error;
  return FindLfm3Audio(model_root, &error).has_value();
}

bool ResolveLfm3Audio(std::string_view model_root, std::string* error) {
  std::string reason;
  if (FindLfm3Audio(model_root, &reason))
    return true;
  if (error)
    *error = reason.empty() ? "no LFM3 GGUF with an LFM3-Audio mmproj at " +
                                  std::string(model_root)
                            : reason;
  return false;
}

}  // namespace gufo::models::lfm3
