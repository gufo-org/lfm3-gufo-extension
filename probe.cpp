#include "probe.hpp"

#include <filesystem>
#include <string>

namespace gufo::models::lfm3 {

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

}  // namespace gufo::models::lfm3
