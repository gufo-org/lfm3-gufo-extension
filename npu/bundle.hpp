#pragma once
// Where `--npu` finds a model's NPU bundle: the directory npu/iron/bundle.py
// writes beside the GGUF, named after it with `.npu` in place of `.gguf`
// (LFM3-1.7B-Q8_0.gguf -> LFM3-1.7B-Q8_0.npu).

#include <filesystem>
#include <gufo/ext/options.hpp>
#include <stdexcept>
#include <string>

namespace gufo::models::lfm3::npu {

inline std::filesystem::path BundleFor(const std::filesystem::path& gguf) {
  std::filesystem::path dir = gguf;
  dir.replace_extension(".npu");
  return dir;
}

// The bundle to load when the command line has --npu, else empty. Throws when
// --npu is given and the bundle is missing.
inline std::string SelectedBundle(const ext::ParsedOptions& options,
                                  const std::filesystem::path& gguf) {
  const bool* on = ext::FindOption<bool>(options, "--npu");
  if (!on || !*on)
    return {};
  const auto dir = BundleFor(gguf);
  if (!std::filesystem::is_regular_file(dir / "bundle.txt"))
    throw std::invalid_argument(
        "--npu: no NPU bundle at " + dir.string() +
        " (write it with npu/iron/build.sh; see npu/README.md)");
  return dir.string();
}

}  // namespace gufo::models::lfm3::npu
