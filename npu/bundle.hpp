#pragma once
// Whether `--npu` is on the command line. The NPU path needs nothing beside
// the GGUF: the weights upload to Gufo's installed NPU programs at load time.

#include <filesystem>
#include <gufo/ext/options.hpp>
#include <string>

namespace gufo::models::lfm3::npu {

// The model to run on the NPU (`gguf`) when the command line has --npu, else
// empty.
inline std::string SelectedBundle(const ext::ParsedOptions& options,
                                  const std::filesystem::path& gguf) {
  const bool* on = ext::FindOption<bool>(options, "--npu");
  if (!on || !*on)
    return {};
  return gguf.string();
}

}  // namespace gufo::models::lfm3::npu
