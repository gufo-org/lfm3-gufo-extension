// Fuses generated NPU kernels into one dispatch: each stage's public kernel
// becomes a private child, run in order as pipeline.serial phases of a new
// public kernel. Phases pass data through device buffers; a phase starts after
// the previous one's strands and transfers complete.
#ifndef GUFO_MODELS_LFM3_NPU_FUSE_HPP_
#define GUFO_MODELS_LFM3_NPU_FUSE_HPP_

#include <string>
#include <vector>

namespace gufo::models::lfm3::npu {

struct FuseStage {
  std::string program;    // a complete generated module
  std::string root;       // its public kernel, with '@'
  std::vector<int> args;  // fused binding index of each stage argument
};

// Top-level definitions shared between stages (target, math library,
// helpers) are emitted once, keyed by symbol name.
std::string FuseProgram(const std::vector<FuseStage>& stages, int bindings, const std::string& name);

}  // namespace gufo::models::lfm3::npu

#endif
