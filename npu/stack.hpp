// LFM2/LFM3 stacks of the LFM3-Audio towers on the NPU: projections through
// npu::Dense, the rest on the host, following audio/model.cpp RunStack.
#ifndef GUFO_MODELS_LFM3_NPU_STACK_HPP_
#define GUFO_MODELS_LFM3_NPU_STACK_HPP_

#include <cstdint>
#include <vector>

#include "../audio/weights.hpp"
#include "dense.hpp"

namespace gufo::models::lfm3::npu {

class Stack {
 public:
  // Recurrent state: convolution histories and F16 K/V rings of `window`.
  struct State {
    std::vector<std::vector<float>> conv;
    std::vector<std::vector<_Float16>> k, v;
    std::uint32_t capacity{};
    int position{};
    void Clear();
  };
  explicit Stack(const lfm3_audio::Stack& weights);
  State NewState(std::uint32_t window) const;
  // Runs n rows of x (hidden floats each) through the stack; z receives the
  // output-normed rows. Without state, attention is bidirectional over the
  // rows (segment > 0 splits them into independent sequences of that many
  // rows) and convolutions are absent.
  void Run(std::span<const float> x, int n, State* state, int window, std::span<float> z,
           int segment = 0) const;
  const lfm3_audio::Stack& weights() const { return *w_; }

 private:
  struct Block {
    Dense::Matrix qkvg, out, in, gateup, down;
  };
  const lfm3_audio::Stack* w_;
  std::vector<Block> blocks_;
  std::vector<float> frequencies_;
};

}  // namespace gufo::models::lfm3::npu

#endif
