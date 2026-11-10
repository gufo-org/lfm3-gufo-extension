// Prefill-precision probe: for prompt lengths STRIDE, 2 * STRIDE, ... of a
// tokenized text, prefills the prefix in one Forward, then teacher-forces
// STEPS decode tokens. Writes every logits row (FP32, vocab wide) in order,
// preceded by a header of the evaluated next-token IDs, so runs of different
// builds or weights can be compared position by position.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "../engine.hpp"
#include "../tokenizer.hpp"
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr
        << "usage: lfm3_prefix_probe MODEL TEXT OUTPUT [STRIDE] [STEPS]\n";
    return 2;
  }
  using namespace gufo::models::lfm3;
  const std::size_t stride = argc > 4 ? std::stoul(argv[4]) : 300;
  const std::size_t steps = argc > 5 ? std::stoul(argv[5]) : 8;
  std::ifstream in(argv[2], std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)), {});
  std::string error;
  auto model = Model::Load(argv[1], &error);
  if (!model) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto tokens = model->tokenizer().Encode(text);
  const auto vocab = model->config().vocab;
  auto session = model->CreateSession(tokens.size() + 1, &error);
  if (!session) {
    std::cerr << error << '\n';
    return 1;
  }
  std::vector<std::uint32_t> targets;
  std::vector<std::size_t> starts;
  for (std::size_t n = stride; n + steps + 1 <= tokens.size(); n += stride) {
    starts.push_back(n);
    for (std::size_t s = 0; s <= steps; ++s)
      targets.push_back(tokens[n + s]);
  }
  std::ofstream out(argv[3], std::ios::binary);
  const std::uint32_t header[]{std::uint32_t(targets.size()), vocab};
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(reinterpret_cast<const char*>(targets.data()),
            targets.size() * sizeof(std::uint32_t));
  std::vector<float> logits(vocab);
  for (auto n : starts) {
    if (!session->Reset(&error) ||
        !session->Forward(std::span(tokens).first(n), logits, &error)) {
      std::cerr << error << '\n';
      return 1;
    }
    out.write(reinterpret_cast<const char*>(logits.data()), vocab * 4);
    for (std::size_t s = 0; s < steps; ++s) {
      if (!session->Forward(std::span(tokens).subspan(n + s, 1), logits,
                            &error)) {
        std::cerr << error << '\n';
        return 1;
      }
      out.write(reinterpret_cast<const char*>(logits.data()), vocab * 4);
    }
  }
  std::cerr << tokens.size() << " tokens, " << starts.size() << " prefixes, "
            << targets.size() << " rows\n";
}
