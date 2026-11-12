#include "runner.hpp"

#include <exception>
#include <utility>

#include "../npu/bundle.hpp"
#include "../probe.hpp"
#include "model.hpp"

namespace gufo::models::d1 {
namespace {

class D1Runner final : public ext::DecisionRunner {
public:
  D1Runner(std::unique_ptr<Model> model, bool multimodal)
      : model_(std::move(model)), multimodal_(multimodal) {}
  json::Value Decide(const json::Value& request) override {
    return model_->Decide(request);
  }
  bool Multimodal() const override { return multimodal_; }
  std::string ModelId() const override { return "d1-Omni-2B"; }

private:
  std::unique_ptr<Model> model_;
  bool multimodal_;
};

}  // namespace

std::unique_ptr<ext::DecisionRunner> CreateD1Runner(
    const ext::DecisionRunnerOptions& options, std::string* error) {
  std::string reason;
  const auto backbone = lfm3::FindD1(options.model_root, &reason);
  if (!backbone) {
    *error = reason.empty() ? "no d1 GGUF at " + options.model_root : reason;
    return nullptr;
  }
  if (options.maximum_context_tokens == 0 ||
      options.maximum_context_tokens > 16384) {
    *error = "d1 context must be in 1..16384";
    return nullptr;
  }
  try {
    auto model = Model::Load(
        backbone->string(), options.projector_path,
        unsigned(options.maximum_context_tokens),
        lfm3::npu::SelectedBundle(options.provider_options, *backbone));
    return std::make_unique<D1Runner>(std::move(model),
                                      !options.projector_path.empty());
  } catch (const std::exception& e) {
    *error = e.what();
    return nullptr;
  }
}

}  // namespace gufo::models::d1
