#ifndef GUFO_MODELS_LFM3_VL_VISION_HPP_
#define GUFO_MODELS_LFM3_VL_VISION_HPP_
#include <gufo/ext/gguf_reader.hpp>
#include <gufo/ext/image.hpp>
#include <memory>
#include <string>
#include <vector>
namespace gufo::models::lfm3_vl {
struct Picture {
  int height{}, width{};       // patch grid, in merge-block order
  std::vector<float> patches;  // [patch][RGB][16][16], normalized to [-1,1]
};
Picture ProcessImage(const core::Image& image, unsigned max_pixels = 16777216);
class Vision {
public:
  ~Vision();
  // npu_bundle: products on the NPU (npu::Dense), the rest on the host.
  static std::unique_ptr<Vision> Load(const std::string& path,
                                      const std::string& npu_bundle = {});
  std::vector<float> Encode(const Picture& picture);
  const std::string& Identity() const;

private:
  Vision();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
// The projector for a backbone: null for an LFM3 text backbone without one;
// an LFM3-VL backbone requires its LFM3-VL mmproj. Mismatches throw.
std::shared_ptr<Vision> LoadMatchingVision(const std::string& backbone,
                                           const std::string& mmproj,
                                           const std::string& npu_bundle = {});
}  // namespace gufo::models::lfm3_vl
#endif
