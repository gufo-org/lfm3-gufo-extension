#ifndef GUFO_MODELS_LFM3_UPLOAD_HPP_
#define GUFO_MODELS_LFM3_UPLOAD_HPP_
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "weights.hpp"
namespace gufo::models::lfm3 {
// Device copies of a GGUF's tensor payloads, filled by gufo's parallel
// unbuffered disk reader instead of faulting the mapping page by page.
// Payloads at one host address (tied tensors) share one device copy.
class DeviceWeights {
public:
  DeviceWeights() = default;
  ~DeviceWeights();
  DeviceWeights(const DeviceWeights&) = delete;
  DeviceWeights& operator=(const DeviceWeights&) = delete;

  // Records a payload mapped by the reader later passed to Upload.
  void Add(const Tensor& tensor);
  // Allocates and fills the device copy of every added payload; throws.
  void Upload(const core::GgufReader& reader);
  // Points a recorded tensor at its device copy.
  void Resolve(Tensor& tensor) const;

private:
  std::unordered_map<const void*, std::size_t> bytes_;
  std::unordered_map<const void*, void*> device_;
  std::vector<void*> segments_;
};
}  // namespace gufo::models::lfm3
#endif
