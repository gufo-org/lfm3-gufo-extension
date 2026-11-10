#include "upload.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <functional>
#include <gufo/ext/experimental/weight_upload.hpp>
#include <stdexcept>
#include <string>
#include <utility>
namespace gufo::models::lfm3 {
namespace {
constexpr std::size_t kAlignment = 256;
// Allocation maps zeroed pages at about 28 GB/s, five times the disk rate:
// after the first segment, allocating the next one overlaps the reads.
constexpr std::size_t kSegmentBytes = 128ULL << 20;
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
std::size_t Aligned(std::size_t bytes) {
  return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}
}  // namespace
DeviceWeights::~DeviceWeights() {
  for (auto* segment : segments_)
    (void)hipFree(segment);
}
void DeviceWeights::Add(const Tensor& tensor) {
  if (tensor.data)
    bytes_.emplace(tensor.data, tensor.bytes);
}
void DeviceWeights::Upload(const core::GgufReader& reader) {
  if (!segments_.empty() || bytes_.empty())
    throw std::logic_error("LFM3 weights uploaded twice or empty");
  std::string error;
  auto uploader = hip::WeightUpload::Create(reader.GetMappedRegions(), &error);
  // File order streams the shards front to back.
  std::vector<std::pair<const void*, std::size_t>> order(bytes_.begin(),
                                                         bytes_.end());
  std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
    return std::less<>{}(a.first, b.first);
  });
  bool queued = true;
  for (std::size_t first = 0; first < order.size();) {
    auto end = first;
    std::size_t bytes = 0;
    while (end < order.size() &&
           (end == first || bytes + order[end].second <= kSegmentBytes))
      bytes += Aligned(order[end++].second);
    segments_.emplace_back();
    Check(hipMalloc(&segments_.back(), bytes));
    auto* device = static_cast<char*>(segments_.back());
    for (; first < end; ++first) {
      const auto [host, size] = order[first];
      device_.emplace(host, device);
      if (uploader && queued)
        queued = uploader->CopyMapped(host, size, device, &error);
      else if (!uploader)
        // In-memory readers have no file to read; copy from the image.
        Check(hipMemcpy(device, host, size, hipMemcpyHostToDevice));
      device += Aligned(size);
    }
  }
  // Finish waits for queued copies even after a failed one.
  if (uploader && (!uploader->Finish(&error) || !queued))
    throw std::runtime_error("LFM3 weight upload failed: " + error);
}
void DeviceWeights::Resolve(Tensor& tensor) const {
  if (tensor.data)
    tensor.data = device_.at(tensor.data);
}
}  // namespace gufo::models::lfm3
