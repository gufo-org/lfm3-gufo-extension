#include "npu/dense.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <gufo/ext/experimental/npu.hpp>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace gufo::models::lfm3::npu {
namespace {
std::mutex g_open;
Dense* g_dense{};
}  // namespace

struct Dense::Impl {
  gufo::npu::Device device = gufo::npu::Device::Open();
  gufo::npu::Q8Gemm gemm;
};

Dense::Dense() : impl_(std::make_unique<Impl>()) {}
Dense::~Dense() = default;

void Dense::Open(const std::filesystem::path& bundle) {
  std::lock_guard lock(g_open);
  if (g_dense)
    return;
  auto* d = new Dense();  // lives for the process, like the NPU context
  d->impl_->gemm = gufo::npu::Q8Gemm::Open(d->impl_->device, bundle / "gemm");
  g_dense = d;
}

Dense& Dense::Shared() {
  if (!g_dense)
    throw std::logic_error("npu::Dense used before Open");
  return *g_dense;
}

namespace {
// Q8_0 of 32 values: codes; returns the f16-rounded scale.
float Quantize32(const float* v, std::int8_t* codes) {
  float top = 0;
  for (int i = 0; i < 32; ++i)
    top = std::max(top, std::fabs(v[i]));
  const float d = float(_Float16(top / 127.0f));
  for (int i = 0; i < 32; ++i)
    codes[i] =
        d > 0 ? std::int8_t(std::clamp(std::lround(v[i] / d), -127l, 127l)) : 0;
  return d;
}
}  // namespace

Dense::Matrix Dense::Upload(std::uint32_t rows, std::uint32_t cols,
                            const RowFn& row) {
  // Two Q8_0 planes over [K codes | K residual codes]; x repeats for the
  // second.
  const std::uint32_t kp = (cols + 31) / 32 * 32, k2 = 2 * kp;
  std::vector<std::int8_t> codes(std::size_t(rows) * k2, 0);
  std::vector<float> scales(std::size_t(rows) * (k2 / 32), 0.0f), buf(kp),
      rest(32);
  for (std::uint32_t r = 0; r < rows; ++r) {
    std::fill(buf.begin(), buf.end(), 0.0f);
    row(r, std::span<float>(buf.data(), cols));
    std::int8_t* c = codes.data() + std::size_t(r) * k2;
    float* sc = scales.data() + std::size_t(r) * (k2 / 32);
    for (std::uint32_t b = 0; b < kp / 32; ++b) {
      const float d = sc[b] = Quantize32(&buf[b * 32], c + b * 32);
      for (int i = 0; i < 32; ++i)
        rest[i] = buf[b * 32 + i] - d * float(c[b * 32 + i]);
      sc[kp / 32 + b] = Quantize32(rest.data(), c + kp + b * 32);
    }
  }
  Matrix w;
  w.k = cols, w.m = rows, w.planes = 2;
  w.weights = impl_->gemm.Upload(int(rows), int(k2), codes, scales);
  return w;
}

Dense::Matrix Dense::Upload(std::span<const Tensor* const> tensors) {
  std::uint32_t rows = 0;
  bool q8 = true;
  for (const auto* t : tensors) {
    if (t->cols != tensors[0]->cols)
      throw std::invalid_argument("NPU matrix with mixed columns");
    rows += t->rows;
    q8 = q8 && t->type == core::GgmlType::kQ8_0;
  }
  const std::uint32_t cols = tensors[0]->cols;
  if (!q8 || cols % 32)
    return Upload(rows, cols, [&](std::uint32_t r, std::span<float> out) {
      for (const auto* t : tensors) {
        if (r < t->rows)
          return DecodeRow(*t, r, out);
        r -= t->rows;
      }
    });
  // Q8_0 blocks: an f16 scale then 32 int8 codes.
  std::vector<std::int8_t> codes(std::size_t(rows) * cols);
  std::vector<float> scales(std::size_t(rows) * (cols / 32));
  std::size_t at = 0;
  for (const auto* t : tensors) {
    const auto* blocks = static_cast<const std::uint8_t*>(t->data);
    for (std::size_t b = 0; b < std::size_t(t->rows) * (cols / 32); ++b, ++at) {
      _Float16 d;
      std::memcpy(&d, blocks + b * 34, 2);
      scales[at] = float(d);
      std::memcpy(codes.data() + at * 32, blocks + b * 34 + 2, 32);
    }
  }
  Matrix w;
  w.k = cols, w.m = rows;
  w.weights = impl_->gemm.Upload(int(rows), int(cols), codes, scales);
  return w;
}

void Dense::Run(const Matrix& w, std::span<const float> x, std::uint32_t rows,
                std::span<float> y) {
  if (x.size() < std::size_t(rows) * w.k || y.size() < std::size_t(rows) * w.m)
    throw std::invalid_argument("NPU projection buffers too small");
  const std::uint32_t kp = (w.k + 31) / 32 * 32, width = kp * w.planes;
  const float* xs = x.data();
  std::vector<float> p;
  if (width != w.k) {  // columns padded to a multiple of 32, repeated per plane
    p.assign(std::size_t(rows) * width, 0.0f);
    for (std::uint32_t r = 0; r < rows; ++r)
      for (std::uint32_t q = 0; q < w.planes; ++q)
        std::copy_n(x.data() + std::size_t(r) * w.k, w.k,
                    p.data() + std::size_t(r) * width + q * kp);
    xs = p.data();
  }
  impl_->gemm.Run(w.weights, xs, int(rows), y.data());
}

}  // namespace gufo::models::lfm3::npu
