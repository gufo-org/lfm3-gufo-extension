#include "dense.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

#include "host/bfp.hpp"
#include "host/npu_runtime.hpp"

namespace gufo::models::lfm3::npu {
namespace {
constexpr std::uint32_t kMaxBlocks = 64;  // rows per dispatch = 8 * blocks

// Blocks per dispatch: exact up to eight, then powers of two (rows padded),
// so a few kernels serve every row count.
std::uint32_t DispatchBlocks(std::uint32_t blocks) {
  if (blocks <= 8) return blocks;
  std::uint32_t b = 16;
  while (b < blocks && b < kMaxBlocks) b *= 2;
  return std::min(b, kMaxBlocks);
}
}  // namespace

struct Dense::Impl {
  std::mutex mutex;
  ::lfm3::npu::Runtime npu;
  std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, int>, int> kernels;
  std::map<std::uint64_t, int> buffers;  // byte size -> scratch buffer
  std::vector<std::uint16_t> xb;
  std::vector<float> yb;

  int Kernel(ProjectionShape s) {
    const auto key = std::make_tuple(s.k, s.m, s.blocks, int(s.split));
    if (auto it = kernels.find(key); it != kernels.end()) return it->second;
    const int id = npu.Compile(ProjectionProgram(s), "projection.loom", ProjectionRoot(s));
    kernels.emplace(key, id);
    return id;
  }
  // Scratch buffers by role and size class (rounded to 64 KiB).
  int Scratch(int role, std::uint64_t bytes) {
    bytes = (bytes + 65535) / 65536 * 65536;
    const std::uint64_t key = bytes * 4 + std::uint64_t(role);
    if (auto it = buffers.find(key); it != buffers.end()) return it->second;
    const int id = npu.Alloc(bytes);
    buffers.emplace(key, id);
    return id;
  }
};

Dense::Dense() : impl_(std::make_unique<Impl>()) {}
Dense::~Dense() = default;

Dense& Dense::Shared() {
  static Dense* instance = new Dense();  // the HRX context lives for the process
  return *instance;
}

Dense::Matrix Dense::Upload(std::uint32_t rows, std::uint32_t cols, const RowFn& row) {
  Matrix w;
  w.k = cols;
  w.m = rows;
  w.shape = PlanProjection(cols, rows, 1);
  const auto split = w.shape.split == Split::kRows ? ::lfm3::npu::Split::kRows
                                                   : ::lfm3::npu::Split::kColumns;
  const std::uint32_t kp = w.shape.k;
  auto bytes = ::lfm3::npu::PackWeights(rows, kp, w.shape.m, split, [&](std::uint32_t r, std::span<float> out) {
    row(r, out.subspan(0, cols));
    std::fill(out.begin() + cols, out.end(), 0.0f);
  });
  std::lock_guard lock(impl_->mutex);
  w.weights = impl_->npu.Alloc(bytes.size());
  constexpr std::size_t kChunk = 64u << 20;
  for (std::size_t off = 0; off < bytes.size(); off += kChunk)
    impl_->npu.Write(w.weights, off, bytes.data() + off, std::min(kChunk, bytes.size() - off));
  return w;
}

Dense::Matrix Dense::Upload(std::span<const Tensor* const> tensors) {
  std::uint32_t rows = 0;
  for (const auto* t : tensors) {
    if (t->cols != tensors[0]->cols) throw std::invalid_argument("NPU matrix with mixed columns");
    rows += t->rows;
  }
  return Upload(rows, tensors[0]->cols, [&](std::uint32_t r, std::span<float> out) {
    for (const auto* t : tensors) {
      if (r < t->rows) return DecodeRow(*t, r, out);
      r -= t->rows;
    }
  });
}

void Dense::Run(const Matrix& w, std::span<const float> x, std::uint32_t rows, std::span<float> y) {
  if (x.size() < std::size_t(rows) * w.k || y.size() < std::size_t(rows) * w.m)
    throw std::invalid_argument("NPU projection buffers too small");
  auto& m = *impl_;
  std::lock_guard lock(m.mutex);
  const std::uint32_t kp = w.shape.k, mp = w.shape.m;
  const std::uint32_t parts = w.shape.split == Split::kColumns ? kColumns : 1;
  for (std::uint32_t row0 = 0; row0 < rows;) {
    const std::uint32_t remaining = (rows - row0 + 7) / 8;
    const std::uint32_t blocks = DispatchBlocks(std::min(remaining, kMaxBlocks));
    const std::uint32_t take = std::min(rows - row0, blocks * 8);
    ProjectionShape s = w.shape;
    s.blocks = blocks;
    const int kernel = m.Kernel(s);
    // Blocked BF16 x, zero beyond `take` rows and `k` columns.
    m.xb.assign(std::size_t(blocks) * 8 * kp, 0);
    for (std::uint32_t r = 0; r < take; ++r) {
      const float* src = x.data() + std::size_t(row0 + r) * w.k;
      std::uint16_t* block = m.xb.data() + std::size_t(r / 8) * 8 * kp;
      const std::uint32_t b = r % 8;
      for (std::uint32_t k = 0; k < w.k; ++k)
        block[(std::size_t(k / 8) * 8 + b) * 8 + k % 8] = ::lfm3::npu::ToBF16(src[k]);
    }
    const std::size_t block_out = std::size_t(parts) * mp * 8;
    const int xbuf = m.Scratch(0, m.xb.size() * 2);
    const int ybuf = m.Scratch(1, block_out * blocks * 4);
    m.npu.Write(xbuf, 0, m.xb.data(), m.xb.size() * 2);
    const int bind[] = {xbuf, w.weights, ybuf};
    m.npu.Dispatch(kernel, bind);
    const std::uint32_t used = (take + 7) / 8;
    m.yb.resize(block_out * used);
    m.npu.Read(ybuf, 0, m.yb.data(), m.yb.size() * 4);
    for (std::uint32_t r = 0; r < take; ++r) {
      const float* block = m.yb.data() + std::size_t(r / 8) * block_out;
      const std::uint32_t b = r % 8;
      float* dst = y.data() + std::size_t(row0 + r) * w.m;
      for (std::uint32_t o = 0; o < w.m; ++o) {
        float sum = 0;
        for (std::uint32_t p = 0; p < parts; ++p)
          sum += block[p * std::size_t(mp) * 8 + (std::size_t(o / 8) * 8 + b) * 8 + o % 8];
        dst[o] = sum;
      }
    }
    row0 += take;
  }
}

}  // namespace gufo::models::lfm3::npu
