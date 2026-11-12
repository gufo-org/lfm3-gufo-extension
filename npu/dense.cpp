#include "dense.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>

#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
#include <memory>
#include <string>

#include "../xrt/gufo_xrt_gemm.h"
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

// GUFO_LFM3_DEVICE=xrt: the products run on q8gemm.xclbin through XRT
// (gufo_xrt_gemm.h in LFM3_XRT_LIBRARY, xclbin from LFM3_XRT_GEMM or
// LFM3_XRT_MODEL/gemm) and HRX is never loaded.
struct XrtGemm {
  void* lib{};
  gufo_xrt_gemm* g{};
  decltype(&gufo_xrt_gemm_create) create{};
  decltype(&gufo_xrt_gemm_last_error) error{};
  decltype(&gufo_xrt_gemm_upload) upload{};
  decltype(&gufo_xrt_gemm_run) run{};
  XrtGemm() {
    const char* path = std::getenv("LFM3_XRT_LIBRARY");
    lib = dlopen(path ? path : "liblfm3_xrt.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error(std::string("XRT GEMM library: ") + dlerror());
    auto sym = [&](auto& f, const char* name) {
      f = reinterpret_cast<std::remove_reference_t<decltype(f)>>(dlsym(lib, name));
      if (!f) throw std::runtime_error(std::string("XRT GEMM library lacks ") + name);
    };
    sym(create, "gufo_xrt_gemm_create"), sym(error, "gufo_xrt_gemm_last_error");
    sym(upload, "gufo_xrt_gemm_upload"), sym(run, "gufo_xrt_gemm_run");
    std::string dir;
    if (const char* d = std::getenv("LFM3_XRT_GEMM")) dir = d;
    else if (const char* m = std::getenv("LFM3_XRT_MODEL")) dir = std::string(m) + "/gemm";
    else throw std::runtime_error("XRT GEMM: set LFM3_XRT_GEMM or LFM3_XRT_MODEL");
    g = create(dir.c_str());
    if (!g) throw std::runtime_error(std::string("XRT GEMM: ") + error());
  }
};

bool OnXrt() {
  const char* d = std::getenv("GUFO_LFM3_DEVICE");
  return d && std::string_view(d) == "xrt";
}

struct Dense::Impl {
  std::mutex mutex;
  std::unique_ptr<::lfm3::npu::Runtime> hrx;
  std::unique_ptr<XrtGemm> xrt;
  ::lfm3::npu::Runtime& npu_runtime() {
    if (!hrx) hrx = std::make_unique<::lfm3::npu::Runtime>();
    return *hrx;
  }
  std::map<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, int>, int> kernels;
  std::map<std::uint64_t, int> buffers;  // byte size -> scratch buffer
  std::vector<std::uint16_t> xb;
  std::vector<float> yb;

  int Kernel(ProjectionShape s) {
    const auto key = std::make_tuple(s.k, s.m, s.blocks, int(s.split));
    if (auto it = kernels.find(key); it != kernels.end()) return it->second;
    // K split: each weight record serves every block held at once, and the
    // columns add their partial sums on chip (one output part).
    const bool chain = s.split == Split::kColumns;
    // Long prompt passes use two matrix cores per column (faster from about
    // 12 blocks; below that the single-core chain wins).
    const bool dual = chain && s.blocks >= 12 && DualBlocksPerCore(s) > 0;
    const int id = dual    ? npu_runtime().Compile(ProjectionDualProgram(s), "projection.loom", ProjectionRoot(s) + "_kdual")
                   : chain ? npu_runtime().Compile(ProjectionKMultiProgram(s, true), "projection.loom", ProjectionRoot(s) + "_kchain")
                           : npu_runtime().Compile(ProjectionProgram(s), "projection.loom", ProjectionRoot(s));
    kernels.emplace(key, id);
    return id;
  }
  // Scratch buffers by role and size class (rounded to 64 KiB).
  int Scratch(int role, std::uint64_t bytes) {
    bytes = (bytes + 65535) / 65536 * 65536;
    const std::uint64_t key = bytes * 4 + std::uint64_t(role);
    if (auto it = buffers.find(key); it != buffers.end()) return it->second;
    const int id = npu_runtime().Alloc(bytes);
    buffers.emplace(key, id);
    return id;
  }
};

Dense::Dense() : impl_(std::make_unique<Impl>()) {
  if (OnXrt()) impl_->xrt = std::make_unique<XrtGemm>();
}
Dense::~Dense() = default;

Dense& Dense::Shared() {
  static Dense* instance = new Dense();  // the HRX context lives for the process
  return *instance;
}

Dense::Matrix Dense::Upload(std::uint32_t rows, std::uint32_t cols, const RowFn& row, bool k_split) {
  Matrix w;
  w.k = cols;
  w.m = rows;
  w.shape = PlanProjection(cols, rows, 1);
  if (k_split && w.shape.split == Split::kRows) {
    w.shape.split = Split::kColumns;
    w.shape.k = (cols + 511) / 512 * 512;
    w.shape.m = (rows + 7) / 8 * 8;
  }
  if (impl_->xrt) {
    // Q8_0 of the rows: the codes and scales of Q8_0 tensors come back exactly.
    const std::uint32_t kp = (cols + 31) / 32 * 32;
    std::vector<std::int8_t> codes(std::size_t(rows) * kp, 0);
    std::vector<float> scales(std::size_t(rows) * (kp / 32), 0.0f), buf(kp);
    for (std::uint32_t r = 0; r < rows; ++r) {
      std::fill(buf.begin(), buf.end(), 0.0f);
      row(r, std::span<float>(buf.data(), cols));
      for (std::uint32_t b = 0; b < kp / 32; ++b) {
        float top = 0;
        for (int i = 0; i < 32; ++i) top = std::max(top, std::fabs(buf[b * 32 + i]));
        const float d = float(_Float16(top / 127.0f));
        scales[std::size_t(r) * (kp / 32) + b] = d;
        for (int i = 0; i < 32; ++i)
          codes[std::size_t(r) * kp + b * 32 + i] =
              d > 0 ? std::int8_t(std::clamp(std::lround(buf[b * 32 + i] / d), -127l, 127l)) : 0;
      }
    }
    std::lock_guard lock(impl_->mutex);
    w.weights = impl_->xrt->upload(impl_->xrt->g, int(rows), int(kp), codes.data(), scales.data());
    if (w.weights < 0) throw std::runtime_error(std::string("XRT GEMM upload: ") + impl_->xrt->error());
    w.shape.k = kp;
    return w;
  }
  const auto split = w.shape.split == Split::kRows ? ::lfm3::npu::Split::kRows
                                                   : ::lfm3::npu::Split::kColumns;
  const std::uint32_t kp = w.shape.k;
  auto bytes = ::lfm3::npu::PackWeights(rows, kp, w.shape.m, split, [&](std::uint32_t r, std::span<float> out) {
    row(r, out.subspan(0, cols));
    std::fill(out.begin() + cols, out.end(), 0.0f);
  });
  std::lock_guard lock(impl_->mutex);
  w.weights = impl_->npu_runtime().Alloc(bytes.size());
  constexpr std::size_t kChunk = 64u << 20;
  for (std::size_t off = 0; off < bytes.size(); off += kChunk)
    impl_->npu_runtime().Write(w.weights, off, bytes.data() + off, std::min(kChunk, bytes.size() - off));
  return w;
}

Dense::Matrix Dense::Upload(std::span<const Tensor* const> tensors, bool k_split) {
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
  }, k_split);
}

void Dense::Run(const Matrix& w, std::span<const float> x, std::uint32_t rows, std::span<float> y) {
  if (x.size() < std::size_t(rows) * w.k || y.size() < std::size_t(rows) * w.m)
    throw std::invalid_argument("NPU projection buffers too small");
  auto& m = *impl_;
  std::lock_guard lock(m.mutex);
  if (m.xrt) {
    const float* xs = x.data();
    std::vector<float> padded;
    if (w.shape.k != w.k) {  // columns padded to a multiple of 32
      padded.assign(std::size_t(rows) * w.shape.k, 0.0f);
      for (std::uint32_t r = 0; r < rows; ++r)
        std::copy_n(x.data() + std::size_t(r) * w.k, w.k, padded.data() + std::size_t(r) * w.shape.k);
      xs = padded.data();
    }
    if (m.xrt->run(m.xrt->g, w.weights, xs, int(rows), y.data()))
      throw std::runtime_error(std::string("XRT GEMM: ") + m.xrt->error());
    return;
  }
  const std::uint32_t kp = w.shape.k, mp = w.shape.m;
  const std::uint32_t parts = 1;  // K-split matrices sum their columns on chip
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
    m.npu_runtime().Write(xbuf, 0, m.xb.data(), m.xb.size() * 2);
    const int bind[] = {xbuf, w.weights, ybuf};
    m.npu_runtime().Dispatch(kernel, bind);
    const std::uint32_t used = (take + 7) / 8;
    m.yb.resize(block_out * used);
    m.npu_runtime().Read(ybuf, 0, m.yb.data(), m.yb.size() * 4);
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

::lfm3::npu::Runtime& Dense::runtime() { return impl_->npu_runtime(); }

void Dense::Enqueue(const Matrix& w, int xb, int y, std::uint32_t blocks) {
  auto& m = *impl_;
  std::lock_guard lock(m.mutex);
  for (std::uint32_t b0 = 0; b0 < blocks; b0 += kMaxBlocks)
    if (blocks > kMaxBlocks) throw std::invalid_argument("too many NPU row blocks");
  ProjectionShape s = w.shape;
  s.blocks = blocks;
  const int bind[] = {xb, w.weights, y};
  m.npu_runtime().Enqueue(m.Kernel(s), bind);
}

int Dense::Zeros(std::uint64_t bytes) {
  auto& m = *impl_;
  std::lock_guard lock(m.mutex);
  const int id = m.npu_runtime().Alloc(bytes);
  std::vector<std::uint8_t> zero(std::min<std::uint64_t>(bytes, 64u << 20), 0);
  for (std::uint64_t off = 0; off < bytes; off += zero.size())
    m.npu_runtime().Write(id, off, zero.data(), std::min<std::uint64_t>(zero.size(), bytes - off));
  return id;
}

}  // namespace gufo::models::lfm3::npu
