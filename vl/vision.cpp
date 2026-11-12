#include "vision.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <gufo/ext/gguf_identity.hpp>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

#include "../audio/hip/ops.hpp"
#include "../hip/ops.hpp"
#include "../npu/dense.hpp"
#include "../npu/host_ops.hpp"
#include "../upload.hpp"
#include "hip/ops.hpp"
namespace gufo::models::lfm3_vl {
namespace {
using core::GgmlType;
using lfm3::Tensor;
void Check(hipError_t s) {
  if (s != hipSuccess)
    throw std::runtime_error(hipGetErrorString(s));
}
struct Buffer {
  void* data{};
  Buffer() = default;
  explicit Buffer(std::size_t bytes) {
    if (bytes)
      Check(hipMalloc(&data, bytes));
  }
  ~Buffer() {
    if (data)
      (void)hipFree(data);
  }
  Buffer(Buffer&& b) noexcept : data(std::exchange(b.data, nullptr)) {}
  Buffer& operator=(Buffer&& b) noexcept {
    std::swap(data, b.data);
    return *this;
  }
  float* f() const { return static_cast<float*>(data); }
};
struct Stream {
  hipStream_t s{};
  Stream() { Check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking)); }
  ~Stream() {
    if (s)
      (void)hipStreamDestroy(s);
  }
};
struct Block {
  Tensor norm, ffn_norm, gate, up, down, conv, in, out, q, k, v, qn, kn;
};
const float* F(const Tensor& t) {
  return static_cast<const float*>(t.data);
}
// Activations of one encode, grown to the largest picture so far, up to
// kKeptPatches (a 1-megapixel picture, about 285 MiB). Larger pictures
// allocate their own: allocation is a small part of their encode.
constexpr std::size_t kKeptPatches = 4096;
struct Scratch {
  Stream stream;
  std::size_t patches{};
  Buffer x, z, o, mix, q, k, v, a, gate, up, workspace;
  void Reserve(std::size_t n) {
    if (n <= patches)
      return;
    x = Buffer(n * 1024 * 4);
    z = Buffer(n * 4096 * 4);
    o = Buffer(n * 1024 * 4);
    mix = Buffer(n * 3072 * 4);
    q = Buffer(n * 1024 * 4);
    k = Buffer(n * 512 * 4);
    v = Buffer(n * 512 * 4);
    a = Buffer(n * 1024 * 4);
    gate = Buffer(n * 2560 * 4);
    up = Buffer(n * 2560 * 4);
    workspace = Buffer(lfm3::gpu::DenseWorkspaceBytes(n, 4096));
    patches = n;
  }
};
}  // namespace
struct Vision::Impl {
  // Reused by one encode at a time; concurrent encodes allocate their own.
  // Created on the first GPU encode (none in NPU mode).
  std::mutex scratch_mutex;
  std::optional<Scratch> scratch;
  // NPU mode (the backbone's NPU bundle): products on the NPU, the rest on
  // the host; tensors keep their mapped host data. No GPU.
  bool npu{};
  std::vector<float> patch_host;
  struct NpuBlock {
    lfm3::npu::Dense::Matrix qkv, mix, out, gateup, down;
  };
  std::vector<NpuBlock> npu_blocks;
  lfm3::npu::Dense::Matrix npu_patch, npu_proj1, npu_proj2;
  std::vector<float> EncodeNpu(const Picture& p);
  std::string identity;
  std::once_flag identity_once;
  std::unique_ptr<core::GgufReader> reader;
  lfm3::DeviceWeights device;
  Buffer patch_weights;
  std::vector<Block> blocks;
  Tensor patch, norm, proj_norm, proj_bias, proj1, bias1, proj2, bias2;
  template<class F>
  void Visit(F f) {
    for (auto* t :
         {&norm, &proj_norm, &proj_bias, &proj1, &bias1, &proj2, &bias2})
      f(*t);
    for (auto& b : blocks)
      for (auto* t : {&b.norm, &b.ffn_norm, &b.gate, &b.up, &b.down, &b.conv,
                      &b.in, &b.out, &b.q, &b.k, &b.v, &b.qn, &b.kn})
        f(*t);
  }
  // Validates a mapped tensor; Load uploads every Get after the last.
  Tensor Find(const std::string& name, unsigned cols, unsigned rows) const {
    const auto* t = reader->FindTensor(name);
    std::size_t count = 1;
    if (t)
      for (auto d : t->dimensions)
        count *= d;
    if (!t || count != std::size_t(cols) * rows ||
        (t->type != GgmlType::kF32 && t->type != GgmlType::kF16 &&
         t->type != GgmlType::kBF16 && t->type != GgmlType::kQ8_0))
      throw std::runtime_error("invalid or unsupported LFM3 vision tensor: " +
                               name);
    const auto bytes = quant::EncodedSizeBytes(t->type, count);
    if (!bytes || bytes != t->size_bytes ||
        (t->type == GgmlType::kQ8_0 && cols % 32))
      throw std::runtime_error("invalid d1 vision tensor bytes: " + name);
    return {t->data, t->type, cols, rows, bytes};
  }
  Tensor Get(const std::string& name, unsigned cols, unsigned rows) {
    auto t = Find(name, cols, rows);
    device.Add(t);
    return t;
  }
  Tensor Norm(const std::string& name, unsigned cols) {
    auto t = Get(name, cols, 1);
    if (t.type != GgmlType::kF32)
      throw std::runtime_error("vision norm must be F32: " + name);
    return t;
  }
};
Vision::Vision() : impl_(std::make_unique<Impl>()) {}
Vision::~Vision() = default;
std::unique_ptr<Vision> Vision::Load(const std::string& path,
                                     const std::string& npu_bundle) {
  auto v = std::unique_ptr<Vision>(new Vision);
  auto& m = *v->impl_;
  std::string error;
  m.reader = core::GgufReader::OpenFile(path, &error);
  if (!m.reader)
    throw std::runtime_error(error);
  auto& r = *m.reader;
  if (r.GetMetadataString("clip.vision.projector_type") != "lfm3vl" ||
      r.GetMetadataUint32("clip.vision.embedding_length") != 1024 ||
      r.GetMetadataUint32("clip.vision.block_count") != 14 ||
      r.GetMetadataUint32("clip.vision.attention.head_count") != 16 ||
      r.GetMetadataUint32("clip.vision.attention.head_count_kv") != 8 ||
      r.GetMetadataUint32("clip.vision.patch_size") != 16 ||
      r.GetMetadataUint32("clip.vision.spatial_merge_size") != 2 ||
      r.GetMetadataUint32("clip.vision.projection_dim") != 2048)
    throw std::runtime_error("unsupported LFM3 vision geometry");
  // Static images repeat their frame twice. Add the two temporal slices of
  // the patchifier in F32, retaining their exact stored F16 values.
  std::vector<float> patch(1024 * 768), row(768), second(768);
  const auto p = m.Find("v.patch_embd.weight", 768, 1024);
  const auto pnext = m.Find("v.patch_embd.weight.1", 768, 1024);
  for (int i = 0; i < 1024; ++i) {
    lfm3::DecodeRow(p, i, row);
    lfm3::DecodeRow(pnext, i, second);
    for (int j = 0; j < 768; ++j)
      patch[i * 768 + j] = row[j] + second[j];
  }
  m.npu = !npu_bundle.empty();
  if (m.npu)
    lfm3::npu::Dense::Open(npu_bundle);
  if (m.npu) {
    m.patch_host = patch;
    m.patch = {m.patch_host.data(), GgmlType::kF32, 768, 1024,
               patch.size() * 4};
  } else {
    m.patch_weights = Buffer(patch.size() * 4);
    Check(hipMemcpy(m.patch_weights.data, patch.data(), patch.size() * 4,
                    hipMemcpyHostToDevice));
    m.patch = {m.patch_weights.data, GgmlType::kF32, 768, 1024,
               patch.size() * 4};
  }
  m.norm = m.Norm("v.embedding_norm.weight", 1024);
  for (int i = 0; i < 14; ++i) {
    const auto prefix = "v.blk." + std::to_string(i) + ".";
    Block b;
    b.norm = m.Norm(prefix + "ln1.weight", 1024);
    b.ffn_norm = m.Norm(prefix + "ln2.weight", 1024);
    b.gate = m.Get(prefix + "ffn_gate.weight", 1024, 2560);
    b.up = m.Get(prefix + "ffn_up.weight", 1024, 2560);
    b.down = m.Get(prefix + "ffn_down.weight", 2560, 1024);
    if (r.FindTensor(prefix + "attn_q.weight")) {
      b.q = m.Get(prefix + "attn_q.weight", 1024, 1024);
      b.k = m.Get(prefix + "attn_k.weight", 1024, 512);
      b.v = m.Get(prefix + "attn_v.weight", 1024, 512);
      b.qn = m.Norm(prefix + "attn_q_norm.weight", 64);
      b.kn = m.Norm(prefix + "attn_k_norm.weight", 64);
      b.out = m.Get(prefix + "attn_out.weight", 1024, 1024);
    } else {
      b.conv = m.Get(prefix + "conv2d_spatial.weight", 9, 1024);
      if (b.conv.type != GgmlType::kF32)
        throw std::runtime_error("vision convolution must be F32");
      b.in = m.Get(prefix + "conv2d_in_proj.weight", 1024, 3072);
      b.out = m.Get(prefix + "conv2d_out_proj.weight", 1024, 1024);
    }
    m.blocks.push_back(b);
  }
  m.proj_norm = m.Norm("mm.input_norm.weight", 1024);
  m.proj_bias = m.Norm("mm.input_norm.bias", 1024);
  m.proj1 = m.Get("mm.0.weight", 4096, 4096);
  m.bias1 = m.Norm("mm.0.bias", 4096);
  m.proj2 = m.Get("mm.2.weight", 4096, 2048);
  m.bias2 = m.Norm("mm.2.bias", 2048);
  if (m.npu) {
    auto& dense = lfm3::npu::Dense::Shared();
    m.npu_patch =
        dense.Upload(1024, 768, [&](std::uint32_t r, std::span<float> out) {
          std::copy_n(m.patch_host.data() + std::size_t(r) * 768, 768,
                      out.begin());
        });
    for (const auto& b : m.blocks) {
      Impl::NpuBlock nb;
      if (b.q.data) {
        const Tensor* qkv[] = {&b.q, &b.k, &b.v};
        nb.qkv = dense.Upload(qkv);
      } else {
        const Tensor* in[] = {&b.in};
        nb.mix = dense.Upload(in);
      }
      const Tensor* out[] = {&b.out};
      const Tensor* gateup[] = {&b.gate, &b.up};
      const Tensor* down[] = {&b.down};
      nb.out = dense.Upload(out);
      nb.gateup = dense.Upload(gateup);
      nb.down = dense.Upload(down);
      m.npu_blocks.push_back(nb);
    }
    const Tensor* p1[] = {&m.proj1};
    const Tensor* p2[] = {&m.proj2};
    m.npu_proj1 = dense.Upload(p1);
    m.npu_proj2 = dense.Upload(p2);
    return v;
  }
  m.device.Upload(r);
  m.Visit([&](Tensor& t) { m.device.Resolve(t); });
  return v;
}
std::shared_ptr<Vision> LoadMatchingVision(const std::string& backbone,
                                           const std::string& mmproj,
                                           const std::string& npu_bundle) {
  std::string error;
  auto reader = core::GgufReader::OpenFile(backbone, &error);
  if (!reader)
    throw std::runtime_error(error);
  const bool vl = reader->GetMetadataString("general.basename") == "LFM3-VL";
  if (vl != !mmproj.empty())
    throw std::invalid_argument(
        vl ? "LFM3-VL requires its matching --mmproj GGUF"
           : "a vision projector requires an LFM3-VL backbone");
  if (!vl)
    return nullptr;
  auto projector = core::GgufReader::OpenFile(mmproj, &error);
  if (!projector ||
      projector->GetMetadataString("general.basename") != "LFM3-VL")
    throw std::invalid_argument("--mmproj does not match the LFM3-VL backbone");
  return Vision::Load(mmproj, npu_bundle);
}
const std::string& Vision::Identity() const {
  std::call_once(impl_->identity_once, [&] {
    impl_->identity = core::GgufIdentityHex(*impl_->reader);
  });
  return impl_->identity;
}
std::vector<float> Vision::Encode(const Picture& p) {
  if (p.height <= 0 || p.width <= 0 || p.height % 2 || p.width % 2 ||
      std::size_t(p.height) * p.width > 65536 ||
      p.patches.size() != std::size_t(p.height) * p.width * 768)
    throw std::invalid_argument("invalid LFM3 picture patches");
  const int n = p.height * p.width;
  auto& m = *impl_;
  if (m.npu)
    return m.EncodeNpu(p);
  std::unique_lock lock(m.scratch_mutex, std::defer_lock);
  if (std::size_t(n) <= kKeptPatches)
    (void)lock.try_lock();
  std::optional<Scratch> own;
  auto& scratch =
      lock ? (m.scratch ? *m.scratch : m.scratch.emplace()) : own.emplace();
  scratch.Reserve(n);
  const auto s = scratch.stream.s;
  auto &x = scratch.x, &z = scratch.z, &o = scratch.o, &mix = scratch.mix,
       &q = scratch.q, &k = scratch.k, &v = scratch.v, &a = scratch.a,
       &gate = scratch.gate, &up = scratch.up, &workspace = scratch.workspace;
  auto dense = [&](const Tensor& t, const float* in, float* out, int rows) {
    if (t.type == GgmlType::kBF16)
      lfm3_audio::gpu::SgemmBf16(in, t.data, nullptr, out, rows, t.rows, t.cols,
                                 lfm3_audio::gpu::Activation::kNone, s);
    else if (t.type == GgmlType::kF32)
      // The summed patchifier: one tiled FP32 launch, not a GEMV per patch.
      lfm3_audio::gpu::Sgemm(in, F(t), nullptr, out, rows, t.rows, t.cols,
                             lfm3_audio::gpu::Activation::kNone, s);
    else
      lfm3::gpu::Dense(t, in, out, rows, s, {workspace.data});
  };
  auto norm = [&](const float* in, const Tensor& weight, float* out) {
    lfm3::gpu::LaunchBatchedRMSNorm(in, F(weight), out, nullptr, n, 1024, 1e-5F,
                                    s);
  };
  auto residual = [&] {
    lfm3::gpu::LaunchBatchedResidualAdd(x.f(), o.f(), x.f(), n, 1024, s);
  };
  Check(hipMemcpyAsync(z.data, p.patches.data(), p.patches.size() * 4,
                       hipMemcpyHostToDevice, s));
  dense(m.patch, z.f(), x.f(), n);
  norm(x.f(), m.norm, x.f());
  for (const auto& b : m.blocks) {
    norm(x.f(), b.norm, z.f());
    if (b.q.data) {
      dense(b.q, z.f(), q.f(), n);
      dense(b.k, z.f(), k.f(), n);
      dense(b.v, z.f(), v.f(), n);
      lfm3::gpu::LaunchBatchedPerHeadRMSNorm(q.f(), F(b.qn), q.f(), n, 16, 64,
                                             1e-5F, s);
      lfm3::gpu::LaunchBatchedPerHeadRMSNorm(k.f(), F(b.kn), k.f(), n, 8, 64,
                                             1e-5F, s);
      gpu::AxialRope(q.f(), p.height, p.width, 16, s);
      gpu::AxialRope(k.f(), p.height, p.width, 8, s);
      // The tower keeps no cache: K and V stay FP32.
      gpu::Attention64(q.f(), k.f(), v.f(), a.f(), n, 16, 8, s);
    } else {
      dense(b.in, z.f(), mix.f(), n);
      gpu::SpatialConv(mix.f(), F(b.conv), a.f(), p.height, p.width, 1024, s);
    }
    dense(b.out, a.f(), o.f(), n);
    residual();
    norm(x.f(), b.ffn_norm, z.f());
    dense(b.gate, z.f(), gate.f(), n);
    dense(b.up, z.f(), up.f(), n);
    lfm3_audio::gpu::SwiGlu(gate.f(), up.f(), gate.f(), std::size_t(n) * 2560,
                            s);
    dense(b.down, gate.f(), o.f(), n);
    residual();
  }
  lfm3_audio::gpu::LayerNorm(x.f(), F(m.proj_norm), F(m.proj_bias), z.f(), n,
                             1024, 1e-6F, s);
  dense(m.proj1, z.f(), mix.f(), n / 4);
  lfm3_audio::gpu::BiasAct(mix.f(), F(m.bias1), n / 4, 4096,
                           lfm3_audio::gpu::Activation::kGelu, s);
  dense(m.proj2, mix.f(), o.f(), n / 4);
  lfm3_audio::gpu::BiasAct(o.f(), F(m.bias2), n / 4, 2048,
                           lfm3_audio::gpu::Activation::kNone, s);
  std::vector<float> out(std::size_t(n / 4) * 2048);
  Check(hipMemcpyAsync(out.data(), o.data, out.size() * 4,
                       hipMemcpyDeviceToHost, s));
  Check(hipStreamSynchronize(s));
  return out;
}
std::vector<float> Vision::Impl::EncodeNpu(const Picture& p) {
  namespace n = lfm3::npu;
  auto& dense = n::Dense::Shared();
  const std::size_t N = std::size_t(p.height) * p.width, H = 1024;
  const int w = p.width, h = p.height;
  // Grid position of a patch row in the 2x2-merge order, and back.
  auto grid = [&](std::size_t row, int& y, int& x) {
    y = int(row / 4) / (w / 2) * 2 + int(row % 4) / 2;
    x = int(row / 4) % (w / 2) * 2 + int(row % 2);
  };
  auto index = [&](int y, int x) {
    return std::size_t(((y / 2) * (w / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2);
  };
  std::vector<float> x(N * H), z(N * H), o(N * H), a(N * H);
  dense.Run(npu_patch, p.patches, std::uint32_t(N), x);
  n::RmsNormRows(x.data(), N, H, F(norm), 1e-5f, x.data());
  for (std::size_t bi = 0; bi < blocks.size(); ++bi) {
    const auto& b = blocks[bi];
    const auto& nb = npu_blocks[bi];
    n::RmsNormRows(x.data(), N, H, F(b.norm), 1e-5f, z.data());
    if (b.q.data) {
      constexpr std::uint32_t kHeads = 16, kKv = 8, kDim = 64, kQkv = 2048;
      std::vector<float> qkv(N * kQkv);
      dense.Run(nb.qkv, z, std::uint32_t(N), qkv);
      for (std::size_t r = 0; r < N; ++r) {
        float* q = &qkv[r * kQkv];
        float* k = q + kHeads * kDim;
        for (std::uint32_t hh = 0; hh < kHeads; ++hh)
          n::RmsNorm(q + hh * kDim, kDim, F(b.qn), 1e-5f);
        for (std::uint32_t hh = 0; hh < kKv; ++hh)
          n::RmsNorm(k + hh * kDim, kDim, F(b.kn), 1e-5f);
        int y, xx;
        grid(r, y, xx);
        for (float* v : {q, k})
          for (std::uint32_t hh = 0; hh < (v == q ? kHeads : kKv); ++hh)
            for (int d = 0; d < 32; ++d) {
              const float angle =
                  float(d < 16 ? y : xx) * std::pow(1e6f, -float(d % 16) / 16);
              float* at = v + hh * kDim + d;
              const float c = at[0], s = at[32];
              at[0] = c * std::cos(angle) - s * std::sin(angle);
              at[32] = s * std::cos(angle) + c * std::sin(angle);
            }
      }
      n::Attention(
          qkv.data(), qkv.data() + kHeads * kDim,
          qkv.data() + kHeads * kDim + kKv * kDim, a.data(), N, kHeads, kKv,
          kDim, [](std::size_t, std::size_t) { return true; }, nullptr, 0, kQkv,
          kQkv);
    } else {
      std::vector<float> mix(N * 3 * H);
      dense.Run(nb.mix, z, std::uint32_t(N), mix);
      const float* wc = F(b.conv);
      lfm3::npu::Parallel(N, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
          int y, xx;
          grid(r, y, xx);
          for (std::size_t c = 0; c < H; ++c) {
            float sum = 0;
            for (int dy = -1; dy <= 1; ++dy)
              for (int dx = -1; dx <= 1; ++dx)
                if (y + dy >= 0 && y + dy < h && xx + dx >= 0 && xx + dx < w) {
                  const std::size_t at = index(y + dy, xx + dx) * H * 3 + c;
                  sum = std::fma(mix[at] * mix[at + 2 * H],
                                 wc[c * 9 + (dy + 1) * 3 + dx + 1], sum);
                }
            a[r * H + c] = mix[r * 3 * H + H + c] * sum;
          }
        }
      });
    }
    dense.Run(nb.out, a, std::uint32_t(N), o);
    for (std::size_t i = 0; i < N * H; ++i)
      x[i] += o[i];
    n::RmsNormRows(x.data(), N, H, F(b.ffn_norm), 1e-5f, z.data());
    std::vector<float> gu(N * 5120), act(N * 2560);
    dense.Run(nb.gateup, z, std::uint32_t(N), gu);
    for (std::size_t r = 0; r < N; ++r)
      for (std::size_t i = 0; i < 2560; ++i)
        act[r * 2560 + i] = n::Silu(gu[r * 5120 + i]) * gu[r * 5120 + 2560 + i];
    dense.Run(nb.down, act, std::uint32_t(N), o);
    for (std::size_t i = 0; i < N * H; ++i)
      x[i] += o[i];
  }
  n::LayerNormRows(x.data(), N, H, F(proj_norm), F(proj_bias), 1e-6f, z.data());
  const std::size_t M = N / 4;
  std::vector<float> mix(M * 4096), out(M * 2048);
  dense.Run(npu_proj1, z, std::uint32_t(M), mix);
  for (std::size_t r = 0; r < M; ++r)
    for (std::size_t i = 0; i < 4096; ++i)
      mix[r * 4096 + i] = n::Gelu(mix[r * 4096 + i] + F(bias1)[i]);
  dense.Run(npu_proj2, mix, std::uint32_t(M), out);
  for (std::size_t r = 0; r < M; ++r)
    for (std::size_t i = 0; i < 2048; ++i)
      out[r * 2048 + i] += F(bias2)[i];
  return out;
}
}  // namespace gufo::models::lfm3_vl
