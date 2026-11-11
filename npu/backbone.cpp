#include "backbone.hpp"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>
#include <string_view>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "attention.hpp"
#include "dense.hpp"
#include "glue.hpp"
#include "host/npu_runtime.hpp"
#include "host_ops.hpp"
#include "host/bfp.hpp"

namespace gufo::models::lfm3::npu {
namespace {

const float* Floats(const Tensor& t) { return static_cast<const float*>(t.data); }

void Norm(float* x, std::size_t n, const float* weight, float eps) {
  float sum = 0;
  for (std::size_t i = 0; i < n; ++i) sum += x[i] * x[i];
  const float scale = 1 / std::sqrt(sum / n + eps);
  for (std::size_t i = 0; i < n; ++i) x[i] = x[i] * scale * weight[i];
}

}  // namespace

void State::Reset() {
  position = 0;
  for (auto& l : layers) {
    std::fill(l.conv.begin(), l.conv.end(), 0.0f);
    std::fill(l.k.begin(), l.k.end(), _Float16(0));
    std::fill(l.v.begin(), l.v.end(), _Float16(0));
  }
}

std::uint32_t Pow2(std::uint32_t v) {
  std::uint32_t p = 1;
  while (p < v) p *= 2;
  return p;
}

// The forward with activations, K/V and convolution state on the NPU.
struct Device {
  static constexpr std::uint32_t kSlots = 8, kMaxRows = 512;
  const Weights* w{};
  Dense* dense{};
  ::lfm3::npu::Runtime* npu{};
  std::mutex mutex;
  std::uint32_t context{}, cap_global{}, cap_sliding{}, table_rows{};
  int next_slot{};
  // Per-layer device buffers.
  std::vector<int> w_op, w_ffn, conv_w, qn, kn, hist, kpool, vpool;
  int w_first{-1}, cos_g{-1}, sin_g{-1}, cos_s{-1}, sin_s{-1};
  // Activations for up to kMaxRows rows.
  int x{-1}, x2{-1}, o{-1}, z{-1}, xb2048{-1}, xb_ffn{-1}, y_proj{-1}, y_out{-1}, y_down{-1}, y_head{-1},
      q{-1}, gates{-1}, meta_g{-1}, meta_s{-1}, groups{-1};
  std::map<std::string, int> kernels;

  int Kernel(const std::string& key, const std::function<std::string()>& program, const std::string& root) {
    if (auto it = kernels.find(key); it != kernels.end()) return it->second;
    int id;
    try {
      id = npu->Compile(program(), key + ".loom", root);
    } catch (const std::exception& e) {
      throw std::runtime_error(key + ": " + e.what());
    }
    kernels.emplace(key, id);
    return id;
  }
  int Row(const RowKernel& k) {
    return Kernel(k.name, [&] { return RowKernelProgram(k); }, "@" + k.name);
  }
  int Upload(const std::vector<float>& v) {
    const int id = npu->Alloc(v.size() * 4);
    npu->Write(id, 0, v.data(), v.size() * 4);
    return id;
  }
};

struct Backbone::Impl {
  const Weights* w{};
  Dense* dense{};
  std::unique_ptr<Device> device;
  void ForwardDevice(std::span<const float> inputs, std::span<const Row> rows, std::span<float> hidden,
                     std::span<float> logits, bool decision, std::span<const std::int32_t> groups);
  struct Layer {
    Dense::Matrix in, out, qkvg, gateup, down;
  };
  std::vector<Layer> layers;
  Dense::Matrix head;

  Dense::Matrix Make(std::span<const Tensor* const> tensors) { return dense->Upload(tensors); }
  void Run(const Dense::Matrix& p, std::span<const float> x, std::uint32_t n, std::span<float> y) {
    dense->Run(p, x, n, y);
  }
};

Backbone::Backbone() : impl_(std::make_unique<Impl>()) {}
Backbone::~Backbone() = default;
const Config& Backbone::config() const { return impl_->w->config; }

std::unique_ptr<Backbone> Backbone::Load(const Weights& w, std::string* error) {
  try {
    const auto& c = w.config;
    auto b = std::unique_ptr<Backbone>(new Backbone);
    auto& m = *b->impl_;
    m.w = &w;
    m.dense = &Dense::Shared();
    m.layers.resize(c.layers.size());
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
      const auto& l = w.layers[i];
      auto& p = m.layers[i];
      if (c.layers[i] == LayerType::kConvolution) {
        const Tensor* in[] = {&l.in_proj};
        const Tensor* out[] = {&l.out_proj};
        p.in = m.Make(in);
        p.out = m.Make(out);
      } else {
        const Tensor* qkvg[] = {&l.q, &l.k, &l.v, &l.attn_gate};
        const Tensor* out[] = {&l.attn_out};
        p.qkvg = m.Make(qkvg);
        p.out = m.Make(out);
      }
      const Tensor* gateup[] = {&l.gate, &l.up};
      const Tensor* down[] = {&l.down};
      p.gateup = m.Make(gateup);
      p.down = m.Make(down);
    }
    const Tensor* head[] = {&w.output};
    m.head = m.Make(head);
    const char* glue = std::getenv("GUFO_LFM3_NPU_GLUE");
    if (!(glue && std::string_view(glue) == "host")) {
      auto d = std::make_unique<Device>();
      d->w = &w;
      d->dense = m.dense;
      d->npu = &m.dense->runtime();
      const std::uint32_t H = c.hidden, L = c.conv_width, D = c.head_dim;
      const std::size_t n = c.layers.size();
      d->w_op.assign(n, -1), d->w_ffn.assign(n, -1), d->conv_w.assign(n, -1), d->qn.assign(n, -1),
          d->kn.assign(n, -1), d->hist.assign(n, -1), d->kpool.assign(n, -1), d->vpool.assign(n, -1);
      auto pair = [&](const Tensor& a, const Tensor& b) {
        std::vector<float> v(2 * H);
        std::copy_n(Floats(a), H, v.begin());
        std::copy_n(Floats(b), H, v.begin() + H);
        return d->Upload(v);
      };
      d->w_first = d->Upload(std::vector<float>(Floats(w.layers[0].norm), Floats(w.layers[0].norm) + H));
      for (std::size_t i = 0; i < n; ++i) {
        const auto& l = w.layers[i];
        if (!l.post_norm.data || !l.post_ffn_norm.data)
          throw std::runtime_error("the NPU forward expects LFM3 sandwich norms");
        d->w_op[i] = pair(l.post_norm, l.ffn_norm);
        d->w_ffn[i] = pair(l.post_ffn_norm, i + 1 < n ? w.layers[i + 1].norm : w.output_norm);
        if (c.layers[i] == LayerType::kConvolution) {
          std::vector<float> tw(std::size_t(L) * H);
          for (std::uint32_t ch = 0; ch < H; ++ch)
            for (std::uint32_t tp = 0; tp < L; ++tp) tw[tp * H + ch] = Floats(l.conv)[ch * L + tp];
          d->conv_w[i] = d->Upload(tw);
        } else {
          d->qn[i] = d->Upload(std::vector<float>(Floats(l.q_norm), Floats(l.q_norm) + D));
          d->kn[i] = d->Upload(std::vector<float>(Floats(l.k_norm), Floats(l.k_norm) + D));
        }
      }
      const std::uint64_t R = Device::kMaxRows, B = R / 8, pad = 4096;
      auto& npu = *d->npu;
      d->x = npu.Alloc(R * H * 4), d->x2 = npu.Alloc(R * H * 4), d->o = npu.Alloc(R * H * 4), d->z = npu.Alloc(R * H * 4);
      d->xb2048 = npu.Alloc(B * H * 8 * 2 + pad);
      d->xb_ffn = npu.Alloc(B * c.intermediate * 8 * 2 + pad);
      d->y_proj = npu.Alloc(B * 2 * c.intermediate * 8 * 4 + pad * 4);
      d->y_out = npu.Alloc(B * H * 8 * 4 + pad * 4);
      d->y_down = npu.Alloc(B * 8 * H * 8 * 4 + pad * 4 * 8);
      d->y_head = npu.Alloc(8ull * m.head.shape.m * 8 * 4);
      d->q = npu.Alloc(R * c.heads * D * 4), d->gates = npu.Alloc(R * 16 * 4);
      d->meta_g = npu.Alloc(R * 64), d->meta_s = npu.Alloc(R * 64), d->groups = npu.Alloc(R * 4 + 256);
      m.device = std::move(d);
    }
    return b;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return nullptr;
  }
}

State Backbone::NewState(std::uint32_t context) const {
  const auto& c = impl_->w->config;
  State s;
  if (auto* d = impl_->device.get()) {
    std::lock_guard lock(d->mutex);
    if (!d->context) {
      // Pools for the first session's context; later sessions must fit it.
      d->context = context;
      d->cap_global = std::max(32u, Pow2(context));
      // A pass writes up to kMaxRows keys before its rows attend: the
      // sliding ring keeps them beside the window it still reads.
      d->cap_sliding = Pow2(c.window + Device::kMaxRows);
      d->table_rows = d->cap_global;
      const std::uint32_t D = c.head_dim, H = c.hidden, half = D / 2;
      std::vector<float> cg(std::size_t(d->table_rows) * half), sg(cg.size()), cs(cg.size()), ss(cg.size());
      for (std::uint32_t p = 0; p < d->table_rows; ++p)
        for (std::uint32_t i = 0; i < half; ++i) {
          const float ag = p * std::pow(c.theta, -2.0F * i / D) / Floats(impl_->w->rope_freqs)[i];
          const float as = p * std::pow(c.theta_swa, -2.0F * i / D);
          cg[p * half + i] = std::cos(ag), sg[p * half + i] = std::sin(ag);
          cs[p * half + i] = std::cos(as), ss[p * half + i] = std::sin(as);
        }
      d->cos_g = d->Upload(cg), d->sin_g = d->Upload(sg), d->cos_s = d->Upload(cs), d->sin_s = d->Upload(ss);
      for (std::size_t i = 0; i < c.layers.size(); ++i) {
        if (c.layers[i] == LayerType::kConvolution) {
          d->hist[i] = d->dense->Zeros(std::uint64_t(Device::kSlots) * (c.conv_width - 1) * H * 4);
        } else {
          const std::uint32_t cap = c.layers[i] == LayerType::kSlidingAttention ? d->cap_sliding : d->cap_global;
          const std::uint64_t bytes = std::uint64_t(Device::kSlots) * cap * c.kv_heads[i] * D * 2;
          d->kpool[i] = d->dense->Zeros(bytes);
          d->vpool[i] = d->dense->Zeros(bytes);
        }
      }
    }
    if (context > d->context) throw std::runtime_error("NPU session context exceeds the first session's");
    // The last slot takes the padding rows of short passes.
    s.slot = d->next_slot++ % int(Device::kSlots - 1);
  }
  s.layers.resize(c.layers.size());
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    auto& l = s.layers[i];
    if (c.layers[i] == LayerType::kConvolution) {
      l.conv.assign(std::size_t(c.hidden) * (c.conv_width - 1), 0.0f);
    } else {
      l.capacity = c.layers[i] == LayerType::kSlidingAttention
                       ? std::min(context, c.window)
                       : context;
      l.k.assign(std::size_t(l.capacity) * c.kv_heads[i] * c.head_dim, _Float16(0));
      l.v.assign(l.k.size(), _Float16(0));
    }
  }
  return s;
}

void Backbone::SyncToHost(State& s) {
  auto* d = impl_->device.get();
  if (!d || !s.host_stale || s.slot < 0) return;
  std::lock_guard lock(d->mutex);
  const auto& c = impl_->w->config;
  const std::uint32_t H = c.hidden, L = c.conv_width, D = c.head_dim;
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    auto& l = s.layers[i];
    if (c.layers[i] == LayerType::kConvolution) {
      std::vector<float> tap(std::size_t(L - 1) * H);
      d->npu->Read(d->hist[i], std::uint64_t(s.slot) * tap.size() * 4, tap.data(), tap.size() * 4);
      for (std::uint32_t ch = 0; ch < H; ++ch)
        for (std::uint32_t tp = 0; tp + 1 < L; ++tp) l.conv[ch * (L - 1) + tp] = tap[tp * H + ch];
    } else {
      const std::uint32_t cap = c.layers[i] == LayerType::kSlidingAttention ? d->cap_sliding : d->cap_global;
      const std::uint64_t row = std::uint64_t(c.kv_heads[i]) * D * 2;
      // Host rings hold position t at t % capacity, device rings at t % cap.
      std::vector<_Float16> dk(std::size_t(cap) * row / 2), dv(dk.size());
      d->npu->Read(d->kpool[i], std::uint64_t(s.slot) * cap * row, dk.data(), cap * row);
      d->npu->Read(d->vpool[i], std::uint64_t(s.slot) * cap * row, dv.data(), cap * row);
      const std::uint32_t lo = s.position > l.capacity ? s.position - l.capacity : 0;
      const std::size_t e = row / 2;
      for (std::uint32_t tpos = lo; tpos < s.position; ++tpos) {
        std::copy_n(dk.begin() + std::size_t(tpos % cap) * e, e, l.k.begin() + std::size_t(tpos % l.capacity) * e);
        std::copy_n(dv.begin() + std::size_t(tpos % cap) * e, e, l.v.begin() + std::size_t(tpos % l.capacity) * e);
      }
    }
  }
  s.host_stale = false;
}

void Backbone::SyncToDevice(State& s) {
  auto* d = impl_->device.get();
  if (!d || s.slot < 0) return;
  std::lock_guard lock(d->mutex);
  const auto& c = impl_->w->config;
  const std::uint32_t H = c.hidden, L = c.conv_width, D = c.head_dim;
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    const auto& l = s.layers[i];
    if (c.layers[i] == LayerType::kConvolution) {
      std::vector<float> tap(std::size_t(L - 1) * H);
      for (std::uint32_t ch = 0; ch < H; ++ch)
        for (std::uint32_t tp = 0; tp + 1 < L; ++tp) tap[tp * H + ch] = l.conv[ch * (L - 1) + tp];
      d->npu->Write(d->hist[i], std::uint64_t(s.slot) * tap.size() * 4, tap.data(), tap.size() * 4);
    } else {
      const std::uint32_t cap = c.layers[i] == LayerType::kSlidingAttention ? d->cap_sliding : d->cap_global;
      const std::uint64_t row = std::uint64_t(c.kv_heads[i]) * D * 2;
      std::vector<_Float16> dk(std::size_t(cap) * row / 2, _Float16(0)), dv(dk.size(), _Float16(0));
      const std::uint32_t lo = s.position > l.capacity ? s.position - l.capacity : 0;
      const std::size_t e = row / 2;
      for (std::uint32_t tpos = lo; tpos < s.position; ++tpos) {
        std::copy_n(l.k.begin() + std::size_t(tpos % l.capacity) * e, e, dk.begin() + std::size_t(tpos % cap) * e);
        std::copy_n(l.v.begin() + std::size_t(tpos % l.capacity) * e, e, dv.begin() + std::size_t(tpos % cap) * e);
      }
      if (s.position) {
        d->npu->Write(d->kpool[i], std::uint64_t(s.slot) * cap * row, dk.data(), cap * row);
        d->npu->Write(d->vpool[i], std::uint64_t(s.slot) * cap * row, dv.data(), cap * row);
      }
    }
  }
  s.host_stale = false;
}

void Backbone::Head(std::span<const float> hidden, std::uint32_t n, std::span<float> logits) {
  impl_->Run(impl_->head, hidden, n, logits);
}

void Backbone::Embed(std::uint32_t token, std::span<float> row) const {
  DecodeRow(impl_->w->embedding, token, row);
}

void Backbone::Forward(std::span<const float> inputs, std::span<const Row> rows,
                       std::span<float> hidden, std::span<float> logits, bool decision,
                       std::span<const std::int32_t> groups) {
  auto& m = *impl_;
  if (m.device) return m.ForwardDevice(inputs, rows, hidden, logits, decision, groups);
  const auto& w = *m.w;
  const auto& c = w.config;
  const std::uint32_t H = c.hidden, D = c.head_dim;
  std::size_t logit_row = 0;
  // A decision prompt is one pass: its full-attention rows see each other.
  const std::size_t pass = decision ? std::max<std::size_t>(rows.size(), 1) : kRows;
  for (std::size_t base = 0; base < rows.size(); base += pass) {
    const std::uint32_t n = std::uint32_t(std::min<std::size_t>(pass, rows.size() - base));
    const auto chunk = rows.subspan(base, n);
    std::vector<float> x(inputs.begin() + base * H, inputs.begin() + (base + n) * H);
    std::vector<float> z(std::size_t(n) * H), out(std::size_t(n) * H);
    const char* dbg = std::getenv("GUFO_LFM3_NPU_DEBUG_LAYERS");
    const std::size_t nl = dbg ? std::size_t(std::atoi(dbg)) : c.layers.size();
    for (std::size_t il = 0; il < nl; ++il) {
      const auto& l = w.layers[il];
      const auto& p = m.layers[il];
      z = x;
      for (std::uint32_t r = 0; r < n; ++r) Norm(&z[r * H], H, Floats(l.norm), c.eps);
      if (c.layers[il] == LayerType::kConvolution) {
        std::vector<float> proj(std::size_t(n) * 3 * H), conv(std::size_t(n) * H);
        m.Run(p.in, z, n, proj);
        const float* weight = Floats(l.conv);
        // Rows of one state advance its history in order.
        for (std::uint32_t r = 0; r < n; ++r) {
          auto& history = chunk[r].state->layers[il].conv;
          const float* pr = &proj[std::size_t(r) * 3 * H];
          for (std::uint32_t ch = 0; ch < H; ++ch) {
            float* hist = history.data() + std::size_t(ch) * (c.conv_width - 1);
            const float bx = pr[ch] * pr[2 * H + ch];
            float sum = 0;
            for (std::uint32_t t = 0; t < c.conv_width; ++t)
              sum += weight[ch * c.conv_width + t] * (t + 1 == c.conv_width ? bx : hist[t]);
            for (std::uint32_t t = 1; t < c.conv_width - 1; ++t) hist[t - 1] = hist[t];
            if (c.conv_width > 1) hist[c.conv_width - 2] = bx;
            conv[std::size_t(r) * H + ch] = sum * pr[H + ch];
          }
        }
        m.Run(p.out, conv, n, out);
      } else {
        const std::uint32_t kv = c.kv_heads[il], KV = kv * D;
        const std::uint32_t qkvg = H + 2 * KV + c.heads;
        std::vector<float> proj(std::size_t(n) * qkvg), att(std::size_t(n) * H);
        m.Run(p.qkvg, z, n, proj);
        const bool swa = c.layers[il] == LayerType::kSlidingAttention;
        const float theta = swa ? c.theta_swa : c.theta;
        auto rope = [&](float* v, std::uint32_t heads, std::uint32_t position) {
          for (std::uint32_t h = 0; h < heads; ++h)
            for (std::uint32_t i = 0; i < D / 2; ++i) {
              float angle = position * std::pow(theta, -2.0F * i / D);
              if (!swa) angle /= Floats(w.rope_freqs)[i];
              float a = v[h * D + i], b = v[h * D + i + D / 2];
              v[h * D + i] = a * std::cos(angle) - b * std::sin(angle);
              v[h * D + i + D / 2] = b * std::cos(angle) + a * std::sin(angle);
            }
        };
        // Rows advance in order: each stores its K/V, then attends. A ring
        // of the window's size then never loses a key a later row needs.
        // Decision full-attention layers attend once all rows are stored.
        const bool bidirectional = decision && !swa;
        for (std::uint32_t r = 0; r < n; ++r) {
          float* q = &proj[std::size_t(r) * qkvg];
          float* k = q + H;
          float* v = k + KV;
          const float* gate = q + H + 2 * KV;
          for (std::uint32_t h = 0; h < c.heads; ++h) Norm(q + h * D, D, Floats(l.q_norm), c.eps);
          for (std::uint32_t h = 0; h < kv; ++h) Norm(k + h * D, D, Floats(l.k_norm), c.eps);
          rope(q, c.heads, chunk[r].position);
          rope(k, kv, chunk[r].position);
          auto& s = chunk[r].state->layers[il];
          const std::size_t slot = chunk[r].position % s.capacity;
          for (std::uint32_t i = 0; i < KV; ++i) {
            s.k[slot * KV + i] = static_cast<_Float16>(k[i]);
            s.v[slot * KV + i] = static_cast<_Float16>(v[i]);
          }
          if (bidirectional)
            continue;
          const std::uint32_t pos = chunk[r].position;
          const std::uint32_t first = swa && pos + 1 > c.window ? pos + 1 - c.window : 0;
          auto head = [&](std::uint32_t h, std::vector<float>& scores) {
            const float* qh = q + h * D;
            const std::uint32_t kh = h / (c.heads / kv);
            scores.resize(pos + 1 - first);
            float max = -std::numeric_limits<float>::infinity();
            for (std::uint32_t t = first; t <= pos; ++t) {
              const _Float16* key = s.k.data() + std::size_t(t % s.capacity) * KV + kh * D;
              float score = 0;
              for (std::uint32_t d = 0; d < D; ++d) score += qh[d] * float(key[d]);
              scores[t - first] = score / std::sqrt(float(D));
              max = std::max(max, scores[t - first]);
            }
            float den = 0;
            for (auto& score : scores) {
              score = std::exp(score - max);
              den += score;
            }
            float* a = &att[std::size_t(r) * H + h * D];
            std::fill(a, a + D, 0.0f);
            for (std::uint32_t t = first; t <= pos; ++t) {
              const _Float16* value = s.v.data() + std::size_t(t % s.capacity) * KV + kh * D;
              const float pr = scores[t - first] / den;
              for (std::uint32_t d = 0; d < D; ++d) a[d] += pr * float(value[d]);
            }
            const float g = 1 / (1 + std::exp(-gate[h]));
            for (std::uint32_t d = 0; d < D; ++d) a[d] *= g;
          };
          if (pos - first < 512) {
            std::vector<float> scores;
            for (std::uint32_t h = 0; h < c.heads; ++h) head(h, scores);
          } else {
            ::lfm3::npu::Parallel(c.heads, [&](std::size_t h0, std::size_t h1) {
              std::vector<float> scores;
              for (std::size_t h = h0; h < h1; ++h) head(std::uint32_t(h), scores);
            });
          }
        }
        if (bidirectional) {
          // F16 keys and values as stored, gate per head, group masks.
          std::vector<float> kf(std::size_t(n) * KV), vf(kf.size());
          for (std::uint32_t r = 0; r < n; ++r) {
            const float* k = &proj[std::size_t(r) * qkvg] + H;
            for (std::uint32_t i = 0; i < KV; ++i) {
              kf[std::size_t(r) * KV + i] = float(_Float16(k[i]));
              vf[std::size_t(r) * KV + i] = float(_Float16(k[KV + i]));
            }
          }
          const auto* g = groups.empty() ? nullptr : groups.data() + base;
          Attention(proj.data(), kf.data(), vf.data(), att.data(), n, c.heads, kv, D,
                    [&](std::size_t i, std::size_t t) {
                      return !g || g[i] < 0 || g[t] < 0 || g[i] == g[t];
                    },
                    proj.data() + H + 2 * KV, qkvg, qkvg, KV);
        }
        m.Run(p.out, att, n, out);
      }
      for (std::uint32_t r = 0; r < n; ++r) {
        if (l.post_norm.data) Norm(&out[r * H], H, Floats(l.post_norm), c.eps);
        for (std::uint32_t i = 0; i < H; ++i) x[r * H + i] += out[r * H + i];
      }
      z = x;
      for (std::uint32_t r = 0; r < n; ++r) Norm(&z[r * H], H, Floats(l.ffn_norm), c.eps);
      const std::uint32_t I = c.intermediate;
      std::vector<float> gu(std::size_t(n) * 2 * I), act(std::size_t(n) * I);
      m.Run(p.gateup, z, n, gu);
      for (std::uint32_t r = 0; r < n; ++r)
        for (std::uint32_t i = 0; i < I; ++i) {
          const float g = gu[std::size_t(r) * 2 * I + i], u = gu[std::size_t(r) * 2 * I + I + i];
          act[std::size_t(r) * I + i] = (g / (1 + std::exp(-g))) * u;
        }
      m.Run(p.down, act, n, out);
      for (std::uint32_t r = 0; r < n; ++r) {
        if (l.post_ffn_norm.data) Norm(&out[r * H], H, Floats(l.post_ffn_norm), c.eps);
        for (std::uint32_t i = 0; i < H; ++i) x[r * H + i] += out[r * H + i];
      }
    }
    for (std::uint32_t r = 0; r < n; ++r) {
      if (!dbg) Norm(&x[r * H], H, Floats(w.output_norm), c.eps);
      if (!hidden.empty())
        std::copy(&x[r * H], &x[r * H] + H, hidden.begin() + (base + r) * H);
      chunk[r].state->position = chunk[r].position + 1;
    }
    std::vector<float> wanted;
    std::uint32_t count = 0;
    for (std::uint32_t r = 0; r < n; ++r)
      if (chunk[r].logits) {
        wanted.insert(wanted.end(), &x[r * H], &x[r * H] + H);
        ++count;
      }
    if (count && !logits.empty()) {
      std::vector<float> y(std::size_t(count) * c.vocab);
      m.Run(m.head, wanted, count, y);
      std::copy(y.begin(), y.end(), logits.begin() + logit_row * c.vocab);
    }
    logit_row += count;
  }
}


void Backbone::Impl::ForwardDevice(std::span<const float> inputs, std::span<const Row> rows,
                                   std::span<float> hidden, std::span<float> logits, bool decision,
                                   std::span<const std::int32_t> groups) {
  auto& d = *device;
  std::lock_guard lock(d.mutex);
  auto& npu = *d.npu;
  const auto& c = w->config;
  const std::uint32_t H = c.hidden, I = c.intermediate, D = c.head_dim, Lw = c.conv_width;
  const std::size_t pass = decision ? rows.size() : 64;
  if (pass > Device::kMaxRows) throw std::invalid_argument("NPU decision prompt longer than 512 rows");
  std::size_t logit_row = 0;
  for (std::size_t base = 0; base < rows.size(); base += pass) {
    const std::uint32_t R = std::uint32_t(std::min(pass, rows.size() - base));
    const std::uint32_t blocks = (R + 7) / 8;
    const auto chunk = rows.subspan(base, R);
    // Row metadata: slot, position, first and last visible key, group.
    // Rows past R (kernels run at least eight) write into the scratch slot.
    const std::uint32_t RP = std::max<std::uint32_t>(R, 8);
    std::vector<std::int32_t> mg(std::size_t(RP) * 16, 0), ms(mg.size(), 0), kg(RP + 32, -1);
    for (std::uint32_t r = R; r < RP; ++r) mg[r * 16] = ms[r * 16] = int(Device::kSlots) - 1;
    for (std::uint32_t r = 0; r < R; ++r) {
      const auto& row = chunk[r];
      if (!row.state || row.state->slot < 0) throw std::logic_error("NPU row without a device slot");
      const int pos = int(row.position);
      const int g = groups.empty() ? -1 : groups[base + r];
      mg[r * 16 + 0] = ms[r * 16 + 0] = row.state->slot;
      mg[r * 16 + 1] = ms[r * 16 + 1] = pos;
      mg[r * 16 + 2] = 0;
      mg[r * 16 + 3] = decision ? int(R) - 1 : pos;
      ms[r * 16 + 2] = pos + 1 > int(c.window) ? pos + 1 - int(c.window) : 0;
      ms[r * 16 + 3] = pos;
      mg[r * 16 + 4] = ms[r * 16 + 4] = g;
      kg[r] = g;
    }
    npu.Write(d.meta_g, 0, mg.data(), mg.size() * 4);
    npu.Write(d.meta_s, 0, ms.data(), ms.size() * 4);
    npu.Write(d.groups, 0, kg.data(), kg.size() * 4);
    npu.Write(d.x, 0, inputs.data() + base * H, std::size_t(R) * H * 4);
    // Kernels that spread rows over the eight columns run at least eight
    // rows (a column without rows would leave the array context).
    const std::uint32_t R8 = std::max<std::uint32_t>(R, 8);
    const auto key = [&](const std::string& n) { return n + "_r" + std::to_string(R); };
    auto row_kernel = [&](RowKernel k) { return d.Row(k); };
    const int first = row_kernel(NormPackKernel(H, R8, Residual::kNone, false, false, c.eps));
    {
      const int b[] = {d.x, d.w_first, d.xb2048};
      npu.Enqueue(first, b);
    }
    int x = d.x, x2 = d.x2;
    const char* dbg = std::getenv("GUFO_LFM3_NPU_DEBUG_LAYERS");
    const std::size_t nl = dbg ? std::size_t(std::atoi(dbg)) : c.layers.size();
    for (std::size_t il = 0; il < nl; ++il) {
      const auto& p = layers[il];
      if (c.layers[il] == LayerType::kConvolution) {
        dense->Enqueue(p.in, d.xb2048, d.y_proj, blocks);
        const int conv = d.Kernel(key("conv"), [&] { return ConvProgram(H, Lw, R, Device::kSlots, key("conv")); },
                                  "@" + key("conv"));
        const int b[] = {d.y_proj, d.hist[il], d.conv_w[il], d.meta_g, d.xb2048};
        npu.Enqueue(conv, b);
      } else {
        const bool swa = c.layers[il] == LayerType::kSlidingAttention;
        const std::uint32_t cap = swa ? d.cap_sliding : d.cap_global;
        dense->Enqueue(p.qkvg, d.xb2048, d.y_proj, blocks);
        AttentionShape s{c.heads, c.kv_heads[il], D, p.qkvg.shape.m / 8, true, cap, Device::kSlots, d.table_rows, R,
                         decision && !swa};
        const std::string tag = std::string(swa ? "s" : "g") + std::to_string(c.kv_heads[il]) + (s.groups ? "x" : "");
        AttentionShape s8 = s;
        s8.rows = R8;
        const std::string pk = "prep" + tag + "_r" + std::to_string(R8);
        const int prep = d.Kernel(pk, [&] { return QkPrepProgram(s8, pk); }, "@" + pk);
        const int attend = d.Kernel(key("attend" + tag), [&] { return AttendProgram(s, key("attend" + tag)); },
                                    "@" + key("attend" + tag));
        const int meta = swa ? d.meta_s : d.meta_g;
        const int b1[] = {d.y_proj, d.qn[il], d.kn[il], swa ? d.cos_s : d.cos_g, swa ? d.sin_s : d.sin_g, meta, d.q, d.gates, d.kpool[il], d.vpool[il]};
        npu.Enqueue(prep, b1);
        const int b2[] = {d.q, d.gates, d.kpool[il], d.vpool[il], meta, d.groups, d.xb2048};
        npu.Enqueue(attend, b2);
      }
      dense->Enqueue(p.out, d.xb2048, d.y_out, blocks);
      {
        const int k = row_kernel(NormPackKernel(H, R8, Residual::kBlocked, true, false, c.eps));
        const int b[] = {x, d.y_out, d.w_op[il], x2, d.xb2048};
        npu.Enqueue(k, b);
        std::swap(x, x2);
      }
      dense->Enqueue(p.gateup, d.xb2048, d.y_proj, blocks);
      {
        const int k = row_kernel(SwiGluPackKernel(I, R8));
        const int b[] = {d.y_proj, d.xb_ffn};
        npu.Enqueue(k, b);
      }
      dense->Enqueue(p.down, d.xb_ffn, d.y_down, blocks);
      {
        const int k = row_kernel(ReducePartialKernel(H, R8));
        const int b[] = {d.y_down, d.o};
        npu.Enqueue(k, b);
      }
      {
        const bool last = il + 1 == c.layers.size();
        const int k = row_kernel(NormPackKernel(H, R8, Residual::kRows, true, last, c.eps));
        if (last) {
          const int b[] = {x, d.o, d.w_ffn[il], x2, d.xb2048, d.z};
          npu.Enqueue(k, b);
        } else {
          const int b[] = {x, d.o, d.w_ffn[il], x2, d.xb2048};
          npu.Enqueue(k, b);
        }
        std::swap(x, x2);
      }
    }
    std::uint32_t wanted = 0;
    for (const auto& row : chunk) wanted += row.logits;
    if (wanted && !logits.empty()) dense->Enqueue(head, d.xb2048, d.y_head, blocks);
    npu.Finish();
    std::vector<float> z(std::size_t(R) * H);
    npu.Read(dbg ? x : d.z, 0, z.data(), z.size() * 4);
    if (!hidden.empty()) std::copy(z.begin(), z.end(), hidden.begin() + base * H);
    for (std::uint32_t r = 0; r < R; ++r) {
      chunk[r].state->position = chunk[r].position + 1;
      chunk[r].state->host_stale = true;
    }
    if (wanted && !logits.empty()) {
      const std::uint32_t mp = head.shape.m, V = c.vocab;
      std::vector<float> block(std::size_t(mp) * 64);
      int loaded = -1;
      for (std::uint32_t r = 0; r < R; ++r) {
        if (!chunk[r].logits) continue;
        if (int(r / 8) != loaded) {
          npu.Read(d.y_head, std::uint64_t(r / 8) * block.size() * 4, block.data(), block.size() * 4);
          loaded = int(r / 8);
        }
        float* out = logits.data() + logit_row * V;
        for (std::uint32_t v = 0; v < V; ++v) out[v] = block[(std::size_t(v / 8) * 8 + r % 8) * 8 + v % 8];
        ++logit_row;
      }
    }
  }
}

}  // namespace gufo::models::lfm3::npu
