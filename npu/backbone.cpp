#include "backbone.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "host/bfp.hpp"
#include "host/npu_runtime.hpp"

namespace gufo::models::lfm3::npu {
namespace {

constexpr std::string_view kProgram =
#include "lfm3_proj.inc"
    ;

constexpr std::uint32_t kQkvgRows = 3136;  // q 2048 + k 512 + v 512 + gate 16, padded
using ::lfm3::npu::Split;

// A projection: encoded weights on the NPU and the kernel that reads them.
struct Projection {
  int kernel{-1}, weights{-1};
  std::uint32_t k{}, m{}, padded{};
  Split split{Split::kRows};
};

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
    std::fill(l.k.begin(), l.k.end(), 0.0f);
    std::fill(l.v.begin(), l.v.end(), 0.0f);
  }
}

struct Backbone::Impl {
  const Weights* w{};
  std::unique_ptr<::lfm3::npu::Runtime> npu;
  struct Layer {
    Projection in, out, qkvg, gateup, down;
  };
  std::vector<Layer> layers;
  Projection head;
  // Activation and output buffers per K and per output shape.
  int x2048{-1}, x7680{-1};
  std::vector<std::pair<std::uint32_t, int>> outputs;  // padded rows x parts -> buffer
  std::vector<std::uint16_t> xb;
  std::vector<float> yb;

  int Kernel(std::uint32_t k, std::uint32_t m, Split split) {
    const std::string root = split == Split::kRows
                                 ? "@proj_m_k" + std::to_string(k) + "_m" + std::to_string(m)
                                 : "@proj_k_k" + std::to_string(k) + "_m" + std::to_string(m);
    for (const auto& l : layers)
      for (const auto* p : {&l.in, &l.out, &l.qkvg, &l.gateup, &l.down})
        if (p->kernel >= 0 && p->k == k && p->padded == m && p->split == split) return p->kernel;
    if (head.kernel >= 0 && head.k == k && head.padded == m) return head.kernel;
    return npu->Compile(kProgram, "lfm3_proj.loom", root);
  }

  int Output(std::uint32_t padded, std::uint32_t parts) {
    const std::uint32_t key = padded * 16 + parts;
    for (auto& [k, id] : outputs)
      if (k == key) return id;
    const int id = npu->Alloc(std::uint64_t(parts) * padded * 8 * 4);
    outputs.emplace_back(key, id);
    return id;
  }

  // rows: [r][tensor rows] from consecutive tensors.
  Projection Make(std::span<const Tensor* const> tensors, std::uint32_t padded, Split split) {
    Projection p;
    p.k = tensors[0]->cols;
    p.split = split;
    for (const auto* t : tensors) {
      if (t->cols != p.k) throw std::runtime_error("NPU projection with mixed K");
      p.m += t->rows;
    }
    p.padded = padded ? padded : p.m;
    p.kernel = Kernel(p.k, p.padded, split);
    auto bytes = ::lfm3::npu::PackWeights(p.m, p.k, p.padded, split, [&](std::uint32_t r, std::span<float> out) {
      for (const auto* t : tensors) {
        if (r < t->rows) return DecodeRow(*t, r, out);
        r -= t->rows;
      }
    });
    p.weights = npu->Alloc(bytes.size());
    constexpr std::size_t kChunk = 64u << 20;
    for (std::size_t off = 0; off < bytes.size(); off += kChunk)
      npu->Write(p.weights, off, bytes.data() + off, std::min(kChunk, bytes.size() - off));
    Output(p.padded, split == Split::kColumns ? 8 : 1);
    return p;
  }

  // y[r][m] = W x[r] for rows r < n (n <= 8).
  void Run(const Projection& p, std::span<const float> x, std::uint32_t n, std::span<float> y) {
    const int xbuf = p.k == 2048 ? x2048 : x7680;
    xb.resize(std::size_t(p.k) * 8);
    ::lfm3::npu::PackActivations(x, n, p.k, xb);
    npu->Write(xbuf, 0, xb.data(), xb.size() * 2);
    const std::uint32_t parts = p.split == Split::kColumns ? 8 : 1;
    const int ybuf = Output(p.padded, parts);
    const int bind[] = {xbuf, p.weights, ybuf};
    npu->Dispatch(p.kernel, bind);
    yb.resize(std::size_t(parts) * p.padded * 8);
    npu->Read(ybuf, 0, yb.data(), yb.size() * 4);
    ::lfm3::npu::UnpackOutputs(yb, n, p.m, y, parts, p.padded);
  }
};

Backbone::Backbone() : impl_(std::make_unique<Impl>()) {}
Backbone::~Backbone() = default;
const Config& Backbone::config() const { return impl_->w->config; }

std::unique_ptr<Backbone> Backbone::Load(const Weights& w, std::string* error) {
  try {
    const auto& c = w.config;
    if (c.hidden != 2048 || c.intermediate != 7680 || c.heads * c.head_dim != 2048)
      throw std::runtime_error("NPU backbone supports the LFM3 1.7B/2B shape only");
    auto b = std::unique_ptr<Backbone>(new Backbone);
    auto& m = *b->impl_;
    m.w = &w;
    m.npu = std::make_unique<::lfm3::npu::Runtime>();
    m.x2048 = m.npu->Alloc(2048 * 8 * 2);
    m.x7680 = m.npu->Alloc(7680 * 8 * 2);
    m.layers.resize(c.layers.size());
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
      const auto& l = w.layers[i];
      auto& p = m.layers[i];
      if (c.layers[i] == LayerType::kConvolution) {
        const Tensor* in[] = {&l.in_proj};
        const Tensor* out[] = {&l.out_proj};
        p.in = m.Make(in, 0, Split::kRows);
        p.out = m.Make(out, 0, Split::kRows);
      } else {
        const Tensor* qkvg[] = {&l.q, &l.k, &l.v, &l.attn_gate};
        const Tensor* out[] = {&l.attn_out};
        p.qkvg = m.Make(qkvg, kQkvgRows, Split::kRows);
        p.out = m.Make(out, 0, Split::kRows);
      }
      const Tensor* gateup[] = {&l.gate, &l.up};
      const Tensor* down[] = {&l.down};
      p.gateup = m.Make(gateup, 0, Split::kRows);
      p.down = m.Make(down, 0, Split::kColumns);
    }
    const Tensor* head[] = {&w.output};
    m.head = m.Make(head, 0, Split::kRows);
    return b;
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return nullptr;
  }
}

State Backbone::NewState(std::uint32_t context) const {
  const auto& c = impl_->w->config;
  State s;
  s.layers.resize(c.layers.size());
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    auto& l = s.layers[i];
    if (c.layers[i] == LayerType::kConvolution) {
      l.conv.assign(std::size_t(c.hidden) * (c.conv_width - 1), 0.0f);
    } else {
      // Sliding layers keep kRows extra slots so rows of one pass never
      // overwrite keys an earlier row of the pass still attends to.
      l.capacity = c.layers[i] == LayerType::kSlidingAttention
                       ? std::min(context, c.window + kRows)
                       : context;
      l.k.assign(std::size_t(l.capacity) * c.kv_heads[i] * c.head_dim, 0.0f);
      l.v.assign(l.k.size(), 0.0f);
    }
  }
  return s;
}

void Backbone::Embed(std::uint32_t token, std::span<float> row) const {
  DecodeRow(impl_->w->embedding, token, row);
}

void Backbone::Forward(std::span<const float> inputs, std::span<const Row> rows,
                       std::span<float> hidden, std::span<float> logits) {
  auto& m = *impl_;
  const auto& w = *m.w;
  const auto& c = w.config;
  const std::uint32_t H = c.hidden, D = c.head_dim;
  std::size_t logit_row = 0;
  for (std::size_t base = 0; base < rows.size(); base += kRows) {
    const std::uint32_t n = std::uint32_t(std::min<std::size_t>(kRows, rows.size() - base));
    const auto chunk = rows.subspan(base, n);
    std::vector<float> x(inputs.begin() + base * H, inputs.begin() + (base + n) * H);
    std::vector<float> z(std::size_t(n) * H), out(std::size_t(n) * H);
    for (std::size_t il = 0; il < c.layers.size(); ++il) {
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
        // Normalize, rotate and store every row's K/V first; then attend.
        for (std::uint32_t r = 0; r < n; ++r) {
          float* q = &proj[std::size_t(r) * qkvg];
          float* k = q + H;
          float* v = k + KV;
          for (std::uint32_t h = 0; h < c.heads; ++h) Norm(q + h * D, D, Floats(l.q_norm), c.eps);
          for (std::uint32_t h = 0; h < kv; ++h) Norm(k + h * D, D, Floats(l.k_norm), c.eps);
          rope(q, c.heads, chunk[r].position);
          rope(k, kv, chunk[r].position);
          auto& s = chunk[r].state->layers[il];
          const std::size_t slot = chunk[r].position % s.capacity;
          std::copy(k, k + KV, s.k.begin() + slot * KV);
          std::copy(v, v + KV, s.v.begin() + slot * KV);
        }
        ::lfm3::npu::Parallel(std::size_t(n) * c.heads, [&](std::size_t b0, std::size_t b1) {
          std::vector<float> scores;
          for (std::size_t job = b0; job < b1; ++job) {
            const std::uint32_t r = std::uint32_t(job / c.heads), h = std::uint32_t(job % c.heads);
            const float* q = &proj[std::size_t(r) * qkvg] + h * D;
            const float* gate = &proj[std::size_t(r) * qkvg] + H + 2 * KV;
            const auto& s = chunk[r].state->layers[il];
            const std::uint32_t pos = chunk[r].position;
            const std::uint32_t first = swa && pos + 1 > c.window ? pos + 1 - c.window : 0;
            const std::uint32_t kh = h / (c.heads / kv);
            scores.resize(pos + 1 - first);
            float max = -std::numeric_limits<float>::infinity();
            for (std::uint32_t t = first; t <= pos; ++t) {
              const float* key = s.k.data() + std::size_t(t % s.capacity) * KV + kh * D;
              float score = 0;
              for (std::uint32_t d = 0; d < D; ++d) score += q[d] * key[d];
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
              const float* value = s.v.data() + std::size_t(t % s.capacity) * KV + kh * D;
              const float pr = scores[t - first] / den;
              for (std::uint32_t d = 0; d < D; ++d) a[d] += pr * value[d];
            }
            const float g = 1 / (1 + std::exp(-gate[h]));
            for (std::uint32_t d = 0; d < D; ++d) a[d] *= g;
          }
        });
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
      Norm(&x[r * H], H, Floats(w.output_norm), c.eps);
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

}  // namespace gufo::models::lfm3::npu
