#include "backbone.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "dense.hpp"
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

struct Backbone::Impl {
  const Weights* w{};
  Dense* dense{};
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
      l.capacity = c.layers[i] == LayerType::kSlidingAttention
                       ? std::min(context, c.window)
                       : context;
      l.k.assign(std::size_t(l.capacity) * c.kv_heads[i] * c.head_dim, _Float16(0));
      l.v.assign(l.k.size(), _Float16(0));
    }
  }
  return s;
}

void Backbone::Head(std::span<const float> hidden, std::uint32_t n, std::span<float> logits) {
  impl_->Run(impl_->head, hidden, n, logits);
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
        // Rows advance in order: each stores its K/V, then attends. A ring
        // of the window's size then never loses a key a later row needs.
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
