#include "npu/backbone.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <gufo/ext/experimental/npu.hpp>
#include <mutex>
#include <stdexcept>

#include "npu/dense.hpp"
#include "npu/grouped_attention.hpp"
#include "npu/host_ops.hpp"

namespace gufo::models::lfm3::npu {
namespace host = gufo::npu::host;
namespace {

const float* F(const Tensor& t) {
  return static_cast<const float*>(t.data);
}

// Projections of at most this many rows run on Q8Gemm, longer ones on
// BfpGemm.
constexpr int kQ8Rows = 16;
// Prefill chunks: rows per chunk (BfpGemm runs rows in bands of 512) and
// at most this many chunks (BfpGemm stages four threads at once).
constexpr int kChunkRows = 512, kMaxChunks = 4;

enum Which { kMixIn, kMixOut, kGateUp, kDown, kCount };

bool Timing() {
  static const bool on = std::getenv("GUFO_LFM3_NPU_TIMING") != nullptr;
  return on;
}

}  // namespace

struct Backbone::Layer {
  LayerType type{};
  const LayerWeights* w{};
  std::uint32_t kv_heads{};
  Dense::Matrix q8[kCount];
  int bfp[kCount]{-1, -1, -1, -1};
  std::vector<float> conv;    // [hidden][conv_width - 1], oldest first
  std::vector<float> conv_t;  // [conv_width][hidden], transposed for SIMD
  std::vector<float> k, v;    // [position][kv_heads * head_dim]
};

Backbone::Backbone(const Weights& w, int max_length) : w_(w) {
  const auto& c = w.config;
  max_length_ = std::max(1, std::min<int>(max_length, int(c.context)));
  Dense::Open();
  auto& dense = Dense::Shared();
  auto& bfp = dense.Bfp();
  const auto t0 = std::chrono::steady_clock::now();
  // bfp16 copy of stacked tensors: rows dequantized on the pool's threads.
  std::vector<float> rows;
  auto upload_bfp = [&](std::span<const Tensor* const> ts) {
    std::uint32_t n = 0;
    for (const auto* t : ts)
      n += t->rows;
    const std::uint32_t k = ts[0]->cols;
    rows.resize(std::size_t(n) * k);
    host::ParallelFor(n, 64, [&](std::size_t r0, std::size_t r1) {
      for (std::size_t r = r0; r < r1; ++r) {
        std::uint32_t local = std::uint32_t(r);
        for (const auto* t : ts) {
          if (local < t->rows) {
            DecodeRow(*t, local, std::span<float>(rows.data() + r * k, k));
            break;
          }
          local -= t->rows;
        }
      }
    });
    return bfp.Upload(int(n), int(k), rows);
  };
  layers_.resize(c.layers.size());
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    auto& l = layers_[i];
    const auto& lw = w.layers[i];
    l.type = c.layers[i];
    l.w = &lw;
    l.kv_heads = c.kv_heads[i];
    max_kv_ = std::max(max_kv_, std::size_t(l.kv_heads) * c.head_dim);
    std::vector<const Tensor*> in, out;
    if (l.type == LayerType::kConvolution) {
      in = {&lw.in_proj};
      out = {&lw.out_proj};
      l.conv.assign(std::size_t(c.hidden) * (c.conv_width - 1), 0.0f);
      l.conv_t.resize(std::size_t(c.conv_width) * c.hidden);
      const float* cw = F(lw.conv);
      for (std::size_t ch = 0; ch < c.hidden; ++ch)
        for (std::size_t j = 0; j < c.conv_width; ++j)
          l.conv_t[j * c.hidden + ch] = cw[ch * c.conv_width + j];
    } else {
      in = {&lw.q, &lw.k, &lw.v, &lw.attn_gate};
      out = {&lw.attn_out};
    }
    const Tensor* gateup[] = {&lw.gate, &lw.up};
    const Tensor* down[] = {&lw.down};
    const std::span<const Tensor* const> mats[kCount] = {in, out, gateup, down};
    for (int m = 0; m < kCount; ++m) {
      l.q8[m] = dense.Upload(mats[m]);
      l.bfp[m] = upload_bfp(mats[m]);
    }
  }
  rows = {};
  const Tensor* head[] = {&w.output};
  head_ = dense.Upload(head);
  const std::uint32_t D = c.head_dim;
  inv_full_.resize(D / 2);
  inv_swa_.resize(D / 2);
  for (std::uint32_t i = 0; i < D / 2; ++i) {
    inv_full_[i] = std::pow(c.theta, -2.0F * i / D);
    if (w.rope_freqs.data)
      inv_full_[i] /= F(w.rope_freqs)[i];
    inv_swa_[i] = std::pow(c.theta_swa, -2.0F * i / D);
  }
  if (Timing())
    std::fprintf(stderr, "lfm3 npu: weights uploaded in %.0f ms\n",
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count());
}

Backbone::~Backbone() = default;

void Backbone::Clear() {
  length_ = 0;
  for (auto& l : layers_) {
    std::fill(l.conv.begin(), l.conv.end(), 0.0f);
    l.k.clear();
    l.v.clear();
  }
}

void Backbone::Embed(std::int32_t id, float* out) const {
  const auto& c = w_.config;
  if (id < 0 || std::uint32_t(id) >= c.vocab)
    throw std::out_of_range("LFM3 token id");
  DecodeRow(w_.embedding, std::uint32_t(id), std::span<float>(out, c.hidden));
}

void Backbone::Project(const Layer& l, int which, const float* x, int n,
                       float* y) {
  auto& dense = Dense::Shared();
  const auto& m = l.q8[which];
  if (n <= kQ8Rows) {
    dense.Run(m, std::span<const float>(x, std::size_t(n) * m.k),
              std::uint32_t(n), std::span<float>(y, std::size_t(n) * m.m));
    return;
  }
  dense.Bfp().Run(gufo::npu::BfpGemm::Call{l.bfp[which], x, n, y});
}

// Chunks of one prompt run on their own threads, chunk j a step behind chunk
// j - 1: its layer's mixing (convolution, attention) waits until j - 1 has
// mixed that layer (the convolution history and K/V it reads), so one chunk's
// host work overlaps another's NPU products.
struct Backbone::Sync {
  std::mutex mutex;
  std::condition_variable changed;
  std::vector<int> mixed;  // per layer: chunks that have mixed it
  void Wait(std::size_t layer, int chunk) {
    std::unique_lock lock(mutex);
    changed.wait(lock, [&] { return mixed[layer] >= chunk; });
  }
  void Done(std::size_t layer) {
    {
      std::lock_guard lock(mutex);
      ++mixed[layer];
    }
    changed.notify_all();
  }
};

void Backbone::Forward(float* x, int n, std::span<const std::int32_t> groups) {
  if (n <= 0)
    return;
  const auto& c = w_.config;
  if (length_ + n > max_length_)
    throw std::runtime_error("NPU context is full");
  if (!groups.empty() && (length_ != 0 || groups.size() != std::size_t(n)))
    throw std::invalid_argument("grouped NPU prompt");
  const int start = length_;
  for (auto& l : layers_)
    if (l.type != LayerType::kConvolution) {
      const std::size_t KV = std::size_t(l.kv_heads) * c.head_dim;
      l.k.resize(std::size_t(start + n) * KV);
      l.v.resize(l.k.size());
    }
  const auto t0 = std::chrono::steady_clock::now();
  // Grouped prompts see later rows: one chunk.
  const int chunks =
      groups.empty() ? std::min(kMaxChunks, (n + kChunkRows - 1) / kChunkRows)
                     : 1;
  Times total;
  if (chunks == 1) {
    total = Rows(x, n, start, groups, nullptr, 0);
  } else {
    Sync sync;
    sync.mixed.assign(layers_.size(), 0);
    std::vector<std::future<Times>> runs;
    for (int j = 0; j < chunks; ++j) {
      const int r0 = int(std::int64_t(n) * j / chunks),
                r1 = int(std::int64_t(n) * (j + 1) / chunks);
      runs.push_back(std::async(std::launch::async, [&, j, r0, r1] {
        host::SharePool share{unsigned(j), unsigned(chunks)};
        return Rows(x + std::size_t(r0) * c.hidden, r1 - r0, start + r0, {},
                    &sync, j);
      }));
    }
    std::exception_ptr error;
    for (auto& r : runs) {
      try {
        const Times t = r.get();
        total.products += t.products;
        total.attention += t.attention;
        total.convolutions += t.convolutions;
      } catch (...) {
        error = std::current_exception();
      }
    }
    if (error)
      std::rethrow_exception(error);
  }
  length_ += n;
  if (Timing())
    std::fprintf(stderr,
                 "lfm3 npu: forward pos %d rows %d chunks %d %.2f ms "
                 "(summed over chunks: products %.2f ms, attention %.2f ms, "
                 "convolutions %.2f ms)\n",
                 start, n, chunks,
                 std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count(),
                 total.products, total.attention, total.convolutions);
}

Backbone::Times Backbone::Rows(float* x, int n, int start,
                               std::span<const std::int32_t> groups, Sync* sync,
                               int chunk) {
  const auto& c = w_.config;
  const std::size_t H = c.hidden, D = c.head_dim, I = c.intermediate,
                    Q = std::size_t(c.heads) * D, L = c.conv_width;
  const std::size_t rows = std::size_t(n);
  std::vector<float> z(rows * H), a(rows * H), o(rows * H),
      p(rows * std::max(3 * H, Q + 2 * max_kv_ + c.heads)), gu(rows * 2 * I),
      act(rows * I), bx;
  Times times;
  using Clock = std::chrono::steady_clock;
  auto ms = [](Clock::time_point a) {
    return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
  };
  auto project = [&](const Layer& l, int which, const float* in, float* out) {
    const auto t0 = Clock::now();
    Project(l, which, in, n, out);
    times.products += ms(t0);
  };
  for (std::size_t li = 0; li < layers_.size(); ++li) {
    auto& l = layers_[li];
    const auto& lw = *l.w;
    host::RmsNormRows(x, z.data(), rows, H, F(lw.norm), c.eps);
    if (l.type == LayerType::kConvolution) {
      project(l, kMixIn, z.data(), p.data());
      if (sync)
        sync->Wait(li, chunk);
      const auto tc = Clock::now();
      // p rows [B | C | x]: a = C * conv(B * x), causal over the history.
      // bx [L - 1 + rows][hidden] holds the history, then B * x.
      const float* weight_t = l.conv_t.data();
      bx.resize((L - 1 + rows) * H);
      host::ParallelFor(H, 256, [&](std::size_t c0, std::size_t c1) {
        for (std::size_t j = 0; j + 1 < L; ++j)
          for (std::size_t ch = c0; ch < c1; ++ch)
            bx[j * H + ch] = l.conv[ch * (L - 1) + j];
        for (std::size_t t = 0; t < rows; ++t) {
          const float* pt = p.data() + t * 3 * H;
          float* b = bx.data() + (L - 1 + t) * H;
          for (std::size_t ch = c0; ch < c1; ++ch)
            b[ch] = pt[ch] * pt[2 * H + ch];
        }
        for (std::size_t t = 0; t < rows; ++t) {
          const float* pt = p.data() + t * 3 * H;
          float* out = a.data() + t * H;
          const float* b0 = bx.data() + t * H;
          const float* w0 = weight_t;
          for (std::size_t ch = c0; ch < c1; ++ch)
            out[ch] = w0[ch] * b0[ch];
          for (std::size_t j = 1; j < L; ++j) {
            const float* bj = bx.data() + (t + j) * H;
            const float* wj = weight_t + j * H;
            for (std::size_t ch = c0; ch < c1; ++ch)
              out[ch] += wj[ch] * bj[ch];
          }
          for (std::size_t ch = c0; ch < c1; ++ch)
            out[ch] *= pt[H + ch];
        }
        for (std::size_t j = 0; j + 1 < L; ++j)
          for (std::size_t ch = c0; ch < c1; ++ch)
            l.conv[ch * (L - 1) + j] = bx[(rows + j) * H + ch];
      });
      times.convolutions += ms(tc);
      if (sync)
        sync->Done(li);
      project(l, kMixOut, a.data(), o.data());
    } else {
      const std::size_t KV = std::size_t(l.kv_heads) * D,
                        width = Q + 2 * KV + c.heads;
      project(l, kMixIn, z.data(), p.data());
      const auto ta = Clock::now();
      const bool swa = l.type == LayerType::kSlidingAttention;
      const float* inv = swa ? inv_swa_.data() : inv_full_.data();
      host::ParallelFor(rows, 1, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t t = r0; t < r1; ++t) {
          float* pt = p.data() + t * width;
          host::RmsNormRows(pt, pt, c.heads, D, F(lw.q_norm), c.eps);
          host::RmsNormRows(pt + Q, pt + Q, l.kv_heads, D, F(lw.k_norm), c.eps);
          const float pos = float(start + int(t));
          for (std::size_t i = 0; i < D / 2; ++i) {
            const float angle = pos * inv[i], cs = std::cos(angle),
                        sn = std::sin(angle);
            for (std::size_t h = 0; h < c.heads + l.kv_heads; ++h) {
              float* head = pt + h * D;
              const float u = head[i], w2 = head[i + D / 2];
              head[i] = u * cs - w2 * sn;
              head[i + D / 2] = w2 * cs + u * sn;
            }
          }
          const std::size_t at = std::size_t(start) + t;
          std::copy_n(pt + Q, KV, l.k.data() + at * KV);
          std::copy_n(pt + Q + KV, KV, l.v.data() + at * KV);
        }
      });
      if (sync)
        sync->Wait(li, chunk);  // earlier chunks' K/V
      const float scale = 1.0f / std::sqrt(float(D));
      if (!groups.empty() && !swa) {
        GroupedAttention(p.data(), width, l.k.data(), l.v.data(), KV, a.data(),
                         Q, groups, c.heads, l.kv_heads, D, scale);
      } else {
        const std::size_t window = c.window;
        host::AttentionArgs args{
            p.data(), width, l.k.data(), KV,         l.v.data(), KV, a.data(),
            Q,        rows,  c.heads,    l.kv_heads, D,          D,  scale};
        host::Attention(args,
                        [&](std::size_t i, std::size_t* b, std::size_t* e) {
                          const std::size_t pos = std::size_t(start) + i;
                          *b = swa && pos + 1 > window ? pos + 1 - window : 0;
                          *e = pos + 1;
                        });
      }
      host::ParallelFor(rows, 4, [&](std::size_t r0, std::size_t r1) {
        for (std::size_t t = r0; t < r1; ++t) {
          const float* gate = p.data() + t * width + Q + 2 * KV;
          for (std::size_t h = 0; h < c.heads; ++h) {
            const float g = Sigmoid(gate[h]);
            float* row = a.data() + t * Q + h * D;
            for (std::size_t d = 0; d < D; ++d)
              row[d] *= g;
          }
        }
      });
      times.attention += ms(ta);
      if (sync)
        sync->Done(li);
      project(l, kMixOut, a.data(), o.data());
    }
    if (lw.post_norm.data)
      host::AddNormRows(x, o.data(), rows, H, F(lw.post_norm), c.eps, 1.0f);
    else
      for (std::size_t i = 0; i < rows * H; ++i)
        x[i] += o[i];
    host::RmsNormRows(x, z.data(), rows, H, F(lw.ffn_norm), c.eps);
    project(l, kGateUp, z.data(), gu.data());
    host::GatedRows(gu.data(), 2 * I, gu.data() + I, 2 * I, act.data(), I, rows,
                    I, host::Activation::kSilu);
    project(l, kDown, act.data(), o.data());
    if (lw.post_ffn_norm.data)
      host::AddNormRows(x, o.data(), rows, H, F(lw.post_ffn_norm), c.eps, 1.0f);
    else
      for (std::size_t i = 0; i < rows * H; ++i)
        x[i] += o[i];
  }
  return times;
}

void Backbone::FinalNorm(const float* x, int rows, float* out) const {
  const auto& c = w_.config;
  host::RmsNormRows(x, out, std::size_t(rows), c.hidden, F(w_.output_norm),
                    c.eps);
}

void Backbone::Head(const float* normed, int rows, float* logits) {
  const auto& c = w_.config;
  Dense::Shared().Run(
      head_, std::span<const float>(normed, std::size_t(rows) * c.hidden),
      std::uint32_t(rows),
      std::span<float>(logits, std::size_t(rows) * c.vocab));
}

}  // namespace gufo::models::lfm3::npu
