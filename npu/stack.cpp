#include "stack.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "host_ops.hpp"

namespace gufo::models::lfm3::npu {
namespace {
const float* F(const Tensor& t) { return static_cast<const float*>(t.data); }
}  // namespace

void Stack::State::Clear() {
  for (auto& c : conv) std::fill(c.begin(), c.end(), 0.0f);
  position = 0;
}

Stack::Stack(const lfm3_audio::Stack& s) : w_(&s) {
  auto& dense = Dense::Shared();
  for (const auto& b : s.blocks) {
    Block nb;
    if (b.attention()) {
      std::vector<const Tensor*> qkvg = {&b.q, &b.k, &b.v};
      if (b.attn_gate.data) qkvg.push_back(&b.attn_gate);
      nb.qkvg = dense.Upload(qkvg);
      const Tensor* out[] = {&b.attn_out};
      nb.out = dense.Upload(out);
    } else {
      const Tensor* in[] = {&b.in_proj};
      const Tensor* out[] = {&b.out_proj};
      nb.in = dense.Upload(in);
      nb.out = dense.Upload(out);
    }
    const Tensor* gateup[] = {&b.gate, &b.up};
    const Tensor* down[] = {&b.down};
    nb.gateup = dense.Upload(gateup);
    nb.down = dense.Upload(down);
    blocks_.push_back(nb);
  }
  frequencies_.resize(s.head_dim / 2);
  for (std::uint32_t i = 0; i < s.head_dim / 2; ++i)
    frequencies_[i] = float(std::pow(double(s.theta), double(-2.0f * i / s.head_dim)));
}

Stack::State Stack::NewState(std::uint32_t window) const {
  State st;
  st.capacity = window;
  const auto& s = *w_;
  for (const auto& b : s.blocks) {
    if (b.attention()) {
      st.conv.emplace_back();
      st.k.emplace_back(std::size_t(window) * s.kv_heads * s.head_dim, _Float16(0));
      st.v.emplace_back(std::size_t(window) * s.kv_heads * s.head_dim, _Float16(0));
    } else {
      st.conv.emplace_back(std::size_t(s.hidden) * (s.conv_width - 1), 0.0f);
      st.k.emplace_back();
      st.v.emplace_back();
    }
  }
  return st;
}

void Stack::Run(std::span<const float> input, int n, State* state, int window, std::span<float> z_out,
                int segment) const {
  const auto& s = *w_;
  const std::size_t H = s.hidden, D = s.head_dim, Q = std::size_t(s.heads) * D,
                    KV = std::size_t(s.kv_heads) * D;
  const int position = state ? state->position : 0;
  auto& dense = Dense::Shared();
  std::vector<float> x(input.begin(), input.begin() + std::size_t(n) * H), z(x.size()), o(x.size());
  for (std::size_t i = 0; i < s.blocks.size(); ++i) {
    const auto& b = s.blocks[i];
    const auto& nb = blocks_[i];
    RmsNormRows(x.data(), n, H, F(b.norm), s.eps, z.data());
    if (b.attention()) {
      const std::size_t width = Q + 2 * KV + (b.attn_gate.data ? s.heads : 0);
      std::vector<float> p(std::size_t(n) * width), a(std::size_t(n) * Q);
      dense.Run(nb.qkvg, z, std::uint32_t(n), p);
      // Per-head norms, rotary per segment, F16 K/V as the HIP path packs them.
      const int rows = segment > 0 && !state ? segment : n;
      std::vector<float> kf(std::size_t(n) * KV), vf(kf.size());
      for (int r = 0; r < n; ++r) {
        float* q = &p[std::size_t(r) * width];
        float* k = q + Q;
        const float* v = k + KV;
        for (std::uint32_t h = 0; h < s.heads; ++h) RmsNorm(q + h * D, D, F(b.q_norm), s.eps);
        for (std::uint32_t h = 0; h < s.kv_heads; ++h) RmsNorm(k + h * D, D, F(b.k_norm), s.eps);
        const int token = position + r % rows;
        for (float* t : {q, k})
          for (std::uint32_t h = 0; h < (t == q ? s.heads : s.kv_heads); ++h)
            for (std::size_t pr = 0; pr < D / 2; ++pr) {
              const float angle = float(token) * frequencies_[pr];
              const float sn = std::sin(angle), cs = std::cos(angle);
              float* at = t + h * D + pr;
              const float c = at[0], d = at[D / 2];
              at[0] = c * cs - d * sn;
              at[D / 2] = d * cs + c * sn;
            }
        for (std::size_t j = 0; j < KV; ++j) {
          kf[std::size_t(r) * KV + j] = float(_Float16(k[j]));
          vf[std::size_t(r) * KV + j] = float(_Float16(v[j]));
        }
      }
      const float* gate = b.attn_gate.data ? p.data() + Q + 2 * KV : nullptr;
      if (!state) {
        for (int r0 = 0; r0 < n; r0 += rows) {
          Attention(p.data() + std::size_t(r0) * width, kf.data() + std::size_t(r0) * KV,
                    vf.data() + std::size_t(r0) * KV, a.data() + std::size_t(r0) * Q, std::size_t(rows),
                    s.heads, s.kv_heads, std::uint32_t(D), [](std::size_t, std::size_t) { return true; },
                    gate ? gate + std::size_t(r0) * width : nullptr, width, width, KV);
        }
      } else {
        // Causal over the ring of earlier rows and this chunk, as the HIP
        // kernel reads them (window > 0 bounds the keys).
        const auto& ck = state->k[i];
        const auto& cv = state->v[i];
        const std::uint32_t group = s.heads / s.kv_heads;
        const float scale = 1 / std::sqrt(float(D));
        ::lfm3::npu::Parallel(std::size_t(n) * s.heads, [&](std::size_t j0, std::size_t j1) {
          for (std::size_t job = j0; job < j1; ++job) {
            const int row = int(job / s.heads);
            const std::uint32_t h = std::uint32_t(job % s.heads), kh = h / group;
            const int query = position + row;
            const int first = window > 0 ? std::max(0, query + 1 - window) : 0;
            const float* q = &p[std::size_t(row) * width] + h * D;
            float maximum = -std::numeric_limits<float>::infinity(), den = 0;
            std::vector<float> acc(D, 0.0f);
            for (int t = first; t <= query; ++t) {
              const float* key = nullptr;
              const float* value = nullptr;
              std::vector<float> kk, vv;
              if (t >= position) {
                key = &kf[std::size_t(t - position) * KV + kh * D];
                value = &vf[std::size_t(t - position) * KV + kh * D];
              } else {
                kk.resize(D);
                vv.resize(D);
                const std::size_t slot = std::size_t(t % state->capacity) * KV + kh * D;
                for (std::size_t d = 0; d < D; ++d) {
                  kk[d] = float(ck[slot + d]);
                  vv[d] = float(cv[slot + d]);
                }
                key = kk.data();
                value = vv.data();
              }
              float dot = 0;
              for (std::size_t d = 0; d < D; ++d) dot = std::fma(q[d], key[d], dot);
              const float score = dot * scale;
              const float next = std::max(maximum, score);
              const float alpha = std::exp(maximum - next), beta = std::exp(score - next);
              den = den * alpha + beta;
              maximum = next;
              for (std::size_t d = 0; d < D; ++d) acc[d] = acc[d] * alpha + beta * value[d];
            }
            const float g = gate ? Sigmoid(gate[std::size_t(row) * width + h]) : 1.0f;
            float* out = &a[std::size_t(row) * Q + h * D];
            for (std::size_t d = 0; d < D; ++d) out[d] = acc[d] / den * g;
          }
        });
        for (int r = 0; r < n; ++r) {
          const std::size_t slot = std::size_t((position + r) % state->capacity) * KV;
          for (std::size_t j = 0; j < KV; ++j) {
            state->k[i][slot + j] = _Float16(kf[std::size_t(r) * KV + j]);
            state->v[i][slot + j] = _Float16(vf[std::size_t(r) * KV + j]);
          }
        }
      }
      dense.Run(nb.out, a, std::uint32_t(n), o);
    } else {
      if (!state) throw std::logic_error("NPU stack convolution requires state");
      std::vector<float> p(std::size_t(n) * 3 * H), a(std::size_t(n) * H);
      dense.Run(nb.in, z, std::uint32_t(n), p);
      const float* w = F(b.conv);
      const std::uint32_t L = s.conv_width;
      auto& hist = state->conv[i];
      for (int r = 0; r < n; ++r) {
        const float* pr = &p[std::size_t(r) * 3 * H];
        for (std::size_t ch = 0; ch < H; ++ch) {
          float* own = hist.data() + ch * (L - 1);
          const float current = pr[ch] * pr[2 * H + ch];
          float sum = 0;
          for (std::uint32_t t = 0; t < L; ++t) sum = std::fma(w[ch * L + t], t == L - 1 ? current : own[t], sum);
          a[std::size_t(r) * H + ch] = sum * pr[H + ch];
          for (std::uint32_t j = 0; j + 2 < L; ++j) own[j] = own[j + 1];
          if (L > 1) own[L - 2] = current;
        }
      }
      dense.Run(nb.out, a, std::uint32_t(n), o);
    }
    if (b.post_norm.data) RmsNormRows(o.data(), n, H, F(b.post_norm), s.eps, o.data());
    for (std::size_t j = 0; j < x.size(); ++j) x[j] += o[j];
    RmsNormRows(x.data(), n, H, F(b.ffn_norm), s.eps, z.data());
    const std::size_t I = s.intermediate;
    std::vector<float> gu(std::size_t(n) * 2 * I), act(std::size_t(n) * I);
    dense.Run(nb.gateup, z, std::uint32_t(n), gu);
    for (int r = 0; r < n; ++r)
      for (std::size_t j = 0; j < I; ++j)
        act[r * I + j] = Silu(gu[r * 2 * I + j]) * gu[r * 2 * I + I + j];
    dense.Run(nb.down, act, std::uint32_t(n), o);
    if (b.post_ffn_norm.data) RmsNormRows(o.data(), n, H, F(b.post_ffn_norm), s.eps, o.data());
    for (std::size_t j = 0; j < x.size(); ++j) x[j] += o[j];
  }
  RmsNormRows(x.data(), n, H, F(s.output_norm), s.eps, z_out.data());
  if (state) state->position += n;
}

}  // namespace gufo::models::lfm3::npu
