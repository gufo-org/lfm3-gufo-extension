// FP32 host operators between NPU projections, shared by the backbone and
// the towers. Arithmetic follows the HIP kernels they stand in for.
#ifndef GUFO_MODELS_LFM3_NPU_HOST_OPS_HPP_
#define GUFO_MODELS_LFM3_NPU_HOST_OPS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "host/bfp.hpp"

namespace gufo::models::lfm3::npu {

inline void RmsNorm(float* x, std::size_t n, const float* weight, float eps, float* out = nullptr) {
  if (!out) out = x;
  float sum = 0;
  for (std::size_t i = 0; i < n; ++i) sum += x[i] * x[i];
  const float scale = 1 / std::sqrt(sum / n + eps);
  for (std::size_t i = 0; i < n; ++i) out[i] = x[i] * scale * weight[i];
}
inline void RmsNormRows(const float* x, std::size_t rows, std::size_t dim, const float* weight,
                        float eps, float* out) {
  for (std::size_t r = 0; r < rows; ++r)
    RmsNorm(const_cast<float*>(x + r * dim), dim, weight, eps, out + r * dim);
}
inline void LayerNormRows(const float* x, std::size_t rows, std::size_t dim, const float* weight,
                          const float* bias, float eps, float* out) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* row = x + r * dim;
    float mean = 0;
    for (std::size_t i = 0; i < dim; ++i) mean += row[i];
    mean /= dim;
    float squares = 0;
    for (std::size_t i = 0; i < dim; ++i) squares += (row[i] - mean) * (row[i] - mean);
    const float inv = 1 / std::sqrt(squares / dim + eps);
    for (std::size_t i = 0; i < dim; ++i)
      out[r * dim + i] = (row[i] - mean) * inv * weight[i] + (bias ? bias[i] : 0.0f);
  }
}
inline float Silu(float x) { return x / (1 + std::exp(-x)); }
inline float Gelu(float x) { return 0.5f * x * (1 + std::erf(x * 0.70710678118654752f)); }
inline float Sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

// Attention of n query rows over n key rows of one sequence, rows of q/out
// [row][heads][dim], k/v [row][kv_heads][dim]. visible(i, t) selects keys;
// gate, when set, is [row][heads] pre-sigmoid.
template <class Visible>
void Attention(const float* q, const float* k, const float* v, float* out, std::size_t n,
               std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t dim, Visible visible,
               const float* gate = nullptr, std::size_t gate_stride = 0, std::size_t q_stride = 0,
               std::size_t kv_stride = 0) {
  if (!q_stride) q_stride = std::size_t(heads) * dim;
  if (!kv_stride) kv_stride = std::size_t(kv_heads) * dim;
  if (!gate_stride) gate_stride = heads;
  const float scale = 1 / std::sqrt(float(dim));
  const std::uint32_t group = heads / kv_heads;
  ::lfm3::npu::Parallel(n * heads, [&](std::size_t j0, std::size_t j1) {
    std::vector<float> scores(n);
    for (std::size_t job = j0; job < j1; ++job) {
      const std::size_t i = job / heads;
      const std::uint32_t h = std::uint32_t(job % heads), kh = h / group;
      const float* qi = q + i * q_stride + std::size_t(h) * dim;
      float max = -std::numeric_limits<float>::infinity();
      for (std::size_t t = 0; t < n; ++t) {
        if (!visible(i, t)) {
          scores[t] = -std::numeric_limits<float>::infinity();
          continue;
        }
        const float* kt = k + t * kv_stride + std::size_t(kh) * dim;
        float s = 0;
        for (std::uint32_t d = 0; d < dim; ++d) s += qi[d] * kt[d];
        scores[t] = s * scale;
        max = std::max(max, scores[t]);
      }
      float den = 0;
      for (std::size_t t = 0; t < n; ++t) {
        scores[t] = std::isinf(scores[t]) ? 0.0f : std::exp(scores[t] - max);
        den += scores[t];
      }
      float* o = out + i * std::size_t(heads) * dim + std::size_t(h) * dim;
      std::fill(o, o + dim, 0.0f);
      for (std::size_t t = 0; t < n; ++t) {
        if (scores[t] == 0) continue;
        const float p = scores[t] / den;
        const float* vt = v + t * kv_stride + std::size_t(kh) * dim;
        for (std::uint32_t d = 0; d < dim; ++d) o[d] += p * vt[d];
      }
      if (gate) {
        const float g = Sigmoid(gate[i * gate_stride + h]);
        for (std::uint32_t d = 0; d < dim; ++d) o[d] *= g;
      }
    }
  });
}

// y[r][m] = act(sum_k x[r][k] w[m][k] + bias[m]) in FP32 on the host, for
// small exact matrices (DFT bases, 1x1 convolutions). act: 0 none, 1 SiLU,
// 2 GELU.
inline void HostGemm(const float* x, const float* w, const float* bias, float* y, std::size_t rows,
                     std::size_t m, std::size_t k, int act = 0) {
  ::lfm3::npu::Parallel(rows, [&](std::size_t r0, std::size_t r1) {
    for (std::size_t r = r0; r < r1; ++r)
      for (std::size_t o = 0; o < m; ++o) {
        float sum = 0;
        const float* xr = x + r * k;
        const float* wr = w + o * k;
        for (std::size_t i = 0; i < k; ++i) sum = std::fma(xr[i], wr[i], sum);
        if (bias) sum += bias[o];
        y[r * m + o] = act == 1 ? Silu(sum) : act == 2 ? Gelu(sum) : sum;
      }
  });
}

}  // namespace gufo::models::lfm3::npu

#endif
