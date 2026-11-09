#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ops.hpp"
namespace gufo::models::lfm3_vl::gpu {
namespace {
__device__ int Index(int y, int x, int width) {
  return ((y / 2) * (width / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
}
__global__ void Conv(const float* p, const float* w, float* out, int h,
                     int width, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h * width * hidden)
    return;
  const int c = i % hidden, row = i / hidden;
  const int y = (row / 4) / (width / 2) * 2 + row % 4 / 2;
  const int x = (row / 4) % (width / 2) * 2 + row % 2;
  float sum = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
      if (y + dy >= 0 && y + dy < h && x + dx >= 0 && x + dx < width) {
        const auto at =
            std::size_t(Index(y + dy, x + dx, width)) * hidden * 3 + c;
        const float gated = p[at] * p[at + 2 * hidden];
        sum = fmaf(gated, w[c * 9 + (dy + 1) * 3 + dx + 1], sum);
      }
  out[i] = p[std::size_t(row) * 3 * hidden + hidden + c] * sum;
}
__global__ void Rope(float* x, int h, int w, int heads) {
  constexpr int dim = 64, half = 32;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h * w * heads * half)
    return;
  const int d = i % half, row = i / (heads * half);
  const int y = (row / 4) / (w / 2) * 2 + row % 4 / 2;
  const int col = (row / 4) % (w / 2) * 2 + row % 2;
  const float angle = (d < 16 ? y : col) * powf(1e6F, -float(d % 16) / 16);
  float sn, cs;
  sincosf(angle, &sn, &cs);
  const int at = (i / half) * dim + d;
  const float a = x[at], b = x[at + half];
  x[at] = a * cs - b * sn;
  x[at + half] = b * cs + a * sn;
}
using Half = std::uint16_t;
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
__device__ __forceinline__ v16h Fragment(const Half* p) {
  union {
    v16h f;
    uint4 u[2];
  } c;
  c.u[0] = reinterpret_cast<const uint4*>(p)[0];
  c.u[1] = reinterpret_cast<const uint4*>(p)[1];
  return c.f;
}
__device__ __forceinline__ unsigned OtherHalf(unsigned x) {
  return __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xfedcba98u, false,
                                      false);
}
__device__ __forceinline__ float OtherHalf(float x) {
  return __builtin_bit_cast(float, OtherHalf(__builtin_bit_cast(unsigned, x)));
}
__device__ __forceinline__ v8f Mma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
// x = high + low with both parts F16; about 22 significant bits of x.
__device__ __forceinline__ void Split(float x, _Float16& high, _Float16& low) {
  high = static_cast<_Float16>(x);
  low = static_cast<_Float16>(x - float(high));
}
// Bidirectional D = 64 attention of FP32 Q, K and V on the matrix cores
// without narrowing them: Q, K, probabilities and V are split into F16
// high/low pairs and each product is the sum of three WMMAs (high x high,
// high x low, low x high; low x low is below FP32 rounding), the two small
// terms in their own accumulators. Every 32-key tile accumulates from zero
// and is added to the FP32 state, and the softmax stays FP32. A block owns
// 128 (head, query) rows of one KV head, 16 queries of one head per wave.
// The next 32-key K/V tile is loaded into registers during the current
// tile's products and split once into LDS. Waves compute S^T = K Q^T and
// O^T = V^T P^T, so lane L holds query L % 16 and keys (or dimensions)
// 2i + L / 16, as in WmmaAttn.
__launch_bounds__(256) __global__
    void SplitAttn64(const float* __restrict__ q, const float* __restrict__ k,
                     const float* __restrict__ v, float* __restrict__ y, int n,
                     int heads, int kh) {
  constexpr int D = 64, kKeys = 32, kSteps = D / 16, kKStride = D + 8,
                kVStride = kKeys + 8;
  __shared__ __attribute__((aligned(16))) Half sk[2][kKeys * kKStride];
  __shared__ __attribute__((aligned(16))) Half svt[2][D * kVStride];
  const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
  const int col = lane % 16, half = lane / 16;
  const int group = heads / kh, g = blockIdx.z;
  const int per_head = 128 / group;  // queries of each head in the block
  const int h = g * group + wave % group;
  const int w0 = blockIdx.x * per_head + wave / group * 16;
  const int row = w0 + col, stride = kh * D;
  const bool live = row < n;
  const float scale = rsqrtf(float(D)) * 1.44269504088896341F;
  v16h qh[kSteps], ql[kSteps];
  {
    const float* qr = q + (std::size_t(live ? row : 0) * heads + h) * D;
#pragma unroll
    for (int st = 0; st < kSteps; ++st)
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float4 f = live ? reinterpret_cast<const float4*>(qr)[st * 4 + c]
                              : make_float4(0, 0, 0, 0);
        const float e[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          _Float16 a, b;
          Split(e[j] * scale, a, b);
          qh[st][c * 4 + j] = a;
          ql[st][c * 4 + j] = b;
        }
      }
  }
  v8f acc[kSteps] = {};
  float maximum = -INFINITY, sum = 0;
  // Each thread loads eight consecutive values of one key's K and V row.
  const int lk = tid / 8, ld = tid % 8 * 8;
  float kv[2][8];
  const auto load = [&](int base) {
    const int t = base + lk;
#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const float* src = (m ? v : k) + std::size_t(t) * stride + g * D + ld;
      float4 a = make_float4(0, 0, 0, 0), b = a;
      if (t < n) {
        a = reinterpret_cast<const float4*>(src)[0];
        b = reinterpret_cast<const float4*>(src)[1];
      }
      kv[m][0] = a.x, kv[m][1] = a.y, kv[m][2] = a.z, kv[m][3] = a.w;
      kv[m][4] = b.x, kv[m][5] = b.y, kv[m][6] = b.z, kv[m][7] = b.w;
    }
  };
  load(0);
  for (int base = 0; base < n; base += kKeys) {
    __syncthreads();
    {
      union {
        _Float16 h[8];
        uint4 u;
      } hi, lo;
#pragma unroll
      for (int j = 0; j < 8; ++j)
        Split(kv[0][j], hi.h[j], lo.h[j]);
      *reinterpret_cast<uint4*>(&sk[0][lk * kKStride + ld]) = hi.u;
      *reinterpret_cast<uint4*>(&sk[1][lk * kKStride + ld]) = lo.u;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        _Float16 a, b;
        Split(kv[1][j], a, b);
        svt[0][(ld + j) * kVStride + lk] = __builtin_bit_cast(Half, a);
        svt[1][(ld + j) * kVStride + lk] = __builtin_bit_cast(Half, b);
      }
    }
    __syncthreads();
    if (base + kKeys < n)
      load(base + kKeys);
    if (w0 >= n)
      continue;
    v8f s[2], cross[2] = {}, main[2] = {};
#pragma unroll
    for (int st = 0; st < kSteps; ++st)
#pragma unroll
      for (int kb = 0; kb < 2; ++kb) {
        const v16h kf = Fragment(&sk[0][(kb * 16 + col) * kKStride + st * 16]);
        const v16h kl = Fragment(&sk[1][(kb * 16 + col) * kKStride + st * 16]);
        cross[kb] = Mma(kl, qh[st], cross[kb]);
        cross[kb] = Mma(kf, ql[st], cross[kb]);
        main[kb] = Mma(kf, qh[st], main[kb]);
      }
#pragma unroll
    for (int kb = 0; kb < 2; ++kb)
      s[kb] = main[kb] + cross[kb];
    // Only the last tile holds keys past n.
    if (base + kKeys > n)
#pragma unroll
      for (int kb = 0; kb < 2; ++kb)
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (base + kb * 16 + 2 * i + half >= n)
            s[kb][i] = -INFINITY;
    float tile_max = -INFINITY;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        tile_max = fmaxf(tile_max, s[kb][i]);
    tile_max = fmaxf(tile_max, OtherHalf(tile_max));
    const float next = fmaxf(maximum, tile_max);
    const float alpha = __builtin_amdgcn_exp2f(maximum - next);
    maximum = next;
    unsigned own_h[2][4], own_l[2][4];
    float local = 0;
#pragma unroll
    for (int kb = 0; kb < 2; ++kb)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float a = __builtin_amdgcn_exp2f(s[kb][2 * j] - next),
                    b = __builtin_amdgcn_exp2f(s[kb][2 * j + 1] - next);
        local += a + b;
        _Float16 ah, al, bh, bl;
        Split(a, ah, al);
        Split(b, bh, bl);
        own_h[kb][j] = unsigned(__builtin_bit_cast(Half, ah)) |
                       unsigned(__builtin_bit_cast(Half, bh)) << 16;
        own_l[kb][j] = unsigned(__builtin_bit_cast(Half, al)) |
                       unsigned(__builtin_bit_cast(Half, bl)) << 16;
      }
    sum = sum * alpha + local;
    v8f tile[kSteps] = {}, small[kSteps] = {};
#pragma unroll
    for (int kb = 0; kb < 2; ++kb) {
      // Key 2m comes from the first half's element m, key 2m + 1 from the
      // second half's.
      union {
        v16h f;
        unsigned u[8];
      } ph, pl;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const unsigned oh = OtherHalf(own_h[kb][j]),
                       ol = OtherHalf(own_l[kb][j]);
        const unsigned fh = half ? oh : own_h[kb][j],
                       sh = half ? own_h[kb][j] : oh;
        const unsigned fl = half ? ol : own_l[kb][j],
                       sl = half ? own_l[kb][j] : ol;
        ph.u[2 * j] = (fh & 0xffffu) | sh << 16;
        ph.u[2 * j + 1] = fh >> 16 | (sh & 0xffff0000u);
        pl.u[2 * j] = (fl & 0xffffu) | sl << 16;
        pl.u[2 * j + 1] = fl >> 16 | (sl & 0xffff0000u);
      }
#pragma unroll
      for (int t = 0; t < kSteps; ++t) {
        const v16h vh = Fragment(&svt[0][(t * 16 + col) * kVStride + kb * 16]);
        const v16h vl = Fragment(&svt[1][(t * 16 + col) * kVStride + kb * 16]);
        small[t] = Mma(vl, ph.f, small[t]);
        small[t] = Mma(vh, pl.f, small[t]);
        tile[t] = Mma(vh, ph.f, tile[t]);
      }
    }
#pragma unroll
    for (int t = 0; t < kSteps; ++t)
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[t][i] = fmaf(acc[t][i], alpha, tile[t][i] + small[t][i]);
  }
  const float total = sum + OtherHalf(sum);
  if (!live)
    return;
  float* out = y + (std::size_t(row) * heads + h) * D;
#pragma unroll
  for (int t = 0; t < kSteps; ++t)
#pragma unroll
    for (int i = 0; i < 8; ++i)
      out[t * 16 + 2 * i + half] = acc[t][i] / total;
}
}  // namespace
void SpatialConv(const float* p, const float* w, float* out, int h, int width,
                 int hidden, hipStream_t s) {
  Conv<<<(std::size_t(h) * width * hidden + 255) / 256, 256, 0, s>>>(
      p, w, out, h, width, hidden);
}
void AxialRope(float* x, int h, int w, int heads, hipStream_t s) {
  Rope<<<(std::size_t(h) * w * heads * 32 + 255) / 256, 256, 0, s>>>(x, h, w,
                                                                     heads);
}
void Attention64(const float* q, const float* k, const float* v, float* out,
                 int n, int heads, int kv_heads, hipStream_t s) {
  const int per_head = 128 / (heads / kv_heads);
  SplitAttn64<<<dim3((n + per_head - 1) / per_head, 1, kv_heads), 256, 0, s>>>(
      q, k, v, out, n, heads, kv_heads);
}
}  // namespace gufo::models::lfm3_vl::gpu
