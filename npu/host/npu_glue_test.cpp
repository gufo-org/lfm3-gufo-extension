// Glue kernels (npu/glue.cpp) against host FP32 at LFM3 1.7B widths.
#include <cmath>
#include <cstdio>
#include <random>

#include "../glue.hpp"
#include "../host_ops.hpp"
#include "bfp.hpp"
#include "npu_runtime.hpp"

using namespace gufo::models::lfm3::npu;
static std::size_t B(std::uint32_t r, std::uint32_t g, std::uint32_t e) {  // blocked element index
  return ((std::size_t(r / 8) * g + e / 8) * 8 + r % 8) * 8 + e % 8;
}
int main() {
  lfm3::npu::Runtime npu;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> u(-2, 2);
  const std::uint32_t R = 11, H = 2048, I = 7680, NB = (R + 7) / 8;
  int fails = 0;
  auto rep = [&](const char* n, double e, double tol) { std::printf("%-12s %.3g %s\n", n, e, e <= tol ? "ok" : "FAIL"); fails += e > tol; };
  auto vec = [&](std::size_t n, float lo = -2, float hi = 2) { std::vector<float> v(n); std::uniform_real_distribution<float> d(lo, hi); for (auto& x : v) x = d(rng); return v; };
  {  // normpack with blocked residual and post norm, z out
    auto k = NormPackKernel(H, R, Residual::kBlocked, true, false, 1e-5f);
    const int id = npu.Compile(RowKernelProgram(k), "np.loom", "@" + k.name);
    auto x = vec(R * H), yb = vec(NB * H * 8 + 64), w = vec(2 * H, 0.5f, 1.5f);
    const int bx = npu.Upload(std::span<const float>(x)), by = npu.Upload(std::span<const float>(yb)), bw = npu.Upload(std::span<const float>(w));
    const int bx2 = npu.Alloc(R * H * 4), bxb = npu.Alloc((NB * H * 8 + 64) * 2), bz = npu.Alloc(R * H * 4);
    const int bind[] = {bx, by, bw, bx2, bxb};
    npu.Dispatch(id, bind);
    std::vector<float> x2(R * H), z(R * H);
    std::vector<std::uint16_t> xb(NB * H * 8 + 64);
    npu.Read(bx2, 0, x2.data(), x2.size() * 4); (void)bz; npu.Read(bxb, 0, xb.data(), xb.size() * 2);
    double ex = 0, ez = 0, eb = 0;
    for (std::uint32_t r = 0; r < R; ++r) {
      std::vector<float> y(H), s(H), zz(H);
      for (std::uint32_t i = 0; i < H; ++i) y[i] = yb[B(r, H / 8, i)];
      RmsNorm(y.data(), H, w.data(), 1e-5f);
      for (std::uint32_t i = 0; i < H; ++i) s[i] = x[r * H + i] + y[i];
      RmsNorm(s.data(), H, w.data() + H, 1e-5f, zz.data());
      for (std::uint32_t i = 0; i < H; ++i) {
        ex = std::max(ex, double(std::abs(x2[r * H + i] - s[i])));
        eb = std::max(eb, double(std::abs(lfm3::npu::FromBF16(xb[B(r, H / 8, i)]) - zz[i]) / (1e-2 + std::abs(zz[i]))));
      }
    }
    rep("normpack x", ex, 5e-5); rep("normpack xb", eb, 1e-2);
  }
  {  // reduce partial
    auto k = ReducePartialKernel(H, R);
    const int id = npu.Compile(RowKernelProgram(k), "rd.loom", "@" + k.name);
    auto yp = vec(NB * 8 * H * 8 + 64 * (H / 8 + 1));
    const int by = npu.Upload(std::span<const float>(yp)), bo = npu.Alloc(R * H * 4);
    const int bind[] = {by, bo};
    npu.Dispatch(id, bind);
    std::vector<float> o(R * H);
    npu.Read(bo, 0, o.data(), o.size() * 4);
    double e = 0;
    for (std::uint32_t r = 0; r < R; ++r)
      for (std::uint32_t i = 0; i < H; ++i) {
        float s = 0;
        for (int c = 0; c < 8; ++c) s += yp[((std::size_t(r / 8) * 8 + c) * (H / 8) + i / 8) * 64 + (r % 8) * 8 + i % 8];
        e = std::max(e, double(std::abs(o[r * H + i] - s)));
      }
    rep("reduce", e, 1e-5);
  }
  {  // swiglu pack
    auto k = SwiGluPackKernel(I, R);
    const int id = npu.Compile(RowKernelProgram(k), "sw.loom", "@" + k.name);
    auto gu = vec(NB * 2 * I * 8 + 64, -6, 6);
    const int bg = npu.Upload(std::span<const float>(gu)), bo = npu.Alloc((NB * I * 8 + 64) * 2);
    const int bind[] = {bg, bo};
    npu.Dispatch(id, bind);
    std::vector<std::uint16_t> ob(NB * I * 8 + 64);
    npu.Read(bo, 0, ob.data(), ob.size() * 2);
    double e = 0;
    for (std::uint32_t r = 0; r < R; ++r)
      for (std::uint32_t i = 0; i < I; ++i) {
        const float g = gu[B(r, 2 * I / 8, i)], uu = gu[B(r, 2 * I / 8, I + i)];
        const float ref = Silu(g) * uu;
        e = std::max(e, double(std::abs(lfm3::npu::FromBF16(ob[B(r, I / 8, i)]) - ref) / (1e-2 + std::abs(ref))));
      }
    rep("swiglu", e, 1.5e-2);
  }
  {  // conv with blocked in/out, 4 slots, rows 0-4 same slot
    const std::uint32_t S = 4, L = 7;
    const int id = npu.Compile(ConvProgram(H, L, R, S, "conv"), "cv.loom", "@conv");
    auto pr = vec(NB * 3 * H * 8 + 64), hist = vec(S * (L - 1) * H), w = vec(L * H);
    std::vector<std::int32_t> meta(R * 16, 0);
    const int slots[] = {1, 1, 1, 1, 1, 0, 2, 3, 2, 0, 3};
    for (std::uint32_t r = 0; r < R; ++r) meta[r * 16] = slots[r];
    const int bp = npu.Upload(std::span<const float>(pr)), bh = npu.Upload(std::span<const float>(hist)),
              bw = npu.Upload(std::span<const float>(w)), bm = npu.Upload(std::span<const std::int32_t>(meta)),
              bo = npu.Alloc((NB * H * 8 + 64) * 2);
    const std::int32_t cn[16] = {std::int32_t(R)};
    const int bcn = npu.Upload(std::span<const std::int32_t>(cn, 16));
    const int bind[] = {bp, bh, bw, bm, bo, bcn};
    npu.Dispatch(id, bind);
    std::vector<std::uint16_t> ob(NB * H * 8 + 64);
    std::vector<float> gh(hist.size());
    npu.Read(bo, 0, ob.data(), ob.size() * 2); npu.Read(bh, 0, gh.data(), gh.size() * 4);
    double e = 0, eh = 0;
    for (std::uint32_t r = 0; r < R; ++r) {
      float* h = &hist[std::size_t(slots[r]) * (L - 1) * H];
      for (std::uint32_t ch = 0; ch < H; ++ch) {
        const float Bv = pr[B(r, 3 * H / 8, ch)], C = pr[B(r, 3 * H / 8, H + ch)], X = pr[B(r, 3 * H / 8, 2 * H + ch)];
        const float bx = Bv * X;
        float acc = w[(L - 1) * H + ch] * bx;
        for (std::uint32_t t = 0; t + 1 < L; ++t) acc += w[t * H + ch] * h[t * H + ch];
        const float ref = C * acc;
        e = std::max(e, double(std::abs(lfm3::npu::FromBF16(ob[B(r, H / 8, ch)]) - ref) / (1e-2 + std::abs(ref))));
        for (std::uint32_t t = 0; t + 2 < L; ++t) h[t * H + ch] = h[(t + 1) * H + ch];
        h[(L - 2) * H + ch] = bx;
      }
    }
    for (std::size_t i = 0; i < hist.size(); ++i) eh = std::max(eh, double(std::abs(gh[i] - hist[i])));
    rep("conv out", e, 1e-2); rep("conv hist", eh, 1e-5);
  }
  return fails;
}
