// Row kernels (npu/rowops.cpp) against the host FP32 operators.
#include <cmath>
#include <cstdio>
#include <random>

#include "../host_ops.hpp"
#include "../rowops.hpp"
#include "bfp.hpp"
#include "npu_runtime.hpp"

using namespace gufo::models::lfm3::npu;

static const char* kNorm = R"(      %step = index.constant 16 : index
      %n = index.constant @N : index
      %zero = vector.splatf 0.0 : vector<16xf32>
      %fzero = scalar.constant 0.0 : f32
      %sq = scf.for %i = [%c0 to %n step %step](%acc = %zero : vector<16xf32>) -> (vector<16xf32>) {
        %v = vector.load %x[%i] : view<@Nxf32> -> vector<16xf32>
        %p = func.call @fmul(%v, %v) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %s = vector.addf %acc, %p : vector<16xf32>
        scf.yield %s : vector<16xf32>
      }
      %total = vector.reduce<addf> %sq, %fzero : vector<16xf32>, f32
      %tv = vector.splat %total : vector<16xf32>
      %inv_n = vector.splatf @INV : vector<16xf32>
      %eps = vector.splatf @EPS : vector<16xf32>
      %mean = func.call @fmul(%tv, %inv_n) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
      %me = vector.addf %mean, %eps : vector<16xf32>
      %scale = func.call @rsqrt(%me) : (vector<16xf32>) -> (vector<16xf32>)
      scf.for %i = [%c0 to %n step %step] {
        %v = vector.load %x[%i] : view<@Nxf32> -> vector<16xf32>
        %g = vector.load %w[%i] : view<@Nxf32> -> vector<16xf32>
        %t = func.call @fmul(%v, %scale) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %u = func.call @fmul(%t, %g) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        vector.store %u, %y[%i] : vector<16xf32>, view<@Nxf32>
      }
)";
static const char* kSwiGlu = R"(      %step = index.constant 16 : index
      %n = index.constant @N : index
      scf.for %i = [%c0 to %n step %step] {
        %gv = vector.load %g[%i] : view<@Nxf32> -> vector<16xf32>
        %uv = vector.load %u[%i] : view<@Nxf32> -> vector<16xf32>
        %sg = func.call @sigmoid(%gv) : (vector<16xf32>) -> (vector<16xf32>)
        %silu = func.call @fmul(%gv, %sg) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %o = func.call @fmul(%silu, %uv) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        vector.store %o, %y[%i] : vector<16xf32>, view<@Nxf32>
      }
)";

static const char* kAddNormPack = R"(      %step = index.constant 16 : index
      %n = index.constant @N : index
      %zero = vector.splatf 0.0 : vector<16xf32>
      %fzero = scalar.constant 0.0 : f32
      %sq = scf.for %i = [%c0 to %n step %step](%acc = %zero : vector<16xf32>) -> (vector<16xf32>) {
        %xv = vector.load %x[%i] : view<@Nxf32> -> vector<16xf32>
        %yv = vector.load %y[%i] : view<@Nxf32> -> vector<16xf32>
        %s = vector.addf %xv, %yv : vector<16xf32>
        vector.store %s, %x2[%i] : vector<16xf32>, view<@Nxf32>
        %p = func.call @fmul(%s, %s) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %a = vector.addf %acc, %p : vector<16xf32>
        scf.yield %a : vector<16xf32>
      }
      %total = vector.reduce<addf> %sq, %fzero : vector<16xf32>, f32
      %tv = vector.splat %total : vector<16xf32>
      %inv_n = vector.splatf @INV : vector<16xf32>
      %eps = vector.splatf @EPS : vector<16xf32>
      %mean = func.call @fmul(%tv, %inv_n) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
      %me = vector.addf %mean, %eps : vector<16xf32>
      %scale = func.call @rsqrt(%me) : (vector<16xf32>) -> (vector<16xf32>)
      scf.for %i = [%c0 to %n step %step] {
        %v = vector.load %x2[%i] : view<@Nxf32> -> vector<16xf32>
        %g = vector.load %w[%i] : view<@Nxf32> -> vector<16xf32>
        %t = func.call @fmul(%v, %scale) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %u = func.call @fmul(%t, %g) : (vector<16xf32>, vector<16xf32>) -> (vector<16xf32>)
        %h = vector.fptrunc %u : vector<16xf32> to vector<16xbf16>
        vector.store %h, %xb[%i] : vector<16xbf16>, view<@Nxbf16>
      }
)";

static std::string Fill(std::string s, std::uint32_t n, float eps = 1e-5f) {
  auto rep = [&](const std::string& k, const std::string& v) {
    for (auto at = s.find(k); at != std::string::npos; at = s.find(k, at + v.size())) s.replace(at, k.size(), v);
  };
  char inv[64], e[64];
  std::snprintf(inv, sizeof inv, "%.9e", 1.0 / n);
  std::snprintf(e, sizeof e, "%.9e", double(eps));
  rep("@INV", inv);
  rep("@EPS", e);
  rep("@N", std::to_string(n));
  return s;
}

int main() {
  lfm3::npu::Runtime npu;
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(-3, 3);
  int failures = 0;
  auto report = [&](const char* name, double worst, double tol) {
    const bool ok = worst <= tol;
    failures += !ok;
    std::printf("%-10s max err %.3g (tolerance %.3g) %s\n", name, worst, tol, ok ? "ok" : "FAIL");
  };
  {  // RMSNorm of 24 rows of 2048.
    const std::uint32_t R = 24, H = 2048;
    RowKernel k{"norm", 3, {{"x", 0, H, 0, H}, {"w", 1, 0, 0, H}}, {{"y", 2, H, 0, H}}, Fill(kNorm, H), R, 1};
    const int id = npu.Compile(RowKernelProgram(k), "norm.loom", "@norm");
    std::vector<float> x(R * H), w(H), y(R * H), ref(R * H);
    for (auto& v : x) v = u(rng);
    for (auto& v : w) v = 0.5f + std::abs(u(rng)) / 3;
    const int bx = npu.Upload(std::span<const float>(x)), bw = npu.Upload(std::span<const float>(w)), by = npu.Alloc(y.size() * 4);
    const int bind[] = {bx, bw, by};
    npu.Dispatch(id, bind);
    npu.Read(by, 0, y.data(), y.size() * 4);
    RmsNormRows(x.data(), R, H, w.data(), 1e-5f, ref.data());
    double worst = 0;
    for (std::size_t i = 0; i < y.size(); ++i) worst = std::max(worst, double(std::abs(y[i] - ref[i]) / (1 + std::abs(ref[i]))));
    report("rmsnorm", worst, 1e-5);
  }
  {  // SwiGLU over 9 rows of gate/up 7680, in pieces of 512.
    const std::uint32_t R = 9, I = 7680, P = 512;
    RowKernel k{"swiglu", 2, {{"g", 0, 2 * I, 0, P}, {"u", 0, 2 * I, I, P}}, {{"y", 1, I, 0, P}}, Fill(kSwiGlu, P), R, I / P};
    const int id = npu.Compile(RowKernelProgram(k), "swiglu.loom", "@swiglu");
    std::vector<float> gu(R * 2 * I), y(R * I);
    for (auto& v : gu) v = u(rng) * 3;
    const int bg = npu.Upload(std::span<const float>(gu)), by = npu.Alloc(y.size() * 4);
    const int bind[] = {bg, by};
    npu.Dispatch(id, bind);
    npu.Read(by, 0, y.data(), y.size() * 4);
    double worst = 0;
    for (std::uint32_t r = 0; r < R; ++r)
      for (std::uint32_t i = 0; i < I; ++i) {
        const float g = gu[r * 2 * I + i], uu = gu[r * 2 * I + I + i];
        const float ref = Silu(g) * uu;
        worst = std::max(worst, double(std::abs(y[r * I + i] - ref) / (1 + std::abs(ref))));
      }
    report("swiglu", worst, 1e-5);
  }
  {  // Residual add from blocked Yb, RMSNorm, BF16 blocked Xb out (11 rows).
    using L = RowOperand::Layout;
    const std::uint32_t R = 11, H = 2048, G = H / 8, B = (R + 7) / 8;
    RowKernel k{"addnorm", 5,
                {{"x", 0, H, 0, H}, {"y", 1, 0, 0, H, false, L::kBlocked, G}, {"w", 2, 0, 0, H}},
                {{"x2", 3, H, 0, H}, {"xb", 4, 0, 0, H, true, L::kBlocked, G}},
                Fill(kAddNormPack, H), R, 1};
    const int id = npu.Compile(RowKernelProgram(k), "addnorm.loom", "@addnorm");
    std::vector<float> x(R * H), yb(B * G * 64 + 64), w(H), x2(R * H);
    std::vector<std::uint16_t> xb(B * G * 64 + 64);
    for (auto& v : x) v = u(rng);
    for (auto& v : yb) v = u(rng);
    for (auto& v : w) v = 0.5f + std::abs(u(rng)) / 3;
    const int bx = npu.Upload(std::span<const float>(x)), by = npu.Upload(std::span<const float>(yb)),
              bw = npu.Upload(std::span<const float>(w)), bx2 = npu.Alloc(x2.size() * 4), bxb = npu.Alloc(xb.size() * 2);
    const int bind[] = {bx, by, bw, bx2, bxb};
    npu.Dispatch(id, bind);
    npu.Read(bx2, 0, x2.data(), x2.size() * 4);
    npu.Read(bxb, 0, xb.data(), xb.size() * 2);
    double worst = 0, worst_b = 0;
    for (std::uint32_t r = 0; r < R; ++r) {
      std::vector<float> s(H), z(H);
      for (std::uint32_t i = 0; i < H; ++i) {
        const float yv = yb[((std::size_t(r / 8) * G + i / 8) * 8 + r % 8) * 8 + i % 8];
        s[i] = x[r * H + i] + yv;
        worst = std::max(worst, double(std::abs(x2[r * H + i] - s[i])));
      }
      RmsNorm(s.data(), H, w.data(), 1e-5f, z.data());
      for (std::uint32_t i = 0; i < H; ++i) {
        const float got = lfm3::npu::FromBF16(xb[((std::size_t(r / 8) * G + i / 8) * 8 + r % 8) * 8 + i % 8]);
        worst_b = std::max(worst_b, double(std::abs(got - z[i]) / (1e-3 + std::abs(z[i]))));
      }
    }
    report("add", worst, 1e-6);
    report("norm->xb", worst_b, 8e-3);
  }
  return failures;
}
