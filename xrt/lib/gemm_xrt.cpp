// gufo_xrt_gemm.h on q8gemm.xclbin (xrt/iron/gemm). A matrix splits into
// parts of at most 2048 codes (K) by 4096 outputs (N), laid out in one weight
// arena as q8gemm.py streams them: [stream 16][chunk][tile][parity 2]. A
// product over any rows becomes jobs of up to 8 G rows per part (G chunks <= 4
// for the cores' A buffer, G tiles <= 32 for their accumulators); the jobs of a
// call or of a batch run as one dispatch whose instruction stream concatenates
// the jobs' streams (cached per batch shape).

#include <dlfcn.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "../gufo_xrt_gemm.h"

namespace {

thread_local std::string g_error;

constexpr int kStreams = 16, kCores = 32, kRec = 4352, kCodes = 4096, kMaxA = 8,
              kMaxTiles = 32;
constexpr int kMaxK = 2048,
              kMaxN = kCores * 8 * kMaxTiles / 2;  // two row groups per job
constexpr size_t kRecWords = kRec / 4, kOutWords = 128;
constexpr size_t kABytes = 16u << 20, kCBytes = 16u << 20;
using Dispatch = int64_t (*)(int32_t, int32_t, int32_t, int32_t, int32_t,
                             int32_t, int32_t, int32_t, int32_t, int32_t,
                             uint32_t**);

struct Part {
  int k0, kp, n0, np, kc, nt;
  size_t b_words;          // offset in the arena
  std::vector<float> top;  // per output: the largest block scale / 32767
};

struct Matrix {
  int n, k;
  std::vector<Part> parts;
};

struct Job {
  const Matrix* m;
  const Part* p;
  int call, r0, nr, g;
  size_t a_words, c_words;
};

bool Timing() {
  static const bool on = std::getenv("GUFO_XRT_GEMM_TIMING") != nullptr;
  return on;
}

double Ms(std::chrono::steady_clock::time_point a,
          std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

struct gufo_xrt_gemm {
  xrt::device dev{0};
  xrt::hw_context ctx;
  xrt::kernel kernel;
  void* lib{};
  Dispatch generate{};
  xrt::bo a, c, arena;
  size_t arena_bytes{}, arena_used{};
  std::vector<Matrix> matrices;
  std::map<std::vector<int>, std::pair<xrt::bo, uint32_t>>
      streams;  // (id, rows)... -> instructions
  std::mutex mutex;
  std::vector<int16_t> q;
  std::vector<float> r;
  std::vector<int32_t> out;  // C copied out of the (uncached) buffer

  explicit gufo_xrt_gemm(const std::string& dir) {
    xrt::xclbin xb(dir + "/q8gemm.xclbin");
    dev.register_xclbin(xb);
    ctx = xrt::hw_context(dev, xb.get_uuid());
    for (auto& k : xb.get_kernels())
      if (k.get_name().rfind("MLIR_AIE", 0) == 0)
        kernel = xrt::kernel(ctx, k.get_name());
    lib = dlopen((dir + "/q8gemm_dispatch.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib)
      throw std::runtime_error(std::string("q8gemm_dispatch.so: ") + dlerror());
    generate = reinterpret_cast<Dispatch>(dlsym(lib, "dispatch_generate"));
    if (!generate)
      throw std::runtime_error("q8gemm_dispatch.so: no dispatch_generate");
    a = xrt::bo(dev, kABytes, xrt::bo::flags::host_only, kernel.group_id(3));
    c = xrt::bo(dev, kCBytes, xrt::bo::flags::host_only, kernel.group_id(5));
    const char* mb = std::getenv("GUFO_XRT_GEMM_ARENA_MB");
    arena_bytes = size_t(mb ? std::atoi(mb) : 1024) << 20;
  }
  ~gufo_xrt_gemm() {
    streams.clear();
    if (lib)
      dlclose(lib);
  }

  int Upload(int n, int k, const int8_t* codes, const float* scales) {
    if (n <= 0 || k <= 0 || k % 32)
      throw std::invalid_argument(
          "q8gemm: k must be a positive multiple of 32");
    if (!arena)
      arena = xrt::bo(dev, arena_bytes, xrt::bo::flags::host_only,
                      kernel.group_id(4));
    auto* base = static_cast<uint8_t*>(arena.map());
    Matrix m{n, k, {}};
    for (int k0 = 0; k0 < k; k0 += kMaxK)
      for (int n0 = 0; n0 < n; n0 += kMaxN) {
        Part p;
        p.k0 = k0, p.kp = std::min(kMaxK, k - k0), p.n0 = n0,
        p.np = std::min(kMaxN, n - n0);
        p.kc = (p.kp + 511) / 512,
        p.nt = (p.np + kCores * 8 - 1) / (kCores * 8);
        const size_t bytes = size_t(kStreams) * p.kc * p.nt * 2 * kRec;
        if (arena_used + bytes > arena_bytes)
          throw std::runtime_error(
              "q8gemm: weight arena full (GUFO_XRT_GEMM_ARENA_MB)");
        p.b_words = arena_used / 4;
        uint8_t* w = base + arena_used;
        std::memset(w, 0, bytes);
        const int blocks = p.kp / 32;
        p.top.assign(p.np, 0.0f);
        for (int j = 0; j < p.np; ++j) {
          const float* s = scales + size_t(n0 + j) * (k / 32) + k0 / 32;
          float top = 0;
          for (int b = 0; b < blocks; ++b)
            top = std::max(top, s[b]);
          if (!(top > 0))
            top = 1;
          p.top[j] = top / 32767.0f;
          // output j: core j / (8 nt), tile (j / 8) % nt, column j % 8
          const int core = j / (8 * p.nt), t = (j / 8) % p.nt, nn = j % 8;
          const int stream = core / 2, parity = core % 2;
          const int8_t* row = codes + size_t(n0 + j) * k + k0;
          for (int kc = 0; kc < p.kc; ++kc) {
            uint8_t* rec = w + ((size_t(stream) * p.kc + kc) * p.nt * 2 +
                                size_t(t) * 2 + parity) *
                                   kRec;
            auto* u = reinterpret_cast<int16_t*>(rec + kCodes);
            for (int blk = 0; blk < 16; ++blk) {
              const int b = kc * 16 + blk;
              if (b >= blocks)
                break;
              u[blk * 8 + nn] = int16_t(std::lround(s[b] / top * 32767.0f));
              for (int kk = 0; kk < 32; ++kk)
                rec[blk * 256 + kk * 8 + nn] = uint8_t(row[b * 32 + kk]);
            }
          }
        }
        arena.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, arena_used);
        arena_used += (bytes + 4095) / 4096 * 4096;
        m.parts.push_back(std::move(p));
      }
    matrices.push_back(std::move(m));
    return int(matrices.size()) - 1;
  }

  // Jobs of the calls, with their A and C offsets.
  std::vector<Job> Plan(int count, const int* ids, const int* rows) {
    std::vector<Job> jobs;
    size_t a_words = 0, c_words = 0;
    for (int i = 0; i < count; ++i) {
      const Matrix& m = matrices.at(size_t(ids[i]));
      for (const Part& p : m.parts) {
        const int G =
            std::max(1, std::min(kMaxA / (2 * p.kc), kMaxTiles / p.nt));
        for (int r0 = 0; r0 < rows[i]; r0 += 8 * G) {
          Job j{&m, &p,      i,      r0, std::min(8 * G, rows[i] - r0),
                0,  a_words, c_words};
          j.g = (j.nr + 7) / 8;
          a_words += size_t(j.g) * p.kc * 2 * kRecWords;
          c_words += size_t(8) * j.g * p.nt * 4 * kOutWords;
          jobs.push_back(j);
        }
      }
    }
    if (a_words * 4 > kABytes || c_words * 4 > kCBytes)
      throw std::runtime_error("q8gemm: batch too large");
    return jobs;
  }

  std::pair<xrt::bo, uint32_t>& Stream(const std::vector<Job>& jobs,
                                       const std::vector<int>& key) {
    if (auto it = streams.find(key); it != streams.end())
      return it->second;
    std::vector<uint32_t> txn(4, 0);
    uint32_t ops = 0;
    for (const Job& j : jobs) {
      const Part& p = *j.p;
      uint32_t* words = nullptr;
      const int64_t n =
          generate(j.g, p.kc, p.nt, 2 * j.g * p.kc, j.g * p.nt, 2 * p.nt,
                   2 * p.kc * p.nt, int32_t(j.a_words), int32_t(p.b_words),
                   int32_t(j.c_words), &words);
      if (n <= 4)
        throw std::runtime_error("q8gemm: instruction generation failed");
      if (ops == 0)
        txn[0] = words[0], txn[1] = words[1];
      ops += words[2];
      txn.insert(txn.end(), words + 4, words + n);
    }
    txn[2] = ops;
    txn[3] = uint32_t(txn.size() * 4);
    xrt::bo bo(dev, txn.size() * 4, xrt::bo::flags::cacheable,
               kernel.group_id(1));
    bo.write(txn.data(), txn.size() * 4, 0);
    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return streams
        .emplace(key, std::make_pair(std::move(bo), uint32_t(txn.size())))
        .first->second;
  }

  void Pack(const Job& j, const float* x) {
    const Part& p = *j.p;
    const int width = p.kc * 512;
    q.assign(size_t(j.g) * 8 * width, 0);
    r.assign(size_t(j.g) * 8, 0.0f);
    for (int i = 0; i < j.nr; ++i) {
      const float* xi = x + size_t(j.r0 + i) * j.m->k + p.k0;
      float top = 0;
      for (int c = 0; c < p.kp; ++c)
        top = std::max(top, std::fabs(xi[c]));
      const float s = top > 0 ? top / 32767.0f : 1.0f,
                  inv = top > 0 ? 32767.0f / top : 0.0f;
      r[i] = s;
      int16_t* qi = &q[size_t(i) * width];
      for (int c = 0; c < p.kp;
           ++c) {  // |x inv| <= 32767; round half away from zero
        const float t = xi[c] * inv;
        qi[c] = int16_t(t + (t >= 0 ? 0.5f : -0.5f));
      }
    }
    // A records [group][chunk][half]: [blk 8][k4 8][mm 8][k 4]
    auto* base = static_cast<uint8_t*>(a.map()) + j.a_words * 4;
    for (int gi = 0; gi < j.g; ++gi)
      for (int kc = 0; kc < p.kc; ++kc)
        for (int h = 0; h < 2; ++h) {
          auto* dst = reinterpret_cast<int16_t*>(
              base + ((size_t(gi) * p.kc + kc) * 2 + h) * kRec);
          for (int blk = 0; blk < 8; ++blk)
            for (int k4 = 0; k4 < 8; ++k4)
              for (int mm = 0; mm < 8; ++mm)
                std::memcpy(dst + blk * 256 + k4 * 32 + mm * 4,
                            &q[(size_t(gi) * 8 + mm) * width + kc * 512 +
                               h * 256 + blk * 32 + k4 * 4],
                            8);
        }
  }

  void Unpack(const Job& j, const std::vector<float>& r, float* y) {
    const Part& p = *j.p;
    const size_t gt = size_t(j.g) * p.nt;
    const int32_t* base = out.data() + j.c_words;
    // column col at col gt 512 words: records [tile][group][core row jj][hi 64
    // | lo 64]; core = 4 col + jj
    for (int col = 0; col < 8; ++col)
      for (int jj = 0; jj < 4; ++jj) {
        const int core = 4 * col + jj;
        for (int t = 0; t < p.nt; ++t) {
          const int out0 = (core * p.nt + t) * 8;
          if (out0 >= p.np)
            continue;
          const int outs = std::min(8, p.np - out0);
          for (int gi = 0; gi < j.g; ++gi) {
            const int32_t* w =
                base +
                (size_t(col) * gt + (size_t(t) * j.g + gi)) * 4 * kOutWords +
                jj * kOutWords;
            for (int mm = 0; mm < 8; ++mm) {
              const int row = gi * 8 + mm;
              if (row >= j.nr)
                break;
              float* dst = y + size_t(j.r0 + row) * j.m->n + p.n0 + out0;
              const float rs = r[row];
              const int32_t *hi = w + mm * 8, *lo = w + 64 + mm * 8;
              const float* top = p.top.data() + out0;
              if (outs == 8) {
                for (int nn = 0; nn < 8; ++nn)
                  dst[nn] +=
                      (float(hi[nn]) * 16777216.0f + float(lo[nn] & 0xFFFFFF)) *
                      (rs * top[nn]);
              } else {
                for (int nn = 0; nn < outs; ++nn)
                  dst[nn] +=
                      (float(hi[nn]) * 16777216.0f + float(lo[nn] & 0xFFFFFF)) *
                      (rs * top[nn]);
              }
            }
          }
        }
      }
  }

  void Run(int count, const int* ids, const float* const* xs, const int* rows,
           float* const* ys) {
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<Job> jobs = Plan(count, ids, rows);
    std::vector<int> key;
    for (int i = 0; i < count; ++i)
      key.push_back(ids[i]), key.push_back(rows[i]);
    auto& ins = Stream(jobs, key);
    size_t a_end = 0, c_end = 0;
    std::vector<std::vector<float>> scales;
    for (const Job& j : jobs) {
      Pack(j, xs[j.call]);
      scales.push_back(r);
      a_end = j.a_words + size_t(j.g) * j.p->kc * 2 * kRecWords;
      c_end = j.c_words + size_t(8) * j.g * j.p->nt * 4 * kOutWords;
    }
    a.sync(XCL_BO_SYNC_BO_TO_DEVICE, a_end * 4, 0);
    const auto t1 = std::chrono::steady_clock::now();
    kernel(3u, ins.first, ins.second, a, arena, c).wait();
    const auto t2 = std::chrono::steady_clock::now();
    c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, c_end * 4, 0);
    out.resize(c_end);
    c.read(out.data(), c_end * 4, 0);
    for (int i = 0; i < count; ++i)
      std::fill_n(ys[i], size_t(rows[i]) * matrices[size_t(ids[i])].n, 0.0f);
    for (size_t i = 0; i < jobs.size(); ++i)
      Unpack(jobs[i], scales[i], ys[jobs[i].call]);
    if (Timing())
      std::fprintf(stderr,
                   "xrt gemm batch of %d (%zu jobs): stage %.3f, run %.3f, "
                   "unpack %.3f ms\n",
                   count, jobs.size(), Ms(t0, t1), Ms(t1, t2),
                   Ms(t2, std::chrono::steady_clock::now()));
  }
};

extern "C" {

gufo_xrt_gemm* gufo_xrt_gemm_create(const char* dir) {
  try {
    return new gufo_xrt_gemm(dir);
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}

void gufo_xrt_gemm_destroy(gufo_xrt_gemm* g) {
  delete g;
}

const char* gufo_xrt_gemm_last_error(void) {
  return g_error.c_str();
}

int gufo_xrt_gemm_upload(gufo_xrt_gemm* g, int n, int k, const int8_t* codes,
                         const float* scales) {
  try {
    std::lock_guard lock(g->mutex);
    return g->Upload(n, k, codes, scales);
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

int gufo_xrt_gemm_run_batch(gufo_xrt_gemm* g, int count, const int* ids,
                            const float* const* xs, const int* rows,
                            float* const* ys) {
  try {
    std::lock_guard lock(g->mutex);
    g->Run(count, ids, xs, rows, ys);
    return 0;
  } catch (const std::exception& e) {
    g_error = e.what();
    return -1;
  }
}

int gufo_xrt_gemm_run(gufo_xrt_gemm* g, int id, const float* x, int rows,
                      float* y) {
  return gufo_xrt_gemm_run_batch(g, 1, &id, &x, &rows, &y);
}

}  // extern "C"
