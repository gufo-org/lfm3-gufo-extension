// LFM3-1.7B Q8_0 on the XDNA NPU through XRT with our own IRON kernels
// (xrt/iron): one prefill xclbin for up to M = 1024 fresh tokens and one decode
// xclbin per token. Implements the C ABI of flm_lfm3_shim.h, so Gufo's
// xrt::Engine loads it in place of FastFlowLM's engine.
//
// model_dir is a bundle written by xrt/iron/bundle.py. A fresh prompt of up to
// M tokens runs on the prefill xclbin, whose last phase is the LM head over the
// 512 rows holding the last token (one of two instruction streams); the KV
// caches and convolution states are then copied into the decode state buffer.
// Anything else runs token by token on decode.

#include "../flm_lfm3_shim.h"

#include <dlfcn.h>

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_error;

using Dispatch = int64_t (*)(int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, uint32_t **);

struct Meta {
  std::map<std::string, std::vector<int64_t>> v;
  int64_t operator()(const std::string &k, size_t i = 0) const {
    auto it = v.find(k);
    if (it == v.end() || it->second.size() <= i) throw std::runtime_error("bundle.txt lacks " + k);
    return it->second[i];
  }
  const std::vector<int64_t> &list(const std::string &k) const {
    auto it = v.find(k);
    if (it == v.end()) throw std::runtime_error("bundle.txt lacks " + k);
    return it->second;
  }
};

Meta ReadMeta(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  Meta m;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string key;
    s >> key;
    int64_t x;
    while (s >> x) m.v[key].push_back(x);
  }
  return m;
}

void ReadInto(const std::string &path, void *dst, size_t bytes) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open " + path);
  const size_t got = std::fread(dst, 1, bytes, f);
  std::fclose(f);
  if (got != bytes) throw std::runtime_error(path + ": short read");
}

size_t FileSize(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("cannot open " + path);
  return static_cast<size_t>(in.tellg());
}

inline uint16_t Bf16(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return static_cast<uint16_t>((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}
inline float FromBf16(uint16_t h) {
  const uint32_t b = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

struct Xclbin {
  xrt::hw_context ctx;
  xrt::kernel kernel;
};

Xclbin Load(xrt::device &dev, const std::string &path) {
  xrt::xclbin xb(path);
  dev.register_xclbin(xb);
  Xclbin out;
  out.ctx = xrt::hw_context(dev, xb.get_uuid());
  for (auto &k : xb.get_kernels())
    if (k.get_name().rfind("MLIR_AIE", 0) == 0) {
      out.kernel = xrt::kernel(out.ctx, k.get_name());
      return out;
    }
  throw std::runtime_error(path + ": no MLIR_AIE kernel");
}

bool Timing() {
  static const bool on = std::getenv("LFM3_XRT_TIMING") != nullptr;
  return on;
}
double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

// One prefill xclbin for M rows: its instruction streams (the LM head over rows 0
// or 512) and its scratch. bundle.txt keys: "p." for M = 1024, "q." for M = 512.
struct Variant {
  std::string key;
  int M{};
  Xclbin x;
  xrt::bo ins[2], c;
  uint32_t ins_n[2]{};
  int streams{};
};

struct flm_lfm3 {
  Meta meta;
  xrt::device dev{0};
  Xclbin dec;
  std::vector<Variant> pre;  // ascending M
  xrt::bo p_rec, p_w, d_s, d_w, d_ins;
  void *dl{};
  Dispatch gen{};
  int vocab{}, hidden{}, ctx{}, window{}, M{}, rec{}, max_length{};
  int length{};

  flm_lfm3(const std::string &dir, int max_len) {
    meta = ReadMeta(dir + "/bundle.txt");
    vocab = meta("vocab");
    hidden = meta("hidden");
    ctx = meta("ctx");
    window = meta("window");
    M = meta("p.M");
    rec = meta("p.rec");
    max_length = std::min(max_len, ctx);

    dec = Load(dev, dir + "/decode.xclbin");
    struct Spec { const char *key, *xclbin, *ins0, *ins512, *scratch; };
    std::vector<Spec> specs;
    if (meta.v.count("q.M")) specs.push_back({"q", "/prefill512.xclbin", "/prefill512_head0.insts", nullptr, "/prefill512_scratch.bin"});
    specs.push_back({"p", "/prefill.xclbin", "/prefill_head0.insts", "/prefill_head512.insts", "/prefill_scratch.bin"});
    for (const Spec &sp : specs) {
      Variant v;
      v.key = sp.key;
      v.M = meta(v.key + ".M");
      v.x = Load(dev, dir + sp.xclbin);
      const char *ins[2] = {sp.ins0, sp.ins512};
      for (int i = 0; i < 2 && ins[i]; ++i) {
        const std::string path = dir + ins[i];
        const size_t bytes = FileSize(path);
        v.ins_n[i] = bytes / 4;
        v.ins[i] = xrt::bo(dev, bytes, xrt::bo::flags::cacheable, v.x.kernel.group_id(1));
        ReadInto(path, v.ins[i].map<void *>(), bytes);
        v.ins[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        v.streams = i + 1;
      }
      v.c = xrt::bo(dev, FileSize(dir + sp.scratch), xrt::bo::flags::host_only, v.x.kernel.group_id(5));
      ReadInto(dir + sp.scratch, v.c.map<void *>(), v.c.size());
      v.c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      pre.push_back(std::move(v));
    }
    // records and weights are shared by the variants (the weights do not depend on M)
    p_rec = xrt::bo(dev, size_t(M) * rec * 2, xrt::bo::flags::host_only, pre.back().x.kernel.group_id(3));
    std::memset(p_rec.map<void *>(), 0, p_rec.size());
    const std::string pw = dir + "/prefill_weights.bin";
    p_w = xrt::bo(dev, FileSize(pw), xrt::bo::flags::host_only, pre.back().x.kernel.group_id(4));
    ReadInto(pw, p_w.map<void *>(), p_w.size());
    p_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    const std::string ds = dir + "/decode_state.bin", dw = dir + "/decode_weights.bin";
    // (decode xclbin loaded above)
    d_s = xrt::bo(dev, FileSize(ds), xrt::bo::flags::host_only, dec.kernel.group_id(3));
    ReadInto(ds, d_s.map<void *>(), d_s.size());
    d_s.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    d_w = xrt::bo(dev, FileSize(dw), xrt::bo::flags::host_only, dec.kernel.group_id(4));
    ReadInto(dw, d_w.map<void *>(), d_w.size());
    d_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    d_ins = xrt::bo(dev, 8u << 20, xrt::bo::flags::cacheable, dec.kernel.group_id(1));

    dl = dlopen((dir + "/decode_dispatch.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!dl) throw std::runtime_error(std::string("dlopen decode_dispatch.so: ") + dlerror());
    gen = reinterpret_cast<Dispatch>(dlsym(dl, "dispatch_generate"));
    if (!gen) throw std::runtime_error("decode_dispatch.so lacks dispatch_generate");
  }

  ~flm_lfm3() {
    if (dl) dlclose(dl);
  }

  // Input rows of a mixed prompt: token ids >= 0, embedding rows for ids < 0.
  const float *rows_{};
  const float *Row(int32_t id) const { return id >= 0 ? Embedding(id) : rows_ + size_t(-id - 1) * hidden; }

  const float *Embedding(int id) const {
    return reinterpret_cast<const float *>(const_cast<xrt::bo &>(d_s).map<const char *>() + meta("d.emb") +
                                           size_t(id) * meta("d.emb_row"));
  }

  // Decodes `id` at position `length` and appends it. A row id (< 0) goes through
  // the spare embedding-table row after the vocabulary, as token `vocab`.
  void Decode(int32_t id, float *logits) {
    if (id < 0) {
      float *spare = reinterpret_cast<float *>(d_s.map<char *>() + meta("d.emb") + size_t(vocab) * meta("d.emb_row"));
      std::memcpy(spare, Row(id), size_t(hidden) * 4);
      d_s.sync(XCL_BO_SYNC_BO_TO_DEVICE, size_t(hidden) * 4, meta("d.emb") + size_t(vocab) * meta("d.emb_row"));
      id = vocab;
    }
    if (length >= max_length) throw std::runtime_error("context is full");
    const int pos = length;
    const int first = std::max(0, pos - window + 1), cached_swa = pos - first;
    const int recs = std::max(2, (pos + 15) / 16), b = std::max(1, recs - 64);
    if (recs - b > 64 || b > 64) throw std::runtime_error("decode beyond 2048 positions is not supported yet");
    uint32_t *ins = nullptr;
    const int64_t n = gen(id, pos, first, cached_swa, recs, std::max(1, (cached_swa + 15) / 16), recs - b, b, &ins);
    if (n <= 0 || size_t(n) * 4 > d_ins.size()) throw std::runtime_error("dispatch_generate failed");
    std::memcpy(d_ins.map<void *>(), ins, size_t(n) * 4);
    d_ins.sync(XCL_BO_SYNC_BO_TO_DEVICE, size_t(n) * 4, 0);
    const double t0 = Now();
    auto run = dec.kernel(3u, d_ins, uint32_t(n), d_s, d_w);
    run.wait();
    if (Timing()) std::fprintf(stderr, "lfm3_xrt: decode pos %d %.2f ms\n", pos, 1e3 * (Now() - t0));
    if (logits) {
      const size_t off = meta("d.logits");
      d_s.sync(XCL_BO_SYNC_BO_FROM_DEVICE, size_t(vocab) * 4, off);
      std::memcpy(logits, d_s.map<const char *>() + off, size_t(vocab) * 4);
    }
    ++length;
  }

  // Prefills m <= M fresh tokens (positions 0 .. m - 1), writes the last token's
  // logits and hands the state to decode.
  void Prefill(const int32_t *ids, int m, float *logits) {
    const double t0 = Now();
    uint16_t *R = p_rec.map<uint16_t *>();
    for (int t = 0; t < m; ++t) {
      const float *e = Row(ids[t]);
      uint16_t *x = R + size_t(t) * rec + hidden;
      for (int i = 0; i < hidden; ++i) x[i] = Bf16(e[i]);
    }
    p_rec.sync(XCL_BO_SYNC_BO_TO_DEVICE, size_t(m) * rec * 2, 0);
    const double t1 = Now();
    size_t vi = 0;
    while (pre[vi].M < m) ++vi;
    Variant &P = pre[vi];
    const int head_rows = meta("p.head_rows"), v = m - 1 >= head_rows;
    auto run = P.x.kernel(3u, P.ins[v], P.ins_n[v], p_rec, p_w, P.c);
    run.wait();
    const double t2 = Now();
    if (logits) {
      const size_t at = meta(P.key + ".logits") + size_t(m - 1 - v * head_rows) * vocab;
      P.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, size_t(vocab) * 2, at * 2);
      const uint16_t *lg = P.c.map<const uint16_t *>() + at;
      for (int i = 0; i < vocab; ++i) logits[i] = FromBf16(lg[i]);
    }
    Handoff(P, m);
    const double t3 = Now();
    if (Timing())
      std::fprintf(stderr, "lfm3_xrt: prefill %d tokens (M %d): records %.2f ms, NPU %.2f ms, handoff %.2f ms\n", m,
                   P.M, 1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2));
    length = m;
  }

  // KV caches: decode head h = 2 c + j keeps [k bf16 128 | v bf16 128] per position at
  // kv + ((4 c + j) ctx + pos) 512 bytes; prefill pieces hold K at 9216 and V at
  // 11264 of piece (pos / 16) * 4 + g, as [kb8][dblk][8 keys][8 dims].
  // Convolution states: per column c and conv core j in {2, 3}, 96 chunks of 8 f32 at
  // (4 k + j) * 32 bytes, f32 index f = 8 k + e holding tap f / 128 (oldest first) of
  // channel ((8 c + lane / 16) * 2 + j - 2) * 16 + lane % 16, lane = f % 128.
  void Handoff(Variant &P, int m) {
    xrt::bo &p_c = P.c;
    const auto &kv = meta.list("d.kv"), &state = meta.list("d.state"), &pieces = meta.list(P.key + ".pieces"),
               &cs = meta.list(P.key + ".cs");
    const size_t piece = meta("p.piece"), conv_row = meta("p.conv_row"), state_rows = meta("p.state_rows"),
                 state_col = meta("d.state_col");
    char *S = d_s.map<char *>();
    const uint16_t *C = p_c.map<const uint16_t *>();
    const size_t blocks = (m + 15) / 16;
    for (size_t i = 0; i < pieces.size(); ++i) {
      p_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, blocks * 4 * piece * 2, pieces[i] * 2);
      for (int g = 0; g < 4; ++g) {
        uint16_t entry[256];
        for (int t = 0; t < m; ++t) {
          const uint16_t *p = C + pieces[i] + ((t / 16) * 4 + g) * piece;
          const int s = t % 16;
          for (int d = 0; d < 128; d += 8) {
            const size_t in = (((s >> 3) * 16 + (d >> 3)) << 6) + ((s & 7) << 3);
            std::memcpy(entry + d, p + 9216 + in, 16);
            std::memcpy(entry + 128 + d, p + 11264 + in, 16);
          }
          for (int h = 4 * g; h < 4 * g + 4; ++h)
            std::memcpy(S + kv[i] + ((size_t(4 * (h / 2) + h % 2) * ctx) + t) * 512, entry, 512);
        }
      }
      for (int h = 0; h < 16; ++h)
        d_s.sync(XCL_BO_SYNC_BO_TO_DEVICE, size_t(m) * 512, kv[i] + size_t(4 * (h / 2) + h % 2) * ctx * 512);
    }
    for (size_t i = 0; i < cs.size(); ++i) {
      const size_t row0 = cs[i] + (size_t(m) - 6 + state_rows) * conv_row;  // scratch row of token m - 6
      p_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE, 6 * conv_row * 2, row0 * 2);
      float *st = reinterpret_cast<float *>(S + state[i]);
      for (int c = 0; c < 8; ++c)
        for (int j = 2; j < 4; ++j)
          for (int f = 0; f < 768; ++f) {
            const int tap = f / 128, lane = f % 128;
            const int ch = ((c * 8 + lane / 16) * 2 + (j - 2)) * 16 + lane % 16;
            st[(c * state_col + ((f / 8) * 4 + j) * 32) / 4 + f % 8] =
                FromBf16(C[row0 + tap * conv_row + (ch / 64) * 128 + ch % 64]);
          }
      for (int c = 0; c < 8; ++c) d_s.sync(XCL_BO_SYNC_BO_TO_DEVICE, 96 * 4 * 32, state[i] + c * state_col);
    }
  }

  void Clear() {
    // Decode reads only cached positions, and conv states are rewritten by the next
    // prefill; a fresh decode-only start needs zero conv states.
    // Only the state chunks (the first 96 x 4 x 32 bytes of each column): the taps follow.
    char *S = d_s.map<char *>();
    const size_t state_col = meta("d.state_col"), state_bytes = 96 * 4 * 32;
    for (int64_t off : meta.list("d.state"))
      for (int c = 0; c < 8; ++c) {
        std::memset(S + off + c * state_col, 0, state_bytes);
        d_s.sync(XCL_BO_SYNC_BO_TO_DEVICE, state_bytes, off + c * state_col);
      }
    length = 0;
  }
};

extern "C" {

flm_lfm3 *flm_lfm3_create(const char *model_dir, const char * /*asset_root*/, int max_length) {
  try {
    auto *m = new flm_lfm3(model_dir, max_length);
    m->Clear();
    return m;
  } catch (const std::exception &e) {
    g_error = e.what();
    return nullptr;
  }
}

void flm_lfm3_destroy(flm_lfm3 *m) { delete m; }

const char *flm_lfm3_last_error(void) { return g_error.c_str(); }

int flm_lfm3_vocab(const flm_lfm3 *m) { return m->vocab; }

int flm_lfm3_prefill(flm_lfm3 *m, const int32_t *ids, int count, float *logits) {
  try {
    if (count <= 0) return 0;
    int done = 0;
    if (m->length == 0 && count >= 2) {
      const int pre = std::min(count, m->M);
      m->Prefill(ids, pre, pre == count ? logits : nullptr);
      done = pre;
    }
    for (int i = done; i < count; ++i) m->Decode(ids[i], i + 1 == count ? logits : nullptr);
    return 0;
  } catch (const std::exception &e) {
    g_error = e.what();
    return 1;
  }
}

int flm_lfm3_prefill_mixed(flm_lfm3 *m, const int32_t *ids, const float *rows, int count, float *logits) {
  m->rows_ = rows;
  const int r = flm_lfm3_prefill(m, ids, count, logits);
  m->rows_ = nullptr;
  return r;
}

int flm_lfm3_decode_row(flm_lfm3 *m, const float *row, float *logits) {
  m->rows_ = row;
  const int r = flm_lfm3_decode(m, -1, logits);
  m->rows_ = nullptr;
  return r;
}

int flm_lfm3_decode(flm_lfm3 *m, int32_t id, float *logits) {
  try {
    m->Decode(id, logits);
    return 0;
  } catch (const std::exception &e) {
    g_error = e.what();
    return 1;
  }
}

int flm_lfm3_length(flm_lfm3 *m) { return m->length; }

void flm_lfm3_clear(flm_lfm3 *m) {
  try {
    m->Clear();
  } catch (const std::exception &e) {
    g_error = e.what();
  }
}

}  // extern "C"
