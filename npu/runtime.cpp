#include "npu/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gufo/ext/experimental/npu.hpp>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gufo::models::lfm3::npu {
namespace {

namespace hw = gufo::npu;

// bundle.txt: `key value...` lines (npu/iron/bundle.py).
struct Meta {
  std::map<std::string, std::vector<std::int64_t>> v;
  std::int64_t operator()(const std::string& k, std::size_t i = 0) const {
    auto it = v.find(k);
    if (it == v.end() || it->second.size() <= i)
      throw std::runtime_error("bundle.txt lacks " + k);
    return it->second[i];
  }
  const std::vector<std::int64_t>& list(const std::string& k) const {
    auto it = v.find(k);
    if (it == v.end())
      throw std::runtime_error("bundle.txt lacks " + k);
    return it->second;
  }
  bool has(const std::string& k) const { return v.count(k) != 0; }
};

Meta ReadMeta(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open " + path.string());
  Meta m;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string key;
    s >> key;
    std::int64_t x;
    while (s >> x)
      m.v[key].push_back(x);
  }
  return m;
}

inline std::uint16_t Bf16(float f) {
  std::uint32_t b;
  std::memcpy(&b, &f, 4);
  return static_cast<std::uint16_t>((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}
inline float FromBf16(std::uint16_t h) {
  const std::uint32_t b = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

bool Timing() {
  static const bool on = std::getenv("GUFO_LFM3_NPU_TIMING") != nullptr;
  return on;
}
double Now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Kernel arguments: 3 records / decode state, 4 weights, 5 prefill scratch.
constexpr unsigned kState = 3, kWeights = 4, kScratch = 5;

// One prefill program for M rows: its instruction streams (the LM head over
// rows 0 or 512) and its scratch. bundle.txt keys: "p." M = 1024, "q." M =
// 512, "r." M = 128.
struct Variant {
  std::string key;
  int M{};
  hw::Program program;
  hw::Instructions ins[2];
  hw::Buffer scratch;
  int streams{};
};

}  // namespace

struct Runtime::Impl {
  Meta meta;
  hw::Device dev = hw::Device::Open();
  hw::Program dec;
  std::vector<Variant> pre;  // ascending M
  hw::Buffer p_rec, p_w, d_s, d_w;
  hw::Instructions d_ins;
  hw::DispatchLibrary gen;
  int vocab{}, hidden{}, ctx{}, window{}, M{}, rec{}, max_length{};
  int length{};
  const float* rows{};  // embedding rows of the current call

  Impl(const std::filesystem::path& dir, int max_len) {
    meta = ReadMeta(dir / "bundle.txt");
    vocab = int(meta("vocab"));
    hidden = int(meta("hidden"));
    ctx = int(meta("ctx"));
    window = int(meta("window"));
    M = int(meta("p.M"));
    rec = int(meta("p.rec"));
    max_length = std::min(max_len, ctx);

    dec = hw::Program::Load(dev, dir / "decode.xclbin");
    struct Spec {
      const char *key, *xclbin, *ins0, *ins512, *scratch;
    };
    std::vector<Spec> specs;
    if (meta.has("r.M"))
      specs.push_back({"r", "prefill128.xclbin", "prefill128_head0.insts",
                       nullptr, "prefill128_scratch.bin"});
    if (meta.has("q.M"))
      specs.push_back({"q", "prefill512.xclbin", "prefill512_head0.insts",
                       nullptr, "prefill512_scratch.bin"});
    specs.push_back({"p", "prefill.xclbin", "prefill_head0.insts",
                     "prefill_head512.insts", "prefill_scratch.bin"});
    for (const Spec& sp : specs) {
      Variant v;
      v.key = sp.key;
      v.M = int(meta(v.key + ".M"));
      v.program = hw::Program::Load(dev, dir / sp.xclbin);
      const char* ins[2] = {sp.ins0, sp.ins512};
      for (int i = 0; i < 2 && ins[i]; ++i) {
        v.ins[i] = v.program.LoadInstructions(dir / ins[i]);
        v.streams = i + 1;
      }
      v.scratch = v.program.LoadBuffer(dir / sp.scratch, kScratch);
      pre.push_back(std::move(v));
    }
    // records and weights are shared by the variants (the weights do not depend
    // on M)
    p_rec = pre.back().program.Allocate(std::size_t(M) * rec * 2, kState);
    std::memset(p_rec.data(), 0, p_rec.size());
    p_rec.ToDevice();
    p_w = pre.back().program.LoadBuffer(dir / "prefill_weights.bin", kWeights);
    d_s = dec.LoadBuffer(dir / "decode_state.bin", kState);
    d_w = dec.LoadBuffer(dir / "decode_weights.bin", kWeights);
    d_ins = dec.AllocateInstructions(2u << 20);
    gen = hw::DispatchLibrary::Open(dir / "decode_dispatch.so", 8);
    FindTokenPatch();
  }

  ~Impl() {
    try {
      Finish(nullptr);
    } catch (...) {
    }
  }

  const float* Embedding(int id) const {
    return reinterpret_cast<const float*>(d_s.data() + meta("d.emb") +
                                          std::size_t(id) * meta("d.emb_row"));
  }
  const float* Row(std::int32_t id) const {
    return id >= 0 ? Embedding(id) : rows + std::size_t(-id - 1) * hidden;
  }

  // Decode streams depend on the token id only through a few address words:
  // the next position's stream is generated (with id 0) while the NPU runs the
  // current token, then patched with the id.
  std::vector<std::uint32_t> next_ins;
  int next_pos = -1;
  std::vector<hw::Instructions::Patch> tok_patch;
  bool patch_ok = false;

  std::span<const std::uint32_t> Generate(int id, int pos) const {
    const int first = std::max(0, pos - window + 1), cached_swa = pos - first;
    const int recs = std::max(2, (pos + 15) / 16), b = std::max(1, recs - 64);
    if (recs - b > 64 || b > 64)
      throw std::runtime_error(
          "NPU decode beyond 2048 positions is not supported yet");
    const std::int32_t s[8] = {id,       pos,
                               first,    cached_swa,
                               recs,     std::max(1, (cached_swa + 15) / 16),
                               recs - b, b};
    return gen.Generate(s);
  }
  void FindTokenPatch() {
    const auto g0 = Generate(0, 40);
    const std::vector<std::uint32_t> a(g0.begin(), g0.end());
    const auto g1 = Generate(1, 40);
    if (g1.size() != a.size())
      return;
    tok_patch = hw::DiffInstructions(a, g1);
    const auto big = Generate(vocab, 40);  // the largest id (the spare row)
    if (big.size() != a.size())
      return;
    std::vector<std::uint32_t> v = a;
    for (const auto& p : tok_patch)
      v[p.index] += std::uint32_t(vocab) * p.delta;
    patch_ok = std::equal(v.begin(), v.end(), big.begin());
  }
  void Prepare(int pos) {
    if (!patch_ok || pos >= max_length)
      return;
    const auto g = Generate(0, pos);
    next_ins.assign(g.begin(), g.end());
    next_pos = pos;
  }

  hw::Run pending;
  double pending_t0 = 0, pending_g = 0;
  bool last_prefill = false;

  // Decodes `id` at position `length` and appends it. A row id (< 0) goes
  // through the spare embedding-table row after the vocabulary, as token
  // `vocab`.
  void Decode(std::int32_t id, float* logits, bool wait = true) {
    Finish(nullptr);
    if (id < 0) {
      const std::size_t at =
          meta("d.emb") + std::size_t(vocab) * meta("d.emb_row");
      std::memcpy(d_s.data() + at, Row(id), std::size_t(hidden) * 4);
      d_s.ToDevice(at, std::size_t(hidden) * 4);
      id = vocab;
    }
    if (length >= max_length)
      throw std::runtime_error("NPU context is full");
    const int pos = length;
    const double g0 = Now();
    if (next_pos == pos) {  // prepared during the previous token
      d_ins.Assign(next_ins);
      d_ins.Apply(tok_patch, std::uint32_t(id));
    } else {
      d_ins.Assign(Generate(id, pos));
    }
    const double t0 = Now();
    pending = dec.Start(d_ins, {&d_s, &d_w});
    pending_t0 = t0;
    pending_g = t0 - g0;
    Prepare(pos + 1);  // on the host while the NPU runs
    ++length;
    last_prefill = false;
    if (wait)
      Finish(logits);
  }

  void Finish(float* logits) {
    if (!pending)
      return;
    pending.Wait();
    pending = hw::Run();
    const double t1 = Now();
    if (logits) {
      const std::size_t off = meta("d.logits");
      d_s.FromDevice(off, std::size_t(vocab) * 4);
      std::memcpy(logits, d_s.data() + off, std::size_t(vocab) * 4);
    }
    if (Timing())
      std::fprintf(stderr,
                   "lfm3 npu: decode pos %d %.2f ms (instructions %.2f ms, "
                   "logits %.2f ms)\n",
                   length - 1, 1e3 * (t1 - pending_t0), 1e3 * pending_g,
                   1e3 * (Now() - t1));
  }

  // Final-normed hidden state of the last row: after a decode from the head
  // phase's input block, after a prefill from the last record's xn.
  void Hidden(float* out) {
    Finish(nullptr);
    if (!length)
      throw std::runtime_error("no rows yet");
    const std::size_t blk = meta("d.final_block");
    const std::byte* S = d_s.data();
    const auto* wout = reinterpret_cast<const std::uint16_t*>(
        S + blk + 10 * std::size_t(hidden));
    if (last_prefill) {
      p_rec.FromDevice(std::size_t(length - 1) * rec * 2,
                       std::size_t(hidden) * 2);
      const auto* xn = reinterpret_cast<const std::uint16_t*>(p_rec.data()) +
                       std::size_t(length - 1) * rec;
      for (int i = 0; i < hidden; ++i)
        out[i] = FromBf16(xn[i]) * FromBf16(wout[i]);
      return;
    }
    d_s.FromDevice(blk, 8 * std::size_t(hidden));
    const auto* h = reinterpret_cast<const float*>(S + blk);
    const float* o = h + hidden;
    const auto* wpost = reinterpret_cast<const std::uint16_t*>(
        S + blk + 8 * std::size_t(hidden));
    double so = 0;
    for (int i = 0; i < hidden; ++i)
      so += double(o[i]) * o[i];
    const float io = float(1.0 / std::sqrt(so / hidden + 1e-5));
    double sx = 0;
    for (int i = 0; i < hidden; ++i) {
      out[i] = h[i] + o[i] * io * FromBf16(wpost[i]);
      sx += double(out[i]) * out[i];
    }
    const float ix = float(1.0 / std::sqrt(sx / hidden + 1e-5));
    for (int i = 0; i < hidden; ++i)
      out[i] *= ix * FromBf16(wout[i]);
  }

  // Prefills m <= M fresh tokens (positions 0 .. m - 1), writes the last
  // token's logits and hands the state to decode.
  void PrefillFresh(const std::int32_t* ids, int m, float* logits) {
    Finish(nullptr);
    const double t0 = Now();
    auto* R = reinterpret_cast<std::uint16_t*>(p_rec.data());
    for (int t = 0; t < m; ++t) {
      const float* e = Row(ids[t]);
      std::uint16_t* x = R + std::size_t(t) * rec + hidden;
      for (int i = 0; i < hidden; ++i)
        x[i] = Bf16(e[i]);
    }
    p_rec.ToDevice(0, std::size_t(m) * rec * 2);
    const double t1 = Now();
    std::size_t vi = 0;
    while (pre[vi].M < m)
      ++vi;
    Variant& P = pre[vi];
    const int head_rows =
        int(meta.has(P.key + ".head_rows") ? meta(P.key + ".head_rows")
                                           : meta("p.head_rows"));
    const int v = m - 1 >= head_rows;
    const std::size_t head_n = meta.has(P.key + ".head_n")
                                   ? std::size_t(meta(P.key + ".head_n"))
                                   : std::size_t(vocab);
    P.program.Start(P.ins[v], {&p_rec, &p_w, &P.scratch}).Wait();
    const double t2 = Now();
    if (logits) {
      const std::size_t at =
          meta(P.key + ".logits") + std::size_t(m - 1 - v * head_rows) * head_n;
      P.scratch.FromDevice(at * 2, std::size_t(vocab) * 2);
      const auto* lg =
          reinterpret_cast<const std::uint16_t*>(P.scratch.data()) + at;
      for (int i = 0; i < vocab; ++i)
        logits[i] = FromBf16(lg[i]);
    }
    Handoff(P, m);
    const double t3 = Now();
    last_prefill = true;
    if (Timing())
      std::fprintf(stderr,
                   "lfm3 npu: prefill %d tokens (M %d): records %.2f ms, NPU "
                   "%.2f ms, handoff %.2f ms\n",
                   m, P.M, 1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2));
    length = m;
  }

  // KV caches: decode head h = 2 c + j keeps [k bf16 128 | v bf16 128] per
  // position at kv + ((4 c + j) ctx + pos) 512 bytes; prefill pieces hold K at
  // 9216 and V at 11264 of piece (pos / 16) * 4 + g, as [kb8][dblk][8 keys][8
  // dims]. Convolution states: per column c and conv core j in {2, 3}, 96
  // chunks of 8 f32 at (4 k + j) * 32 bytes, f32 index f = 8 k + e holding tap
  // f / 128 (oldest first) of channel ((8 c + lane / 16) * 2 + j - 2) * 16 +
  // lane % 16.
  void Handoff(Variant& P, int m) {
    hw::Buffer& p_c = P.scratch;
    const auto &kv = meta.list("d.kv"), &state = meta.list("d.state"),
               &pieces = meta.list(P.key + ".pieces"),
               &cs = meta.list(P.key + ".cs");
    const std::size_t piece = meta("p.piece"), conv_row = meta("p.conv_row"),
                      state_rows = meta("p.state_rows"),
                      state_col = meta("d.state_col");
    std::byte* S = d_s.data();
    const auto* C = reinterpret_cast<const std::uint16_t*>(p_c.data());
    const std::size_t blocks = (m + 15) / 16;
    for (std::size_t i = 0; i < pieces.size(); ++i) {
      p_c.FromDevice(pieces[i] * 2, blocks * 4 * piece * 2);
      for (int g = 0; g < 4; ++g) {
        std::uint16_t entry[256];
        for (int t = 0; t < m; ++t) {
          const std::uint16_t* p = C + pieces[i] + ((t / 16) * 4 + g) * piece;
          const int s = t % 16;
          for (int d = 0; d < 128; d += 8) {
            const std::size_t in =
                (((s >> 3) * 16 + (d >> 3)) << 6) + ((s & 7) << 3);
            std::memcpy(entry + d, p + 9216 + in, 16);
            std::memcpy(entry + 128 + d, p + 11264 + in, 16);
          }
          for (int h = 4 * g; h < 4 * g + 4; ++h)
            std::memcpy(
                S + kv[i] +
                    ((std::size_t(4 * (h / 2) + h % 2) * ctx) + t) * 512,
                entry, 512);
        }
      }
      for (int h = 0; h < 16; ++h)
        d_s.ToDevice(kv[i] + std::size_t(4 * (h / 2) + h % 2) * ctx * 512,
                     std::size_t(m) * 512);
    }
    for (std::size_t i = 0; i < cs.size(); ++i) {
      const std::size_t row0 =
          cs[i] + (std::size_t(m) - 6 + state_rows) *
                      conv_row;  // scratch row of token m - 6
      p_c.FromDevice(row0 * 2, 6 * conv_row * 2);
      auto* st = reinterpret_cast<float*>(S + state[i]);
      for (int c = 0; c < 8; ++c)
        for (int j = 2; j < 4; ++j)
          for (int f = 0; f < 768; ++f) {
            const int tap = f / 128, lane = f % 128;
            const int ch = ((c * 8 + lane / 16) * 2 + (j - 2)) * 16 + lane % 16;
            st[(c * state_col + ((f / 8) * 4 + j) * 32) / 4 + f % 8] =
                FromBf16(C[row0 + tap * conv_row + (ch / 64) * 128 + ch % 64]);
          }
      for (int c = 0; c < 8; ++c)
        d_s.ToDevice(state[i] + c * state_col, 96 * 4 * 32);
    }
  }

  void Clear() {
    Finish(nullptr);
    // Decode reads only cached positions and the next prefill rewrites the
    // convolution states; a decode-only start needs them zero (the state
    // chunks, the first 96 x 4 x 32 bytes of each column; the taps follow).
    const std::size_t state_col = meta("d.state_col"),
                      state_bytes = 96 * 4 * 32;
    for (std::int64_t off : meta.list("d.state"))
      for (int c = 0; c < 8; ++c) {
        std::memset(d_s.data() + off + c * state_col, 0, state_bytes);
        d_s.ToDevice(off + c * state_col, state_bytes);
      }
    length = 0;
  }
};

Runtime::Runtime(const std::filesystem::path& bundle, int max_length)
    : impl_(std::make_unique<Impl>(bundle, max_length)) {
  impl_->Clear();
}
Runtime::~Runtime() = default;
int Runtime::vocab() const {
  return impl_->vocab;
}
int Runtime::hidden() const {
  return impl_->hidden;
}
int Runtime::length() const {
  return impl_->length;
}

void Runtime::Prefill(std::span<const std::int32_t> ids, const float* rows,
                      float* logits) {
  auto& m = *impl_;
  if (ids.empty())
    return;
  m.rows = rows;
  struct Reset {
    const float*& r;
    ~Reset() { r = nullptr; }
  } reset{m.rows};
  const int count = int(ids.size());
  int done = 0;
  if (m.length == 0 && count >= 2) {
    const int n = std::min(count, m.M);
    m.PrefillFresh(ids.data(), n, n == count ? logits : nullptr);
    done = n;
  }
  for (int i = done; i < count; ++i)
    m.Decode(ids[i], i + 1 == count ? logits : nullptr);
}

void Runtime::DecodeStart(std::int32_t id, const float* row) {
  auto& m = *impl_;
  m.rows = row;
  struct Reset {
    const float*& r;
    ~Reset() { r = nullptr; }
  } reset{m.rows};
  m.Decode(id, nullptr, false);
}

void Runtime::Finish(float* logits) {
  impl_->Finish(logits);
}
void Runtime::Hidden(float* out) {
  impl_->Hidden(out);
}
void Runtime::Clear() {
  impl_->Clear();
}

}  // namespace gufo::models::lfm3::npu
