// gufo_xrt_engine.h on engine.xclbin (xrt/iron/engine). The host writes the
// instruction streams: per phase, every stream's tasks (a header record, then
// the phase's records) and every column's drains, then a sync per drain.
//
// Shim DMA: stream s (cores (c, 2h), (c, 2h + 1), s = 2 c + h) is shim
// kStream[s] (column, MM2S channel), as engine.py's reference sequence placed
// them; column c drains through its own shim's S2MM channel 0. The 16 BDs of a
// shim tile are shared by its channels and reused phase by phase (a phase ends
// when its drains are complete, after the cores consumed all of its records).

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <array>
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

#include "../gufo_xrt_engine.h"

namespace {

thread_local std::string g_error;

constexpr int kRec = 1096, kOut = 128, kStreams = 16, kCores = 32, kCols = 8;
constexpr int kMaxA = 8, kMaxTiles = 32;  // per core: A half-records (32 KiB),
                                          // 8 x 8 accumulator tiles
constexpr uint32_t kHeader0 = 0x6040100, kHeader1 = 0x108;
struct Shim {
  int col, ch;
};
constexpr Shim kStream[kStreams] = {
    {0, 0}, {1, 0}, {2, 0}, {1, 1}, {3, 0}, {2, 1}, {4, 0}, {3, 1},
    {5, 0}, {4, 1}, {6, 0}, {5, 1}, {7, 0}, {6, 1}, {7, 1}, {0, 1}};
enum Arg { kS = 0, kW = 1 };

// One DMA transfer: 4 dims outermost first (the outermost repeats the BD).
struct Dims {
  std::array<uint32_t, 4> size{1, 1, 1, 1}, stride{0, 0, 0, 1};
};
Dims Linear(uint64_t words) {
  Dims d;
  d.size = {1, 1, 1, uint32_t(words)};
  return d;
}

struct Txn {
  std::vector<uint32_t> w = std::vector<uint32_t>(4, 0);
  uint32_t ops = 0;
  void Write32(uint32_t addr, uint32_t v) {
    w.insert(w.end(), {0u, 0u, addr, 0u, v, 24u});
    ++ops;
  }
  void MaskWrite(uint32_t addr, uint32_t v, uint32_t mask) {
    w.insert(w.end(), {3u, 0u, addr, 0u, v, mask, 28u});
    ++ops;
  }
  void Block(uint32_t addr, const uint32_t (&bd)[8]) {
    w.insert(w.end(), {1u, 0u, addr, 48u});
    w.insert(w.end(), bd, bd + 8);
    ++ops;
  }
  void Patch(uint32_t addr, uint32_t arg, uint64_t bytes) {
    w.insert(w.end(), {0x81u, 48u, 0u, 0u, 0u, 0u, addr, 0u, arg, 0u,
                       uint32_t(bytes), uint32_t(bytes >> 32)});
    ++ops;
  }
  void Sync(uint32_t col, uint32_t chan) {
    w.insert(w.end(), {0x80u, 16u, (col & 0xff) << 16,
                       (1u << 8) | (1u << 16) | ((chan & 0xff) << 24)});
    ++ops;
  }
  std::vector<uint32_t> Finish() {
    w[0] = kHeader0, w[1] = kHeader1, w[2] = ops, w[3] = uint32_t(w.size() * 4);
    return w;
  }
};

// BD words of a transfer (the layout mlir-aie's generated builders write).
void Encode(const Dims& d, uint32_t (&bd)[8], uint32_t& repeat) {
  const auto [s3, s2, s1, s0] = d.size;
  const auto [t3, t2, t1, t0] = d.stride;
  for (auto& x : bd)
    x = 0;
  const bool linear = s1 == 1 && s2 == 1 && t0 == 1;
  if (!linear && (s0 > 1023 || s1 > 1023))
    throw std::runtime_error("engine: DMA d0/d1 size above 1023");
  if (s3 > 64)
    throw std::runtime_error("engine: DMA iteration count above 64");
  bd[0] = s0 * s1 * s2;
  if (!linear) {
    bd[3] = ((s0 & 1023) << 20) | ((t0 == 0 ? 0 : t0 - 1) & 0xFFFFF);
    bd[4] = ((s1 & 1023) << 20) | ((s1 > 1 ? t1 - 1 : 0) & 0xFFFFF);
    bd[5] = (s2 > 1 ? t2 - 1 : 0) & 0xFFFFF;
  }
  bd[4] |= 0xC0000000u;
  bd[5] |= 0x2000000u;
  if (s3 > 1 && t3 > 0)
    bd[6] = (((s3 - 1) & 63) << 20) | ((t3 - 1) & 0xFFFFF);
  bd[7] = 0x2000000u;
  repeat = s3 - 1;
}

// One phase's DMA tasks, BDs allocated per shim tile.
struct PhaseTxn {
  Txn& t;
  int next_bd[kCols]{};
  std::vector<int> drains;  // columns, in order
  explicit PhaseTxn(Txn& txn) : t(txn) {}
  int Bd(int col) {
    if (next_bd[col] >= 16)
      throw std::runtime_error(
          "engine: phase needs more than 16 BDs on a shim tile");
    return next_bd[col]++;
  }
  void Fill(int stream, Arg arg, int64_t offset, const Dims& d) {
    const auto [col, ch] = kStream[stream];
    const int bd = Bd(col);
    uint32_t w[8], repeat;
    Encode(d, w, repeat);
    const uint32_t base = (uint32_t(col) << 25) | (0x1D000 + 0x20 * bd);
    t.Block(base, w);
    t.Patch(base + 4, arg, uint64_t(offset) * 4);
    t.Write32((uint32_t(col) << 25) | (0x1D214 + 8 * ch),
              uint32_t(bd) | ((repeat & 255) << 16));
  }
  void Drain(int col, int64_t offset, const Dims& d) {
    const int bd = Bd(col);
    uint32_t w[8], repeat;
    Encode(d, w, repeat);
    const uint32_t base = (uint32_t(col) << 25) | (0x1D000 + 0x20 * bd);
    t.Block(base, w);
    t.Patch(base + 4, kS, uint64_t(offset) * 4);
    t.MaskWrite((uint32_t(col) << 25) | 0x1D200, 0xF00, 0x1F00);
    t.Write32((uint32_t(col) << 25) | 0x1D204,
              0x80000000u | uint32_t(bd) | ((repeat & 255) << 16));
    drains.push_back(col);
  }
  void End() {
    for (int col : drains)
      t.Sync(col, 0);
  }
};

struct Matrix {
  int n, k, kc, nt;  // chunks of 512 codes, tiles of 8 outputs per core
  int64_t w;  // word offset in the arena: [stream][chunk][tile][parity] records
};

// How a product over `rows` rows runs: chunks of 8 g rows; per chunk, segments
// of `seg` chunks (A held by the cores) and passes of `pass` tiles
// (accumulators).
struct Plan {
  int rows, chunk_rows, g, seg, segs, pass, passes;
  int64_t a_chunk, out_chunk;  // words per row chunk
  int Chunks() const { return (rows + chunk_rows - 1) / chunk_rows; }
};

Plan MakePlan(const Matrix& m, int rows) {
  for (int g = std::min(4, (rows + 7) / 8); g >= 1; --g) {
    int seg = std::min(m.kc, kMaxA / (2 * g));
    while (m.kc % seg && seg > 1 &&
           (m.kc + seg - 1) / seg == (m.kc + seg - 2) / (seg - 1))
      --seg;
    const int segs = (m.kc + seg - 1) / seg;
    const int pass = std::min(m.nt, kMaxTiles / g),
              passes = (m.nt + pass - 1) / pass;
    if (2 * (1 + 2 * segs * passes) + passes > 16)
      continue;  // BDs per shim tile
    Plan p{rows, 8 * g, g, seg, segs, pass, passes, 0, 0};
    p.a_chunk = int64_t(segs) * g * seg * 2 * kRec;
    p.out_chunk = int64_t(kCols) * m.nt * g * 4 * kOut;
    return p;
  }
  throw std::runtime_error("engine: no product plan fits the shim BDs");
}

struct Phase {
  std::vector<int32_t> header;  // its record
  int64_t
      header_w{};  // word offset of [header | padding | constants] in the arena
  // products
  int id{-1};
  Plan plan{};
  int chunk{};
  int64_t a{}, out{};
  // row operations
  int flags{}, rows{}, k{}, round{}, recs{}, pad{}, lc{}, lx{}, lo{};
  int64_t x_in{}, x_out{}, o_out{}, dst{};
  int o_id{-1}, a_id{-1}, stride{};
};

}  // namespace

struct gufo_xrt_engine {
  xrt::device dev{0};
  xrt::hw_context ctx;
  xrt::kernel kernel;
  xrt::bo s, w;
  size_t s_words{}, w_words{}, s_used{}, w_used{};
  std::vector<Matrix> matrices;
  std::vector<Phase> building;
  struct Program {
    xrt::bo ins;
    uint32_t words;
  };
  std::vector<Program> programs;
  std::mutex mutex;

  explicit gufo_xrt_engine(const std::string& dir) {
    xrt::xclbin xb(dir + "/engine.xclbin");
    dev.register_xclbin(xb);
    ctx = xrt::hw_context(dev, xb.get_uuid());
    for (auto& k : xb.get_kernels())
      if (k.get_name().rfind("MLIR_AIE", 0) == 0)
        kernel = xrt::kernel(ctx, k.get_name());
    const char* mb = std::getenv("GUFO_XRT_ENGINE_ARENA_MB");
    w_words = (size_t(mb ? std::atoi(mb) : 1024) << 20) / 4;
    s_words = (size_t(64) << 20) / 4;
    s = xrt::bo(dev, s_words * 4, xrt::bo::flags::host_only,
                kernel.group_id(3));
    w = xrt::bo(dev, w_words * 4, xrt::bo::flags::host_only,
                kernel.group_id(4));
    std::memset(s.map(), 0, s_words * 4);
    s.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  int64_t ArenaAlloc(size_t words) {
    words = (words + 1023) / 1024 * 1024;
    if (w_used + words > w_words)
      throw std::runtime_error(
          "engine: weight arena full (GUFO_XRT_ENGINE_ARENA_MB)");
    const int64_t at = int64_t(w_used);
    w_used += words;
    return at;
  }
  uint32_t* Arena() { return static_cast<uint32_t*>(w.map()); }
  void SyncArena(int64_t at, size_t words) {
    w.sync(XCL_BO_SYNC_BO_TO_DEVICE, words * 4, size_t(at) * 4);
  }

  int AddMatrix(int n, int k, const int8_t* codes, const float* scales) {
    if (n <= 0 || k <= 0 || k % 32)
      throw std::invalid_argument(
          "engine: k must be a positive multiple of 32");
    Matrix m{n, k, (k + 511) / 512, (n + kCores * 8 - 1) / (kCores * 8), 0};
    const size_t words = size_t(kStreams) * m.kc * m.nt * 2 * kRec;
    m.w = ArenaAlloc(words);
    auto* base = reinterpret_cast<uint8_t*>(Arena() + m.w);
    std::memset(base, 0, words * 4);
    const int blocks = k / 32;
    for (int j = 0; j < n; ++j) {
      const float* sc = scales + size_t(j) * blocks;
      float top = 0;
      for (int b = 0; b < blocks; ++b)
        top = std::max(top, sc[b]);
      if (!(top > 0))
        top = 1;
      const int core = j / (8 * m.nt), t = (j / 8) % m.nt, nn = j % 8;
      const int stream = core / 2, parity = core % 2;
      const int8_t* row = codes + size_t(j) * k;
      for (int kc = 0; kc < m.kc; ++kc) {
        uint8_t* rec = base + ((size_t(stream) * m.kc + kc) * m.nt * 2 +
                               size_t(t) * 2 + parity) *
                                  kRec * 4;
        auto* u = reinterpret_cast<int16_t*>(rec + 4096);
        auto* tops = reinterpret_cast<float*>(rec + 4096 + 256);
        tops[nn] = top / 32767.0f;
        for (int blk = 0; blk < 16; ++blk) {
          const int b = kc * 16 + blk;
          if (b >= blocks)
            break;
          u[blk * 8 + nn] = int16_t(std::lround(sc[b] / top * 32767.0f));
          for (int kk = 0; kk < 32; ++kk)
            rec[blk * 256 + kk * 8 + nn] = uint8_t(row[b * 32 + kk]);
        }
      }
    }
    SyncArena(m.w, words);
    matrices.push_back(m);
    return int(matrices.size()) - 1;
  }

  int64_t Alloc(int64_t words) {
    words = (words + 1023) / 1024 * 1024;
    if (s_used + size_t(words) > s_words)
      throw std::runtime_error("engine: scratch full");
    const int64_t at = int64_t(s_used);
    s_used += size_t(words);
    return at;
  }

  // ---- program building ----

  int64_t HeaderBlock(const std::vector<int32_t>& header, int pad,
                      const float* consts, int lc) {
    const int64_t at = ArenaAlloc(size_t(kRec) + pad + lc);
    uint32_t* p = Arena() + at;
    std::memset(p, 0, (size_t(kRec) + pad) * 4);
    std::memcpy(p, header.data(), header.size() * 4);
    if (lc)
      std::memcpy(p + kRec + pad, consts, size_t(lc) * 4);
    SyncArena(at, size_t(kRec) + pad + lc);
    return at;
  }

  int Gemm(int id, int rows, int64_t a, int64_t out) {
    const Matrix& m = matrices.at(size_t(id));
    const Plan p = MakePlan(m, rows);
    for (int c = 0; c < p.Chunks(); ++c) {
      Phase ph;
      ph.id = id, ph.plan = p, ph.chunk = c, ph.a = a, ph.out = out;
      ph.header = {0, p.g, p.seg, p.segs, p.passes, p.pass, m.kc, m.nt};
      ph.header_w = HeaderBlock(ph.header, 0, nullptr, 0);
      building.push_back(ph);
    }
    return int(building.size()) - 1;
  }

  int Row(int flags, int rows, int k, int64_t x_in, int64_t x_out, int o_id,
          int64_t o_out, const float* consts, int n_consts, float eps, int a_id,
          int64_t a, int stride) {
    if (k % 256)
      throw std::invalid_argument(
          "engine: row width must be a multiple of 256");
    Phase ph;
    ph.flags = flags, ph.rows = rows, ph.k = k, ph.x_in = x_in,
    ph.x_out = x_out, ph.o_id = o_id, ph.o_out = o_out;
    ph.a_id = a_id, ph.dst = a, ph.stride = stride;
    ph.lc = n_consts;
    ph.lx = (flags & GUFO_XRT_ROW_X) ? k + kOut : kOut;
    ph.lo = (flags & GUFO_XRT_ROW_O) ? 256 * matrices.at(size_t(o_id)).nt : 0;
    const int total = ph.lc + ph.lx + ph.lo;
    ph.recs = (total + kRec - 1) / kRec;
    ph.pad = ph.recs * kRec - total;
    if (ph.pad + ph.lc == 0)
      ph.recs += 1, ph.pad = kRec;
    if (ph.recs * kRec > 8 * 1024)
      throw std::invalid_argument("engine: row packet above the cores' 32 KiB");
    const int nx = (flags & GUFO_XRT_ROW_STORE_X) ? k / kOut + 1 : 1;
    const int outs = nx + ((flags & GUFO_XRT_ROW_F32) ? k / kOut : k / 256);
    union {
      float f;
      int32_t i;
    } e{eps};
    ph.header = {1,  flags,  ph.recs,        outs,
                 k,  ph.pad, ph.pad + ph.lc, ph.pad + ph.lc + ph.lx,
                 e.i};
    for (int round = 0; round * 32 < rows; ++round) {
      Phase r = ph;
      r.round = round;
      r.header_w = HeaderBlock(ph.header, ph.pad, consts, ph.lc);
      building.push_back(r);
    }
    return int(building.size()) - 1;
  }

  void EmitGemm(Txn& t, const Phase& ph) {
    const Matrix& m = matrices[size_t(ph.id)];
    const Plan& p = ph.plan;
    PhaseTxn x(t);
    const int64_t a = ph.a + ph.chunk * p.a_chunk,
                  out = ph.out + ph.chunk * p.out_chunk;
    const int64_t per_stream = int64_t(m.kc) * m.nt * 2 * kRec;
    for (int s = 0; s < kStreams; ++s) {
      x.Fill(s, kW, ph.header_w, Linear(kRec));
      for (int pass = 0; pass < p.passes; ++pass) {
        const int tiles = std::min(p.pass, m.nt - pass * p.pass);
        for (int seg = 0; seg < p.segs; ++seg) {
          const int chunks = std::min(p.seg, m.kc - seg * p.seg);
          x.Fill(s, kS, a + int64_t(seg) * p.g * p.seg * 2 * kRec,
                 Linear(int64_t(p.g) * chunks * 2 * kRec));
          const int64_t b = m.w + s * per_stream +
                            int64_t(seg) * p.seg * m.nt * 2 * kRec +
                            int64_t(pass) * p.pass * 2 * kRec;
          if (p.passes == 1) {
            x.Fill(s, kW, b, Linear(int64_t(chunks) * m.nt * 2 * kRec));
          } else {
            Dims d;
            d.size = {1, uint32_t(chunks), uint32_t(tiles * 4),
                      uint32_t(kRec / 2)};
            d.stride = {0, uint32_t(m.nt * 2 * kRec), uint32_t(kRec / 2), 1};
            x.Fill(s, kW, b, d);
          }
        }
      }
    }
    // column c: records [tile][group][core row j] of 4 x 128 words
    for (int c = 0; c < kCols; ++c)
      for (int pass = 0; pass < p.passes; ++pass) {
        const int tiles = std::min(p.pass, m.nt - pass * p.pass);
        x.Drain(c,
                out + int64_t(c) * m.nt * p.g * 4 * kOut +
                    int64_t(pass) * p.pass * p.g * 4 * kOut,
                Linear(int64_t(tiles) * p.g * 4 * kOut));
      }
    x.End();
  }

  void EmitRow(Txn& t, const Phase& ph) {
    PhaseTxn x(t);
    const int K = ph.k, fl = ph.flags;
    const int xs = K + kOut;
    const bool has_x = fl & GUFO_XRT_ROW_X;
    const Matrix* om = ph.o_id >= 0 ? &matrices[size_t(ph.o_id)] : nullptr;
    const Plan op = om ? MakePlan(*om, ph.rows) : Plan{};
    for (int s = 0; s < kStreams; ++s)
      for (int par = 0; par < 2; ++par) {
        const int r = ph.round * 32 + 2 * s + par;
        if (par == 0)
          x.Fill(s, kW, ph.header_w, Linear(kRec + ph.pad + ph.lc));
        else
          x.Fill(s, kW, ph.header_w + kRec, Linear(ph.pad + ph.lc));
        x.Fill(s, kS, ph.x_in + int64_t(r) * xs + (has_x ? 0 : K),
               Linear(ph.lx));
        if (om) {
          // the row's 8 floats of every tile: [column c][core row j][tile t][8]
          const int chunk = r / op.chunk_rows, rr = r % op.chunk_rows,
                    g = rr / 8, mm = rr % 8;
          Dims d;
          d.size = {uint32_t(kCols), 4, uint32_t(om->nt), 8};
          d.stride = {uint32_t(om->nt * op.g * 4 * kOut), uint32_t(kOut),
                      uint32_t(op.g * 4 * kOut), 1};
          x.Fill(
              s, kS,
              ph.o_out + chunk * op.out_chunk + int64_t(g) * 4 * kOut + mm * 8,
              d);
        }
      }
    const int nx = (fl & GUFO_XRT_ROW_STORE_X) ? K / kOut + 1 : 1;
    const int64_t junk = junk_words;
    for (int c = 0; c < kCols; ++c) {
      const int r0 = ph.round * 32 + 4 * c;
      const bool valid = r0 < ph.rows;
      {
        Dims d;
        d.size = {1, uint32_t(nx), 4, uint32_t(kOut)};
        d.stride = {0, uint32_t(kOut), uint32_t(xs), 1};
        const int64_t at = (fl & GUFO_XRT_ROW_STORE_X)
                               ? ph.x_out + int64_t(r0) * xs
                               : ph.x_out + int64_t(r0) * xs + K;
        x.Drain(c, valid ? at : junk, d);
      }
      if (fl & GUFO_XRT_ROW_F32) {
        Dims d;
        d.size = {1, uint32_t(K / kOut), 4, uint32_t(kOut)};
        d.stride = {0, uint32_t(kOut), uint32_t(ph.stride), 1};
        x.Drain(c, valid ? ph.dst + int64_t(r0) * ph.stride : junk, d);
      } else {
        const Matrix& am = matrices.at(size_t(ph.a_id));
        const Plan ap = MakePlan(am, ph.rows);
        const int chunk = r0 / ap.chunk_rows, rr = r0 % ap.chunk_rows,
                  g = rr / 8, mm0 = rr % 8;
        const int halves = K / 256;
        for (int seg = 0; seg < ap.segs; ++seg) {
          const int h0 = seg * ap.seg * 2,
                    n = std::min(ap.seg * 2, halves - h0);
          if (n <= 0)
            break;
          Dims d;
          d.size = {uint32_t(n), 4, 64, 2};
          d.stride = {uint32_t(kRec), 2, 16, 1};
          const int64_t at =
              ph.dst + chunk * ap.a_chunk +
              int64_t(seg) * ap.g * ap.seg * 2 * kRec +
              int64_t(g) * std::min(ap.seg, am.kc - seg * ap.seg) * 2 * kRec +
              mm0 * 2;
          x.Drain(c, valid ? at : junk, d);
        }
      }
    }
    x.End();
  }

  int64_t junk_words{-1};

  int End() {
    if (junk_words < 0)
      junk_words = Alloc(1 << 20);
    Txn t;
    for (const Phase& ph : building)
      (ph.id >= 0 ? EmitGemm(t, ph) : EmitRow(t, ph));
    building.clear();
    const auto words = t.Finish();
    Program p{xrt::bo(dev, words.size() * 4, xrt::bo::flags::cacheable,
                      kernel.group_id(1)),
              uint32_t(words.size())};
    p.ins.write(words.data(), words.size() * 4, 0);
    p.ins.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    programs.push_back(std::move(p));
    return int(programs.size()) - 1;
  }

  void Run(int id) {
    auto& p = programs.at(size_t(id));
    s.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    kernel(3u, p.ins, p.words, s, w).wait();
    s.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }
};

extern "C" {

gufo_xrt_engine* gufo_xrt_engine_create(const char* dir) {
  try {
    return new gufo_xrt_engine(dir);
  } catch (const std::exception& e) {
    g_error = e.what();
    return nullptr;
  }
}
void gufo_xrt_engine_destroy(gufo_xrt_engine* e) {
  delete e;
}
const char* gufo_xrt_engine_last_error(void) {
  return g_error.c_str();
}

#define ENGINE_TRY(expr, fail)         \
  try {                                \
    std::lock_guard lock(e->mutex);    \
    return expr;                       \
  } catch (const std::exception& ex) { \
    g_error = ex.what();               \
    return fail;                       \
  }

int gufo_xrt_engine_matrix(gufo_xrt_engine* e, int n, int k,
                           const int8_t* codes, const float* scales){
    ENGINE_TRY(e -> AddMatrix(n, k, codes, scales), -1)} int64_t
    gufo_xrt_engine_alloc(gufo_xrt_engine* e, int64_t words) {
  ENGINE_TRY(e->Alloc(words), -1)
}
float* gufo_xrt_engine_scratch(gufo_xrt_engine* e) {
  return static_cast<float*>(e->s.map());
}
int64_t gufo_xrt_engine_a_words(gufo_xrt_engine* e, int id,
                                int rows){ENGINE_TRY(
    [&] {
      const Plan p = MakePlan(e->matrices.at(size_t(id)), rows);
      return p.a_chunk * p.Chunks();
    }(),
    -1)} int64_t
    gufo_xrt_engine_out_words(gufo_xrt_engine* e, int id, int rows) {
  ENGINE_TRY(
      [&] {
        const Plan p = MakePlan(e->matrices.at(size_t(id)), rows);
        return p.out_chunk * p.Chunks();
      }(),
      -1)
}
int gufo_xrt_engine_begin(gufo_xrt_engine* e) {
  ENGINE_TRY((e->building.clear(), 0), -1)
}
int gufo_xrt_engine_row(gufo_xrt_engine* e, int flags, int rows, int k,
                        int64_t x_in, int64_t x_out, int o_id, int64_t o_out,
                        const float* consts, int n_consts, float eps, int a_id,
                        int64_t a, int stride) {
  ENGINE_TRY(e->Row(flags, rows, k, x_in, x_out, o_id, o_out, consts, n_consts,
                    eps, a_id, a, stride),
             -1)
}
int gufo_xrt_engine_gemm(gufo_xrt_engine* e, int id, int rows, int64_t a,
                         int64_t out) {
  ENGINE_TRY(e->Gemm(id, rows, a, out), -1)
}
int gufo_xrt_engine_end(gufo_xrt_engine* e) {
  ENGINE_TRY(e->End(), -1)
}
int gufo_xrt_engine_run(gufo_xrt_engine* e, int program) {
  ENGINE_TRY((e->Run(program), 0), -1)
}

}  // extern "C"
