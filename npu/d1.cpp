#include "npu/d1.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <gufo/ext/experimental/npu.hpp>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "npu/dense.hpp"

namespace gufo::models::lfm3::npu {
namespace {

namespace hw = gufo::npu;

std::map<std::string, std::vector<std::int64_t>> ReadMeta(
    const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("cannot open " + path.string());
  std::map<std::string, std::vector<std::int64_t>> m;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string key;
    s >> key;
    std::int64_t x;
    while (s >> x)
      m[key].push_back(x);
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

constexpr unsigned kRecords = 3, kWeights = 4, kScratch = 5;
constexpr std::size_t kMetaWords = 4608;  // one B object (bf16 slots)

struct Variant {
  int M{};
  std::size_t meta{};  // bf16 offset in the scratch
  hw::Program program;
  hw::Instructions ins;
  hw::Buffer scratch;
};

}  // namespace

struct Decider::Impl {
  hw::Device dev = hw::Device::Open();
  std::vector<Variant> variants;  // ascending M
  hw::Buffer records, weights;
  int hidden{}, vocab{}, rec{};
  std::vector<float> wout;  // output norm weight
  Dense::Matrix head;
};

Decider::Decider(const std::filesystem::path& dir, const Weights& w)
    : impl_(std::make_unique<Impl>()) {
  auto& m = *impl_;
  const auto meta = ReadMeta(dir / "bundle.txt");
  auto get = [&](const std::string& k, std::size_t i = 0) {
    auto it = meta.find(k);
    if (it == meta.end() || it->second.size() <= i)
      throw std::runtime_error("bundle.txt lacks " + k + " (not a d1 bundle?)");
    return it->second[i];
  };
  m.hidden = int(get("hidden"));
  m.vocab = int(get("vocab"));
  m.rec = int(get("p.rec"));
  if (std::uint32_t(m.hidden) != w.config.hidden)
    throw std::runtime_error("d1 bundle hidden size differs from the model's");
  std::vector<int> Ms;
  for (auto x : meta.at("d1.M"))
    Ms.push_back(int(x));
  std::sort(Ms.begin(), Ms.end());
  for (int M : Ms) {
    Variant v;
    v.M = M;
    v.meta = std::size_t(get("d1." + std::to_string(M) + ".meta"));
    const std::string stem = "d1_prefill" + std::to_string(M);
    v.program = hw::Program::Load(m.dev, dir / (stem + ".xclbin"));
    v.ins = v.program.LoadInstructions(dir / (stem + ".insts"));
    v.scratch = v.program.LoadBuffer(dir / (stem + "_scratch.bin"), kScratch);
    m.variants.push_back(std::move(v));
  }
  // records and weights are shared by the variants (the weights do not depend
  // on M)
  m.records = m.variants.back().program.Allocate(
      std::size_t(Ms.back()) * m.rec * 2, kRecords);
  m.weights = m.variants.back().program.LoadBuffer(
      dir / "d1_prefill_weights.bin", kWeights);
  m.wout.assign(std::size_t(m.hidden), 1.0f);
  DecodeRow(w.output_norm, 0, m.wout);
  Dense::Open(dir);
  const Tensor* head[] = {w.output.data ? &w.output : &w.embedding};
  m.head = Dense::Shared().Upload(head);
}

Decider::~Decider() = default;

std::uint32_t Decider::max_rows() const {
  return std::uint32_t(impl_->variants.back().M);
}

void Decider::Decide(std::span<const float> rows, std::uint32_t n,
                     std::span<const std::int32_t> groups,
                     std::span<const std::uint32_t> positions,
                     std::span<float> logits) {
  auto& m = *impl_;
  const std::size_t H = std::size_t(m.hidden);
  if (n == 0 || n > max_rows() || rows.size() < n * H)
    throw std::invalid_argument("d1 prompt on the NPU: 1.." +
                                std::to_string(max_rows()) + " rows");
  std::size_t vi = 0;
  while (std::uint32_t(m.variants[vi].M) < n)
    ++vi;
  Variant& v = m.variants[vi];
  // Records [xn | x | ...]: x rows of the prompt; the rows past it repeat row 0
  // (masked keys weigh 0, and they must stay finite).
  auto* R = reinterpret_cast<std::uint16_t*>(m.records.data());
  for (int t = 0; t < v.M; ++t) {
    const float* e =
        rows.data() + std::size_t(std::uint32_t(t) < n ? t : 0) * H;
    std::uint16_t* x = R + std::size_t(t) * m.rec + H;
    for (std::size_t i = 0; i < H; ++i)
      x[i] = Bf16(e[i]);
  }
  m.records.ToDevice(0, std::size_t(v.M) * m.rec * 2);
  // meta: int16 [length, grouped], int8 groups[M] (state rows -1)
  std::vector<std::uint8_t> meta(kMetaWords * 2, 0);
  const std::int16_t head[2] = {std::int16_t(n), std::int16_t(!groups.empty())};
  std::memcpy(meta.data(), head, 4);
  // Bidirectional rounds take key block 0 first: every row must see its first
  // key, so a packed prompt starts with a state row.
  if (!groups.empty() && groups[0] >= 0)
    throw std::invalid_argument(
        "d1 on the NPU needs packed prompts to start with a state row");
  // groups as int8: -1 state rows, the questions renumbered 0, 1, ... (skipping
  // -1)
  std::map<std::int32_t, std::int8_t> ids;
  for (std::size_t i = 0; i < groups.size() && i < n; ++i) {
    std::int8_t g = -1;
    if (groups[i] >= 0) {
      auto it = ids.find(groups[i]);
      if (it == ids.end()) {
        if (ids.size() >= 255)
          throw std::invalid_argument(
              "d1 on the NPU takes up to 255 question groups");
        const int k = int(ids.size());
        it = ids.emplace(groups[i], std::int8_t(k <= 127 ? k : 126 - k))
                 .first;  // 0..127, then -2..-128
      }
      g = it->second;
    }
    meta[4 + i] = std::uint8_t(g);
  }
  std::memcpy(v.scratch.data() + v.meta * 2, meta.data(), meta.size());
  v.scratch.ToDevice(v.meta * 2, meta.size());
  v.program.Start(v.ins, {&m.records, &m.weights, &v.scratch}).Wait();
  // The answer rows: the final norm (records' xn) times its weight, then the
  // head.
  m.records.FromDevice(0, std::size_t(n) * m.rec * 2);
  std::vector<float> hid(positions.size() * H);
  for (std::size_t p = 0; p < positions.size(); ++p) {
    if (positions[p] >= n)
      throw std::invalid_argument("d1 answer position past the prompt");
    const std::uint16_t* xn = R + std::size_t(positions[p]) * m.rec;
    for (std::size_t i = 0; i < H; ++i)
      hid[p * H + i] = FromBf16(xn[i]) * m.wout[i];
  }
  if (logits.size() < positions.size() * std::size_t(m.vocab))
    throw std::invalid_argument("d1 logits too small");
  Dense::Shared().Run(m.head, hid, std::uint32_t(positions.size()), logits);
}

}  // namespace gufo::models::lfm3::npu
