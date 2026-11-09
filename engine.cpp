#include "engine.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "continuation_hooks.hpp"
#include "hip/ops.hpp"
namespace gufo::models::lfm3 {
namespace {
void Check(hipError_t status) {
  if (status != hipSuccess)
    throw std::runtime_error(hipGetErrorString(status));
}
struct Buffer {
  void* data{};
  Buffer() = default;
  explicit Buffer(std::size_t bytes) {
    if (bytes)
      Check(hipMalloc(&data, bytes));
  }
  ~Buffer() {
    if (data)
      (void)hipFree(data);
  }
  Buffer(Buffer&& b) noexcept : data(std::exchange(b.data, nullptr)) {}
  Buffer& operator=(Buffer&& b) noexcept {
    std::swap(data, b.data);
    return *this;
  }
  Buffer(const Buffer&) = delete;
  float* floats() const { return static_cast<float*>(data); }
  gpu::Half* halves() const { return static_cast<gpu::Half*>(data); }
};
struct Stream {
  hipStream_t value{};
  Stream() { Check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
  ~Stream() {
    if (value)
      (void)hipStreamDestroy(value);
  }
};
template<class F>
void Visit(Weights& w, F f) {
  f(w.embedding);
  f(w.output);
  f(w.output_norm);
  f(w.rope_freqs);
  for (auto& l : w.layers)
    for (auto* t :
         {&l.norm, &l.ffn_norm, &l.post_norm, &l.post_ffn_norm, &l.gate, &l.up,
          &l.down, &l.conv, &l.in_proj, &l.out_proj, &l.q, &l.k, &l.v,
          &l.q_norm, &l.k_norm, &l.attn_gate, &l.attn_out})
      if (t->data)
        f(*t);
}
}  // namespace
struct Model::Impl {
  std::unique_ptr<core::GgufReader> reader;
  std::unique_ptr<Tokenizer> tokenizer;
  Weights weights;
  std::vector<Buffer> allocations;
};
Model::Model() : impl_(std::make_unique<Impl>()) {}
Model::~Model() = default;
std::shared_ptr<Model> Model::Load(const std::string& path,
                                   std::string* error) {
  try {
    auto m = std::shared_ptr<Model>(new Model);
    m->impl_->reader = core::GgufReader::OpenFile(path, error);
    if (!m->impl_->reader)
      return nullptr;
    auto w = Weights::Bind(*m->impl_->reader, error);
    if (!w)
      return nullptr;
    m->impl_->tokenizer = Tokenizer::Load(*m->impl_->reader, error);
    if (!m->impl_->tokenizer)
      return nullptr;
    m->impl_->weights = *w;
    // Tied embeddings share one allocation. Copy packed weights verbatim so
    // quantization and mapped-file lifetime never depend on a session.
    std::unordered_map<const void*, void*> uploaded;
    Visit(m->impl_->weights, [&](Tensor& t) {
      if (!uploaded.contains(t.data)) {
        m->impl_->allocations.emplace_back(t.bytes);
        auto* p = m->impl_->allocations.back().data;
        Check(hipMemcpy(p, t.data, t.bytes, hipMemcpyHostToDevice));
        uploaded[t.data] = p;
      }
      t.data = uploaded.at(t.data);
    });
    return m;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}
const Config& Model::config() const {
  return impl_->weights.config;
}
const Tokenizer& Model::tokenizer() const {
  return *impl_->tokenizer;
}
struct Session::Impl {
  static constexpr std::size_t kChunk = Session::kPrefillChunk;
  std::shared_ptr<Model> model;
  Stream stream;
  std::uint32_t context, position{};
  bool poisoned{}, logits_ready{};
  Buffer ids, x, z, out, p, q, k, v, gate, a, ff_gate, ff_up, logits, frontier,
      inputs, decision_groups;
  struct State {
    Buffer conv, k, v;
    std::uint32_t capacity{};
  };
  std::vector<State> states;
  Buffer quantized, row_max, split_partial, split_tickets, attention_scratch,
      frequencies, chunk_k, chunk_v;
  // Decision readout scratch, grown on demand and reused across prompts.
  Buffer decision_indices, decision_hidden, decision_projected,
      decision_selected;
  std::size_t rows, decision_group_bytes{}, decision_capacity[4]{};
  Impl(std::shared_ptr<Model> m, std::uint32_t ctx, std::size_t rows = kChunk)
      : model(std::move(m)),
        context(ctx),
        rows(rows),
        ids(rows * sizeof(std::uint32_t)),
        x(rows * model->config().hidden * 4),
        z(rows * model->config().hidden * 4),
        out(rows * model->config().hidden * 4),
        p(rows * 3 * model->config().hidden * 4),
        q(rows * model->config().heads * model->config().head_dim * 4),
        k(rows * model->config().heads * model->config().head_dim * 4),
        v(rows * model->config().heads * model->config().head_dim * 4),
        gate(rows * model->config().heads * 4),
        a(rows * model->config().heads * model->config().head_dim * 4),
        ff_gate(rows * model->config().intermediate * 4),
        ff_up(rows * model->config().intermediate * 4),
        logits(kMaxBatch * model->config().vocab * 4),
        frontier(model->config().hidden * 4),
        inputs(kMaxBatch * model->config().hidden * 4) {
    const auto& c = model->config();
    const auto width =
        std::max({c.hidden * 3, c.intermediate, c.heads * c.head_dim});
    quantized = Buffer(gpu::DenseWorkspaceBytes(rows, width));
    row_max = Buffer(rows * sizeof(unsigned));
    split_partial = Buffer(gpu::SplitPartialBytes());
    split_tickets = Buffer(gpu::kMaxSplitUnits * sizeof(unsigned));
    Check(hipMemsetAsync(split_tickets.data, 0,
                         gpu::kMaxSplitUnits * sizeof(unsigned), stream.value));
    attention_scratch =
        Buffer(gpu::AttentionScratchBytes(c.heads, c.head_dim, ctx, rows));
    // Global then sliding-window RoPE frequencies.
    frequencies = Buffer(std::size_t(c.head_dim) * 4);
    chunk_k = Buffer(rows * c.heads * c.head_dim * sizeof(gpu::Half));
    chunk_v = Buffer(rows * c.heads * c.head_dim * sizeof(gpu::Half));
    gpu::RopeFrequencies(frequencies.floats(), c.head_dim, c.theta,
                         stream.value);
    gpu::RopeFrequencies(frequencies.floats() + c.head_dim / 2, c.head_dim,
                         c.theta_swa, stream.value);
    Check(hipStreamSynchronize(stream.value));

    states.reserve(c.layers.size());
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
      State s;
      if (c.layers[i] == LayerType::kConvolution)
        s.conv = Buffer(std::size_t(c.hidden) * (c.conv_width - 1) * 4);
      else {
        s.capacity = c.layers[i] == LayerType::kSlidingAttention
                         ? std::min(ctx, c.window)
                         : ctx;
        s.k = Buffer(std::size_t(s.capacity) * c.kv_heads[i] * c.head_dim * 2);
        s.v = Buffer(std::size_t(s.capacity) * c.kv_heads[i] * c.head_dim * 2);
      }
      states.push_back(std::move(s));
    }
  }
  void EnsureLogits() {
    if (logits_ready)
      return;
    gpu::Dense(model->impl_->weights.output, frontier.floats(), logits.floats(),
               1, stream.value);
    Check(hipGetLastError());
    Check(hipStreamSynchronize(stream.value));
    logits_ready = true;
  }
  // Rows of embeddings replace the token lookup when embeddings is set; it
  // holds n * hidden floats on the host, or on the device when on_device.
  void Run(std::span<const std::uint32_t> tokens, bool output_logits,
           std::span<Impl* const> batch = {}, const float* embeddings = nullptr,
           std::size_t embedding_rows = 0, bool on_device = false,
           std::span<const std::int32_t> groups = {}) {
    const auto& w = model->impl_->weights;
    const auto& c = w.config;
    const auto s = stream.value;
    const auto n = embeddings ? embedding_rows : tokens.size();
    if (embeddings)
      Check(hipMemcpyAsync(
          x.data, embeddings, n * c.hidden * 4,
          on_device ? hipMemcpyDeviceToDevice : hipMemcpyHostToDevice, s));
    else {
      Check(hipMemcpyAsync(ids.data, tokens.data(), tokens.size_bytes(),
                           hipMemcpyHostToDevice, s));
      gpu::Embed(w.embedding, static_cast<const std::uint32_t*>(ids.data),
                 x.floats(), n, s);
    }
    if (!groups.empty()) {
      if (decision_group_bytes < groups.size_bytes()) {
        decision_groups = Buffer(groups.size_bytes());
        decision_group_bytes = groups.size_bytes();
      }
      Check(hipMemcpyAsync(decision_groups.data, groups.data(),
                           groups.size_bytes(), hipMemcpyHostToDevice, s));
    }
    const gpu::DenseWorkspace workspace{
        quantized.data, static_cast<unsigned*>(row_max.data),
        split_partial.floats(), static_cast<unsigned*>(split_tickets.data)};
    auto dense = [&](const Tensor& weight, const float* input, float* output,
                     int count, hipStream_t stream) {
      gpu::Dense(weight, input, output, count, stream, workspace);
    };
    auto norm = [&](const float* input, const Tensor& weight, float* output,
                    std::size_t count) {
      gpu::LaunchBatchedRMSNorm(input, static_cast<const float*>(weight.data),
                                output, nullptr, count, weight.cols, c.eps, s);
    };
    // Only the frontier row leaves the last layer, so trailing layers skip
    // rows nothing reads. Walking back, a convolution needs conv_width - 1
    // earlier rows for its outputs and history, and an attention layer
    // needs every row for its K/V cache (a sliding one also for its ring).
    // Layer i reads the last live_in[i] rows and updates the last
    // live_out[i] rows of the residual stream.
    const auto L = c.layers.size();
    std::vector<std::size_t> live_in(L, n), live_out(L, n);
    if (batch.empty() && !c.decision) {
      std::size_t need = 1;
      for (std::size_t i = L; i-- > 0;) {
        if (c.layers[i] == LayerType::kSlidingAttention)
          break;
        live_out[i] = std::min(n, need);
        if (c.layers[i] == LayerType::kFullAttention)
          break;
        need = std::min(n, need + c.conv_width - 1);
        live_in[i] = need;
      }
    }
    // x += post(branch), then z = norm(x), split for the next projections
    // when they all read split prompt inputs, over the last rows rows.
    // Returns whether z was split.
    auto residual_norm = [&](const float* branch, const Tensor* post,
                             const Tensor* next,
                             std::span<const gpu::Output> consumers,
                             std::size_t rows) {
      const auto o = (n - rows) * c.hidden;
      if (branch)
        branch += o;
      float *xr = x.floats() + o, *zr = z.floats() + o;
      const bool split = next && gpu::SharesSplit(consumers, rows, workspace);
      if (gpu::ResidualNorm(branch, post && post->data ? post : nullptr, xr,
                            next, zr, rows, c.hidden, c.eps, split, s,
                            workspace))
        return split;
      if (branch) {
        if (post && post->data)
          norm(branch, *post, out.floats() + o, rows);
        gpu::LaunchBatchedResidualAdd(
            xr, post && post->data ? out.floats() + o : branch, xr, rows,
            c.hidden, s);
      }
      if (next)
        norm(xr, *next, zr, rows);
      return false;
    };
    // Projections reading the normalized input of layer i's mixer, into its
    // last live_in[i] rows.
    std::array<gpu::Output, 4> mixer;
    std::span<const gpu::Output> inputs;
    const auto mixer_inputs = [&](std::size_t i) {
      const auto& l = w.layers[i];
      const auto o = n - live_in[i];
      if (c.layers[i] == LayerType::kConvolution) {
        mixer[0] = {&l.in_proj, p.floats() + o * 3 * c.hidden};
        return std::span<const gpu::Output>(mixer.data(), 1);
      }
      mixer = {{{&l.q, q.floats()},
                {&l.k, k.floats()},
                {&l.v, v.floats()},
                {&l.attn_gate, gate.floats()}}};
      return std::span<const gpu::Output>(mixer);
    };
    inputs = mixer_inputs(0);
    bool split =
        residual_norm(nullptr, nullptr, &w.layers[0].norm, inputs, live_in[0]);
    for (std::size_t i = 0; i < c.layers.size(); ++i) {
      const auto& l = w.layers[i];
      const auto rows_in = live_in[i], rows_out = live_out[i];
      const auto o_in = n - rows_in, o = n - rows_out;
      if (c.layers[i] == LayerType::kConvolution) {
        gpu::Project(inputs, z.floats() + o_in * c.hidden, rows_in, s,
                     workspace, split);
        if (batch.empty()) {
          // Rows before the first live one are not read: outputs that
          // would need them are not live either.
          gpu::ShortConv(p.floats() + o_in * 3 * c.hidden,
                         static_cast<const float*>(l.conv.data),
                         states[i].conv.floats(), a.floats() + o_in * c.hidden,
                         rows_in, c.hidden, c.conv_width, s);
        } else
          for (std::size_t row = 0; row < batch.size(); ++row) {
            gpu::ShortConv(p.floats() + row * 3 * c.hidden,
                           static_cast<const float*>(l.conv.data),
                           batch[row]->states[i].conv.floats(),
                           a.floats() + row * c.hidden, 1, c.hidden,
                           c.conv_width, s);
          }
        dense(l.out_proj, a.floats() + o * c.hidden,
              out.floats() + o * c.hidden, rows_out, s);
      } else {
        gpu::Project(inputs, z.floats(), n, s, workspace, split);
        // Rows normalize, rotate and store K/V in one launch (per decode
        // row).
        const bool rows = !batch.empty() || (n == 1 && !c.decision);
        const bool swa = c.layers[i] == LayerType::kSlidingAttention;
        const auto* factors =
            swa ? nullptr : static_cast<const float*>(w.rope_freqs.data);
        const auto attend = [&](Impl& owner, std::size_t row,
                                std::size_t count) {
          auto& state = owner.states[i];
          const auto stride = std::size_t(c.kv_heads[i]) * c.head_dim;
          auto* qr = q.floats() + row * c.heads * c.head_dim;
          auto* kr = k.floats() + row * stride;
          auto* vr = v.floats() + row * stride;
          auto* ck = state.k.halves();
          auto* cv = state.v.halves();
          const float* table =
              frequencies.floats() + (swa ? c.head_dim / 2 : 0);
          auto* gr = gate.floats() + row * c.heads;
          auto* ar = a.floats() + row * c.heads * c.head_dim;
          if (rows) {
            gpu::DecodeHeads(
                qr, kr, vr, static_cast<const float*>(l.q_norm.data),
                static_cast<const float*>(l.k_norm.data), table, factors, ck,
                cv, c.heads, c.kv_heads[i], c.head_dim, owner.position,
                state.capacity, c.eps, s);
            const auto slot = owner.position % state.capacity * stride;
            gpu::Attention(qr, ck + slot, cv + slot, gr, ck, cv, ar, 1, c.heads,
                           c.kv_heads[i], c.head_dim, owner.position,
                           state.capacity, swa ? c.window : 0, s,
                           owner.attention_scratch.floats());
            return;
          }
          // Full-attention rows never wrap, so the chunk is packed in place.
          // A sliding ring is written only after attention read its history.
          gpu::Half* kc = swa ? chunk_k.halves() : ck + owner.position * stride;
          gpu::Half* vc = swa ? chunk_v.halves() : cv + owner.position * stride;
          gpu::PromptHeads(qr, kr, vr, static_cast<const float*>(l.q_norm.data),
                           static_cast<const float*>(l.k_norm.data), table,
                           factors, kc, vc, count, c.heads, c.kv_heads[i],
                           c.head_dim, owner.position, c.eps, s);
          if (c.decision && !swa)
            gpu::BidirectionalAttention(
                qr, kc, vc, gr, ar, count, c.heads, c.kv_heads[i], s,
                static_cast<const std::int32_t*>(decision_groups.data));
          else {
            // Only the last rows_out queries are live; the earlier chunk rows
            // are already in the (full-attention) cache.
            const auto skip = count - std::min(count, rows_out);
            gpu::Attention(qr + skip * c.heads * c.head_dim, kc + skip * stride,
                           vc + skip * stride, gr + skip * c.heads, ck, cv,
                           ar + skip * c.heads * c.head_dim, count - skip,
                           c.heads, c.kv_heads[i], c.head_dim,
                           owner.position + skip, state.capacity,
                           swa ? c.window : 0, s,
                           owner.attention_scratch.floats());
          }
          if (swa)
            gpu::StoreKV(kc, vc, ck, cv, count, stride, owner.position,
                         state.capacity, s);
        };
        if (batch.empty())
          attend(*this, 0, n);
        else
          for (std::size_t row = 0; row < batch.size(); ++row)
            attend(*batch[row], row, 1);
        dense(l.attn_out, a.floats() + o * c.heads * c.head_dim,
              out.floats() + o * c.hidden, rows_out, s);
      }
      const gpu::Output ffn[] = {{&l.gate, ff_gate.floats()},
                                 {&l.up, ff_up.floats()}};
      split =
          residual_norm(out.floats(), &l.post_norm, &l.ffn_norm, ffn, rows_out);
      gpu::ProjectFfn(l.gate, l.up, l.down, z.floats() + o * c.hidden,
                      ff_gate.floats(), ff_up.floats(),
                      out.floats() + o * c.hidden, rows_out, s, workspace,
                      split);
      const bool last = i + 1 == c.layers.size();
      if (!last)
        inputs = mixer_inputs(i + 1);
      split = residual_norm(out.floats(), &l.post_ffn_norm,
                            last ? nullptr : &w.layers[i + 1].norm, inputs,
                            rows_out);
    }
    if (batch.empty()) {
      norm(x.floats() + (n - 1) * c.hidden, w.output_norm, frontier.floats(),
           1);
      logits_ready = output_logits;
      if (output_logits)
        dense(w.output, frontier.floats(), logits.floats(), 1, s);
    } else {
      norm(x.floats(), w.output_norm, z.floats(), n);
      if (output_logits)
        dense(w.output, z.floats(), logits.floats(), n, s);
      for (std::size_t row = 0; row < batch.size(); ++row) {
        Check(hipMemcpyAsync(
            batch[row]->frontier.data, z.floats() + row * c.hidden,
            std::size_t(c.hidden) * 4, hipMemcpyDeviceToDevice, s));
        if (row && output_logits)
          Check(hipMemcpyAsync(
              batch[row]->logits.data, logits.floats() + row * c.vocab,
              std::size_t(c.vocab) * 4, hipMemcpyDeviceToDevice, s));
      }
    }
    Check(hipGetLastError());
    Check(hipStreamSynchronize(s));
    if (batch.empty())
      position += n;
    else
      for (auto* row : batch) {
        ++row->position;
        row->logits_ready = output_logits;
      }
  }
};
Session::Session(std::shared_ptr<Model> m, std::uint32_t context)
    : impl_(std::make_unique<Impl>(std::move(m), context)) {}
Session::~Session() {
  if (impl_)
    (void)hipStreamSynchronize(impl_->stream.value);
}
std::unique_ptr<Session> Model::CreateSession(std::uint32_t context,
                                              std::string* error) {
  try {
    if (!context || context > config().context)
      throw std::invalid_argument("invalid LFM3 context");
    auto session =
        std::unique_ptr<Session>(new Session(shared_from_this(), context));
    if (!session->Reset(error))
      return nullptr;
    return session;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return nullptr;
  }
}
bool Session::ForwardDecision(std::span<const std::uint32_t> tokens,
                              std::span<const float> embeddings,
                              std::span<float> logits, std::string* error) {
  // Preserve the original full-vocabulary single-slot path for callers
  // requiring the existing d1 decision probabilities bit for bit.
  try {
    const auto& c = impl_->model->config();
    if (!c.decision || tokens.empty() || tokens.size() > impl_->context ||
        logits.size() != c.vocab ||
        (!embeddings.empty() && embeddings.size() != tokens.size() * c.hidden))
      throw std::invalid_argument("invalid d1 prompt, embeddings or logits");
    for (auto id : tokens)
      if (id >= c.vocab)
        throw std::invalid_argument("d1 token out of range");
    // A decision prompt runs in one pass: grow the rows only when needed.
    if (impl_->rows < tokens.size() || impl_->poisoned) {
      auto state = std::make_unique<Impl>(
          impl_->model, impl_->context, std::max(kPrefillChunk, tokens.size()));
      impl_.swap(state);
    }
    if (!Reset(error))
      return false;
    impl_->Run(tokens, true, {},
               embeddings.empty() ? nullptr : embeddings.data(),
               embeddings.empty() ? 0 : tokens.size());
    Check(hipMemcpy(logits.data(), impl_->logits.data, logits.size_bytes(),
                    hipMemcpyDeviceToHost));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Session::ForwardDecision(std::span<const std::uint32_t> tokens,
                              std::span<const float> embeddings,
                              std::span<const std::int32_t> groups,
                              std::span<const std::uint32_t> positions,
                              std::span<const std::uint32_t> vocabulary,
                              std::span<float> logits, std::string* error) {
  try {
    const auto& c = impl_->model->config();
    const auto columns = vocabulary.empty() ? c.vocab : vocabulary.size();
    if (!c.decision || tokens.empty() || tokens.size() > impl_->context ||
        positions.empty() || positions.size() > tokens.size() ||
        logits.size() != positions.size() * columns ||
        (!groups.empty() && groups.size() != tokens.size()) ||
        (!embeddings.empty() && embeddings.size() != tokens.size() * c.hidden))
      throw std::invalid_argument(
          "invalid d1 prompt, groups, slots or readout");
    for (auto id : tokens)
      if (id >= c.vocab)
        throw std::invalid_argument("d1 token out of range");
    for (auto id : vocabulary)
      if (id >= c.vocab)
        throw std::invalid_argument("d1 vocabulary ID out of range");
    for (std::size_t i = 0; i < positions.size(); ++i)
      if (positions[i] >= tokens.size() ||
          (i && positions[i - 1] >= positions[i]))
        throw std::invalid_argument(
            "d1 slots must be increasing prompt positions");
    for (std::size_t i = 0; i < groups.size(); ++i)
      if (groups[i] == 0 || groups[i] < -1 || groups[i] > 128 ||
          (i && groups[i] < groups[i - 1]))
        throw std::invalid_argument(
            "d1 groups must be a state prefix and ordered positive question "
            "IDs");
    // A decision prompt runs in one pass: grow the rows only when needed.
    if (impl_->rows < tokens.size() || impl_->poisoned) {
      auto state = std::make_unique<Impl>(
          impl_->model, impl_->context, std::max(kPrefillChunk, tokens.size()));
      impl_.swap(state);
    }
    if (!Reset(error))
      return false;
    // A lone question sees the entire prompt. Keep its original unmasked
    // specialization, including its FP32 arithmetic and scheduling.
    const bool packed =
        !groups.empty() &&
        std::any_of(groups.begin(), groups.end(), [&](auto group) {
          return group > 0 && group != groups.back();
        });
    impl_->Run(tokens, false, {},
               embeddings.empty() ? nullptr : embeddings.data(),
               embeddings.empty() ? 0 : tokens.size(), false,
               packed ? groups : std::span<const std::int32_t>{});
    const auto stream = impl_->stream.value;
    const auto& w = impl_->model->impl_->weights;
    const auto grow = [](Buffer& buffer, std::size_t& capacity,
                         std::size_t bytes) -> Buffer& {
      if (capacity < bytes) {
        buffer = Buffer(bytes);
        capacity = bytes;
      }
      return buffer;
    };
    auto& d = *impl_;
    auto& indices = grow(
        d.decision_indices, d.decision_capacity[0],
        std::max(positions.size(), vocabulary.size()) * sizeof(std::uint32_t));
    auto& hidden = grow(d.decision_hidden, d.decision_capacity[1],
                        positions.size() * c.hidden * 4);
    auto& projected =
        grow(d.decision_projected, d.decision_capacity[2], logits.size_bytes());
    Check(hipMemcpyAsync(indices.data, positions.data(), positions.size_bytes(),
                         hipMemcpyHostToDevice, stream));
    Tensor states{impl_->x.data, core::GgmlType::kF32, c.hidden,
                  std::uint32_t(tokens.size()), tokens.size() * c.hidden * 4};
    gpu::SelectRows(states, static_cast<const std::uint32_t*>(indices.data),
                    hidden.data, positions.size(), stream);
    gpu::LaunchBatchedRMSNorm(
        hidden.floats(), static_cast<const float*>(w.output_norm.data),
        hidden.floats(), nullptr, positions.size(), c.hidden, c.eps, stream);
    Tensor head = w.output;
    if (!vocabulary.empty()) {
      auto& selected = grow(d.decision_selected, d.decision_capacity[3],
                            vocabulary.size() * (head.bytes / head.rows));
      Check(hipMemcpyAsync(indices.data, vocabulary.data(),
                           vocabulary.size_bytes(), hipMemcpyHostToDevice,
                           stream));
      gpu::SelectRows(head, static_cast<const std::uint32_t*>(indices.data),
                      selected.data, vocabulary.size(), stream);
      head = {selected.data, head.type, head.cols,
              std::uint32_t(vocabulary.size()),
              vocabulary.size() * (head.bytes / head.rows)};
    }
    gpu::Dense(head, hidden.floats(), projected.floats(), positions.size(),
               stream, {impl_->quantized.data});
    Check(hipMemcpyAsync(logits.data(), projected.data, logits.size_bytes(),
                         hipMemcpyDeviceToHost, stream));
    Check(hipStreamSynchronize(stream));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Session::Forward(std::span<const std::uint32_t> tokens,
                      std::span<float> logits, std::string* error) {
  try {
    const auto& c = impl_->model->config();
    if (c.decision)
      throw std::invalid_argument("use the decision API for d1 models");
    if (impl_->poisoned)
      throw std::runtime_error(
          "LFM3 session requires Reset after a device error");
    if (tokens.empty() || tokens.size() > impl_->context - impl_->position ||
        (!logits.empty() && logits.size() != c.vocab))
      throw std::invalid_argument(
          "invalid LFM3 token/logit count or context exhausted");
    for (auto t : tokens)
      if (t >= c.vocab)
        throw std::invalid_argument("LFM3 token out of range");
    if (hooks_)
      hooks_->BeforeWrite(impl_->position,
                          impl_->position + std::uint32_t(tokens.size()));
    for (std::size_t offset = 0; offset < tokens.size();
         offset += Impl::kChunk) {
      auto count = std::min(Impl::kChunk, tokens.size() - offset);
      try {
        impl_->Run(tokens.subspan(offset, count),
                   !logits.empty() && offset + count == tokens.size());
      } catch (...) {
        impl_->poisoned = true;
        throw;
      }
    }
    if (!logits.empty())
      Check(hipMemcpy(logits.data(), impl_->logits.data, logits.size_bytes(),
                      hipMemcpyDeviceToHost));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Session::ForwardEmbeddings(std::span<const float> rows, bool on_device,
                                std::span<float> logits, std::string* error) {
  try {
    const auto& c = impl_->model->config();
    if (c.decision)
      throw std::invalid_argument("use the decision API for d1 models");
    if (impl_->poisoned)
      throw std::runtime_error(
          "LFM3 session requires Reset after a device error");
    const auto n = rows.size() / c.hidden;
    if (!n || rows.size() % c.hidden || n > impl_->context - impl_->position ||
        (!logits.empty() && logits.size() != c.vocab))
      throw std::invalid_argument(
          "invalid LFM3 embedding/logit count or context exhausted");
    if (hooks_)
      hooks_->BeforeWrite(impl_->position, impl_->position + std::uint32_t(n));
    for (std::size_t offset = 0; offset < n; offset += Impl::kChunk) {
      const auto count = std::min(Impl::kChunk, n - offset);
      try {
        impl_->Run({}, !logits.empty() && offset + count == n, {},
                   rows.data() + offset * c.hidden, count, on_device);
      } catch (...) {
        impl_->poisoned = true;
        throw;
      }
    }
    if (!logits.empty())
      Check(hipMemcpy(logits.data(), impl_->logits.data, logits.size_bytes(),
                      hipMemcpyDeviceToHost));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Session::ReadHidden(std::span<float> hidden, std::string* error) const {
  try {
    if (impl_->poisoned || !impl_->position ||
        hidden.size() != impl_->model->config().hidden)
      throw std::invalid_argument("invalid LFM3 hidden frontier");
    Check(hipMemcpy(hidden.data(), impl_->frontier.data, hidden.size_bytes(),
                    hipMemcpyDeviceToHost));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
const float* Session::DeviceHidden() const {
  return impl_->frontier.floats();
}
void* Session::DeviceStream() const {
  return impl_->stream.value;
}
bool Session::ForwardBatch(std::span<Session* const> sessions,
                           std::span<const std::uint32_t> tokens,
                           std::span<std::span<float>> outputs,
                           std::string* error) {
  try {
    if (sessions.empty() || sessions.size() > kMaxBatch ||
        tokens.size() != sessions.size() || outputs.size() != sessions.size())
      throw std::invalid_argument("invalid LFM3 decode batch shape");
    std::vector<Impl*> batch;
    for (std::size_t i = 0; i < sessions.size(); ++i) {
      auto* session = sessions[i];
      if (!session || std::find(sessions.begin(), sessions.begin() + i,
                                session) != sessions.begin() + i)
        throw std::invalid_argument("duplicate or null LFM3 batch session");
      auto* state = session->impl_.get();
      if (state->model->config().decision || state->poisoned ||
          state->position >= state->context ||
          tokens[i] >= state->model->config().vocab ||
          outputs[i].size() != state->model->config().vocab ||
          (!batch.empty() && state->model != batch.front()->model))
        throw std::invalid_argument("incompatible LFM3 batch session");
      batch.push_back(state);
    }
    for (auto* session : sessions)
      if (session->hooks_)
        session->hooks_->BeforeWrite(session->impl_->position,
                                     session->impl_->position + 1);
    try {
      batch.front()->Run(tokens, true, batch);
      for (std::size_t i = 0; i < batch.size(); ++i)
        Check(hipMemcpy(outputs[i].data(), batch[i]->logits.data,
                        outputs[i].size_bytes(), hipMemcpyDeviceToHost));
    } catch (...) {
      for (auto* state : batch)
        state->poisoned = true;
      throw;
    }
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Session::ForwardBatchInputs(std::span<Session* const> sessions,
                                 std::span<const std::uint32_t> tokens,
                                 std::span<const float* const> rows,
                                 bool logits, std::string* error) {
  try {
    if (sessions.empty() || sessions.size() > kMaxBatch ||
        tokens.size() != sessions.size() || rows.size() != sessions.size())
      throw std::invalid_argument("invalid LFM3 decode batch shape");
    std::vector<Impl*> batch;
    for (std::size_t i = 0; i < sessions.size(); ++i) {
      auto* session = sessions[i];
      if (!session || std::find(sessions.begin(), sessions.begin() + i,
                                session) != sessions.begin() + i)
        throw std::invalid_argument("duplicate or null LFM3 batch session");
      auto* state = session->impl_.get();
      if (state->model->config().decision || state->poisoned ||
          state->position >= state->context ||
          (!rows[i] && tokens[i] >= state->model->config().vocab) ||
          (!batch.empty() && state->model != batch.front()->model))
        throw std::invalid_argument("incompatible LFM3 batch session");
      batch.push_back(state);
    }
    for (auto* session : sessions)
      if (session->hooks_)
        session->hooks_->BeforeWrite(session->impl_->position,
                                     session->impl_->position + 1);
    auto& front = *batch.front();
    const auto& w = front.model->impl_->weights;
    const auto H = w.config.hidden;
    const auto s = front.stream.value;
    try {
      // The inputs gather into the front session's staging rows.
      Check(hipMemcpyAsync(front.ids.data, tokens.data(), tokens.size_bytes(),
                           hipMemcpyHostToDevice, s));
      for (std::size_t i = 0; i < batch.size(); ++i) {
        auto* row = front.inputs.floats() + i * H;
        if (rows[i])
          Check(hipMemcpyAsync(row, rows[i], std::size_t(H) * 4,
                               hipMemcpyDeviceToDevice, s));
        else
          gpu::Embed(w.embedding,
                     static_cast<const std::uint32_t*>(front.ids.data) + i, row,
                     1, s);
      }
      front.Run({}, logits, batch, front.inputs.floats(), batch.size(), true);
    } catch (...) {
      for (auto* state : batch)
        state->poisoned = true;
      throw;
    }
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
bool Session::Reset(std::string* error) {
  try {
    if (hooks_)
      hooks_->BeforeReset();
    Check(hipStreamSynchronize(impl_->stream.value));
    const auto& c = impl_->model->config();
    for (std::size_t i = 0; i < impl_->states.size(); ++i)
      if (impl_->states[i].conv.data)
        Check(hipMemsetAsync(impl_->states[i].conv.data, 0,
                             std::size_t(c.hidden) * (c.conv_width - 1) * 4,
                             impl_->stream.value));
    Check(hipStreamSynchronize(impl_->stream.value));
    impl_->position = 0;
    impl_->logits_ready = false;
    impl_->poisoned = false;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
std::uint32_t Session::Position() const {
  return impl_->position;
}
void Session::SetContinuationHooks(ContinuationHooks* hooks) {
  hooks_ = hooks;
}
void* Session::ContinuationData(ContinuationTensor tensor,
                                std::size_t layer) const {
  switch (tensor) {
    case ContinuationTensor::kLogits:
      return impl_->logits.data;
    case ContinuationTensor::kFrontier:
      return impl_->frontier.data;
    case ContinuationTensor::kConvolution:
      return impl_->states[layer].conv.data;
    case ContinuationTensor::kKey:
      return impl_->states[layer].k.data;
    case ContinuationTensor::kValue:
      return impl_->states[layer].v.data;
  }
  return nullptr;
}
std::uint32_t Session::RingCapacity(std::size_t layer) const {
  return impl_->states[layer].capacity;
}
bool Session::IsCoherent() const {
  return !impl_->poisoned;
}
void Session::PrepareLogits() {
  impl_->EnsureLogits();
}
void Session::BeginRestore() {
  Check(hipStreamSynchronize(impl_->stream.value));
  impl_->poisoned = true;
}
void Session::FinishRestore(std::uint32_t position) {
  impl_->position = position;
  impl_->logits_ready = true;
  impl_->poisoned = false;
}
namespace {
constexpr std::uint32_t kSnapshotMagic = 0x334d464c, kSnapshotVersion = 2;
constexpr std::size_t kSnapshotHeaderBytes = 32;
void WriteWord(std::span<std::uint8_t> data, std::size_t word,
               std::uint32_t value) {
  for (int i = 0; i < 4; ++i)
    data[word * 4 + i] = value >> (i * 8);
}
std::uint32_t ReadWord(std::span<const std::uint8_t> data, std::size_t word) {
  std::uint32_t value = 0;
  for (int i = 0; i < 4; ++i)
    value |= std::uint32_t(data[word * 4 + i]) << (i * 8);
  return value;
}
// Counts only executed rows. Sliding rings are copied in physical order, so
// restoring after a wrap preserves the absolute-position modulo mapping.
std::uint32_t ConfigSignature(const Config& c) {
  std::uint32_t hash = 2166136261U;
  const auto mix = [&](std::uint32_t value) {
    for (int i = 0; i < 4; ++i)
      hash = (hash ^ ((value >> (8 * i)) & 255)) * 16777619U;
  };
  for (auto value :
       {c.hidden, c.intermediate, c.vocab, c.heads, c.head_dim, c.window,
        c.conv_width, std::bit_cast<std::uint32_t>(c.eps),
        std::bit_cast<std::uint32_t>(c.theta),
        std::bit_cast<std::uint32_t>(c.theta_swa)})
    mix(value);
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    mix(static_cast<std::uint32_t>(c.layers[i]));
    mix(c.kv_heads[i]);
  }
  return hash;
}
std::size_t StateBytes(const Config& c, std::uint32_t position) {
  std::size_t bytes =
      kSnapshotHeaderBytes + (std::size_t(c.hidden) + c.vocab) * 4;
  for (std::size_t i = 0; i < c.layers.size(); ++i)
    bytes += c.layers[i] == LayerType::kConvolution
                 ? std::size_t(c.hidden) * (c.conv_width - 1) * 4
                 : std::size_t(c.layers[i] == LayerType::kSlidingAttention
                                   ? std::min(position, c.window)
                                   : position) *
                       c.kv_heads[i] * c.head_dim * 2 * sizeof(gpu::Half);
  return bytes;
}
}  // namespace
std::size_t Session::SnapshotBytes() const {
  return StateBytes(impl_->model->config(), impl_->position);
}
bool Session::ReadLogits(std::span<float> output, std::string* error) const {
  try {
    const auto& c = impl_->model->config();
    if (impl_->poisoned || !impl_->position || output.size() != c.vocab)
      throw std::invalid_argument("invalid LFM3 logit frontier");
    impl_->EnsureLogits();
    Check(hipMemcpy(output.data(), impl_->logits.data, output.size_bytes(),
                    hipMemcpyDeviceToHost));
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
std::vector<std::uint8_t> Session::SaveSnapshot() const {
  if (impl_->poisoned)
    throw std::runtime_error("cannot snapshot a poisoned LFM3 session");
  const auto& c = impl_->model->config();
  std::vector<std::uint8_t> bytes(SnapshotBytes());
  const std::uint32_t header[]{kSnapshotMagic,
                               kSnapshotVersion,
                               impl_->context,
                               impl_->position,
                               c.hidden,
                               c.vocab,
                               std::uint32_t(c.layers.size()),
                               ConfigSignature(c)};
  for (std::size_t i = 0; i < 8; ++i)
    WriteWord(bytes, i, header[i]);
  auto offset = kSnapshotHeaderBytes;
  auto copy = [&](const void* source, std::size_t size) {
    if (size)
      Check(hipMemcpy(bytes.data() + offset, source, size,
                      hipMemcpyDeviceToHost));
    offset += size;
  };
  if (impl_->position) {
    std::vector<float> output(c.vocab);
    std::string error;
    if (!ReadLogits(output, &error))
      throw std::runtime_error(error);
    copy(impl_->frontier.data, std::size_t(c.hidden) * 4);
    std::memcpy(bytes.data() + offset, output.data(), output.size() * 4);
    offset += output.size() * 4;
  } else
    offset += (std::size_t(c.hidden) + c.vocab) * 4;
  for (std::size_t i = 0; i < c.layers.size(); ++i) {
    const auto& state = impl_->states[i];
    if (state.conv.data)
      copy(state.conv.data, std::size_t(c.hidden) * (c.conv_width - 1) * 4);
    else {
      auto size = std::size_t(std::min(impl_->position, state.capacity)) *
                  c.kv_heads[i] * c.head_dim * sizeof(gpu::Half);
      copy(state.k.data, size);
      copy(state.v.data, size);
    }
  }
  return bytes;
}
bool Session::RestoreSnapshot(std::span<const std::uint8_t> bytes,
                              std::string* error) {
  try {
    if (hooks_)
      hooks_->BeforeRestore();
    const auto& c = impl_->model->config();
    // Validate the complete header and size before touching the existing state.
    if (bytes.size() < kSnapshotHeaderBytes ||
        ReadWord(bytes, 0) != kSnapshotMagic ||
        ReadWord(bytes, 1) != kSnapshotVersion ||
        ReadWord(bytes, 2) != impl_->context ||
        ReadWord(bytes, 3) > impl_->context || ReadWord(bytes, 4) != c.hidden ||
        ReadWord(bytes, 5) != c.vocab ||
        ReadWord(bytes, 6) != c.layers.size() ||
        ReadWord(bytes, 7) != ConfigSignature(c) ||
        bytes.size() != StateBytes(c, ReadWord(bytes, 3)))
      throw std::invalid_argument("invalid LFM3 snapshot payload");
    const auto position = ReadWord(bytes, 3);
    auto offset = kSnapshotHeaderBytes;
    auto copy = [&](void* destination, std::size_t size) {
      if (size)
        Check(hipMemcpyAsync(destination, bytes.data() + offset, size,
                             hipMemcpyHostToDevice, impl_->stream.value));
      offset += size;
    };
    try {
      Check(hipStreamSynchronize(impl_->stream.value));
      copy(impl_->frontier.data, std::size_t(c.hidden) * 4);
      copy(impl_->logits.data, std::size_t(c.vocab) * 4);
      for (std::size_t i = 0; i < c.layers.size(); ++i) {
        auto& state = impl_->states[i];
        if (state.conv.data)
          copy(state.conv.data, std::size_t(c.hidden) * (c.conv_width - 1) * 4);
        else {
          auto size = std::size_t(std::min(position, state.capacity)) *
                      c.kv_heads[i] * c.head_dim * sizeof(gpu::Half);
          copy(state.k.data, size);
          copy(state.v.data, size);
        }
      }
      Check(hipStreamSynchronize(impl_->stream.value));
    } catch (...) {
      impl_->poisoned = true;
      throw;
    }
    impl_->position = position;
    impl_->logits_ready = position != 0;
    impl_->poisoned = false;
    return true;
  } catch (const std::exception& e) {
    if (error)
      *error = e.what();
    return false;
  }
}
}  // namespace gufo::models::lfm3
