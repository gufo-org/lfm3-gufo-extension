#ifndef GUFO_MODELS_LFM3_ENGINE_HPP_
#define GUFO_MODELS_LFM3_ENGINE_HPP_
#include <memory>

#include "tokenizer.hpp"
#include "weights.hpp"
namespace gufo::models::lfm3 {
class ContinuationHooks;
class Session;
enum class ContinuationTensor {
  kLogits,
  kFrontier,
  kConvolution,
  kKey,
  kValue
};
class Model : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  static constexpr std::uint32_t kModelContext = ~0U;
  // A nonzero session_context (kModelContext: the model's own) allocates
  // the state of the first session of that context while weights load.
  static std::shared_ptr<Model> Load(const std::string& path,
                                     std::string* error = nullptr,
                                     std::uint32_t session_context = 0);
  std::unique_ptr<Session> CreateSession(std::uint32_t context,
                                         std::string* error = nullptr);
  const Config& config() const;
  const Tokenizer& tokenizer() const;
  // The backbone runs on the XDNA NPU when GUFO_LFM3_DEVICE=npu at load.
  // NPU sessions keep their state in host memory: DeviceHidden and
  // ContinuationData return host pointers and DeviceStream is null.
  bool OnNpu() const;
  // The whole model runs on the XDNA NPU through XRT when
  // GUFO_LFM3_DEVICE=xrt at load (xrt/engine.hpp): sessions take tokens,
  // keep no continuation rows and support no snapshots or embeddings.
  bool OnXrt() const;

private:
  Model();
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class Session;
};
class Session {
public:
  static constexpr std::size_t kPrefillChunk = 512;
  static constexpr std::size_t kMaxBatch = 8;
  ~Session();
  // d1 complete prompt: full attention is bidirectional, while
  // convolution and sliding attention remain causal. Starts from empty state.
  // Embeddings, when supplied, replace every input row (host floats).
  bool ForwardDecision(std::span<const std::uint32_t> tokens,
                       std::span<const float> embeddings,
                       std::span<float> logits, std::string* error = nullptr);
  // Packed d1: -1 marks state rows; positive IDs mark question groups.
  // Full-attention state rows see all rows; question rows see the state and
  // their own group. Empty groups means one bidirectional prompt. Projects
  // only positions through vocabulary (empty vocabulary means every token).
  // logits is [positions.size()][vocabulary.size() or config.vocab].
  bool ForwardDecision(std::span<const std::uint32_t> tokens,
                       std::span<const float> embeddings,
                       std::span<const std::int32_t> groups,
                       std::span<const std::uint32_t> positions,
                       std::span<const std::uint32_t> vocabulary,
                       std::span<float> logits, std::string* error = nullptr);
  // Hint that `token` is the next one advanced: on XRT its decode starts now,
  // so host work overlaps the NPU. Correct whatever is advanced next.
  void Speculate(std::uint32_t token);
  // Advances text tokens and optionally returns the final logits.
  // State is session-owned; device weights are shared by sessions.
  bool Forward(std::span<const std::uint32_t> tokens,
               std::span<float> logits = {}, std::string* error = nullptr);
  // One token per independent session, sharing dense projections across rows.
  static bool ForwardBatch(std::span<Session* const> sessions,
                           std::span<const std::uint32_t> tokens,
                           std::span<std::span<float>> logits,
                           std::string* error = nullptr);
  // One row per independent session, sharing dense projections across rows:
  // rows[i], when set, is session i's input embedding in device memory
  // (ready before the call), otherwise tokens[i] is. Logits are computed
  // only when requested; DeviceHidden of every session is updated.
  static bool ForwardBatchInputs(std::span<Session* const> sessions,
                                 std::span<const std::uint32_t> tokens,
                                 std::span<const float* const> rows,
                                 bool logits, std::string* error = nullptr);
  // Advances rows of input embeddings (hidden floats per row) instead of
  // token IDs. Rows live in host memory, or in device memory on this
  // session's stream when on_device is set.
  bool ForwardEmbeddings(std::span<const float> rows, bool on_device = false,
                         std::span<float> logits = {},
                         std::string* error = nullptr);
  // A run of inputs: token IDs, or rows of input embeddings (hidden floats
  // each) in host memory.
  struct Input {
    std::span<const std::uint32_t> tokens;
    std::span<const float> rows;
  };
  // Advances runs of tokens and embedding rows in order. Adjacent runs share
  // their passes (kPrefillChunk rows each), so a prompt mixing text and
  // embeddings reads the weights once per pass rather than once per run.
  bool ForwardInputs(std::span<const Input> inputs,
                     std::span<float> logits = {},
                     std::string* error = nullptr);
  // Final-normed hidden state of the last advanced row (hidden floats).
  bool ReadHidden(std::span<float> hidden, std::string* error = nullptr) const;
  // Device copy of the same hidden state, valid until the next forward.
  const float* DeviceHidden() const;
  void* DeviceStream() const;
  bool Reset(std::string* error = nullptr);
  std::uint32_t Position() const;
  // Payloads are compact, immutable host bytes. Persistent callers must bind
  // them to an artifact identity as well as this versioned state layout.
  std::size_t SnapshotBytes() const;
  std::vector<std::uint8_t> SaveSnapshot() const;
  bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                       std::string* error = nullptr);
  bool ReadLogits(std::span<float> logits, std::string* error = nullptr) const;

  // Continuation components. A session with hooks is owned by the cache
  // adapter, which reads device state directly instead of through a snapshot.
  void SetContinuationHooks(ContinuationHooks* hooks);
  void* ContinuationData(ContinuationTensor tensor,
                         std::size_t layer = 0) const;
  std::uint32_t RingCapacity(std::size_t layer) const;
  // Whether this session's state lives in host memory (NPU backbone).
  bool HostState() const { return npu_ != nullptr; }
  // False after a device error, until Reset.
  bool IsCoherent() const;
  // Completes the frontier's logits on this session's stream.
  void PrepareLogits();
  void BeginRestore();
  void FinishRestore(std::uint32_t position);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  struct Npu;
  std::unique_ptr<Npu> npu_;
  struct Xrt;
  std::unique_ptr<Xrt> xrt_;
  ContinuationHooks* hooks_{};
  Session(std::shared_ptr<Model> model, std::uint32_t context);
  friend class Model;
};
}  // namespace gufo::models::lfm3
#endif
