#ifndef GUFO_MODELS_LFM3_HIP_OPS_HPP_
#define GUFO_MODELS_LFM3_HIP_OPS_HPP_
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

#include "../weights.hpp"

namespace gufo::models::lfm3::gpu {
void Embed(const Tensor& weight, const std::uint32_t* tokens, float* out, int n,
           hipStream_t stream);
// Prompt projections of n rows need SplitActivationBytes(n, K) here.
// row_max (one word per prompt row) lets the FFN down projection split its
// input while staging.
// With partial (SplitPartialBytes) and zeroed tickets (kMaxSplitUnits
// words), projections whose tiles leave the last wave of resident blocks
// partly idle split those tiles across K.
struct DenseWorkspace {
  void* quantized{};
  unsigned* row_max{};
  float* partial{};
  unsigned* tickets{};
};
std::size_t DenseWorkspaceBytes(std::size_t rows, std::size_t K);
inline constexpr int kMaxSplitParts = 8, kMaxSplitUnits = 192;
std::size_t SplitPartialBytes();
struct Output {
  const Tensor* weight;
  float* out;
};
// True when these projections of n rows read one split prompt input.
bool SharesSplit(std::span<const Output> outputs, int n,
                 DenseWorkspace workspace);
// Projects one input through up to four weights with the same column count.
// Q8_0 prompts split the input once and share one launch. split_ready means
// the workspace already holds this input's split (see ResidualNorm).
void Project(std::span<const Output> outputs, const float* x, int n,
             hipStream_t stream, DenseWorkspace workspace,
             bool split_ready = false);
// silu(gate x) * (up x) into out; up_scratch holds up x when the projections
// cannot share the fused prompt launch.
void ProjectSwiGLU(const Tensor& gate, const Tensor& up, const float* x,
                   float* out, float* up_scratch, int n, hipStream_t stream,
                   DenseWorkspace workspace, bool split_ready = false);
// down(silu(gate x) * (up x)) into out, with the activation in hidden.
void ProjectFfn(const Tensor& gate, const Tensor& up, const Tensor& down,
                const float* x, float* hidden, float* up_scratch, float* out,
                int n, hipStream_t stream, DenseWorkspace workspace,
                bool split_ready = false);
// x += post_norm(branch) (branch may be null, post_norm may be null), then
// z = norm(x) unless norm is null. With split, z goes to the workspace as a
// split prompt input instead. Returns false when the width is unsupported.
bool ResidualNorm(const float* branch, const Tensor* post_norm, float* x,
                  const Tensor* norm, float* z, int n, int hidden, float eps,
                  bool split, hipStream_t stream, DenseWorkspace workspace);
void Dense(const Tensor& weight, const float* x, float* out, int n,
           hipStream_t stream, DenseWorkspace workspace = {});
void ShortConv(const float* projection, const float* weight, float* history,
               float* out, int n, int hidden, int width, hipStream_t stream);
// Writes the dim / 2 RoPE frequencies of theta for Rope.
void RopeFrequencies(float* out, int dim, float theta, hipStream_t stream);
void Rope(float* x, const float* frequencies, const float* factors, int n,
          int heads, int dim, int position, hipStream_t stream);
// IEEE binary16 bits. Attention K/V caches and chunk rows are F16.
using Half = std::uint16_t;
// Causal attention of n query rows at position. Key rows at or after
// position come from k and v (n F16 rows), earlier rows from the ring
// (slot t % capacity), so a sliding ring is written after this launch. Prompt
// rows run on the matrix cores with F16 Q and probabilities; decode (n == 1)
// keeps Q and probabilities in FP32. Both accumulate in FP32. Long histories
// split their keys across blocks through scratch of
// AttentionScratchBytes(heads, dim, context, rows) bytes, where rows bounds
// the queries of one call.
void Attention(const float* q, const Half* k, const Half* v, const float* gate,
               const Half* cache_k, const Half* cache_v, float* out, int n,
               int heads, int kv_heads, int dim, int position, int capacity,
               int window, hipStream_t stream, float* scratch);
std::size_t AttentionScratchBytes(int heads, int dim, int context, int rows);
// Complete bidirectional prompt over n F16 K/V rows, without the ring, in
// FP32. A null gate leaves the output ungated.
void BidirectionalAttention(const float* q, const Half* k, const Half* v,
                            const float* gate, float* out, int n, int heads,
                            int kv_heads, hipStream_t stream,
                            const std::int32_t* groups = nullptr,
                            int dim = 128);
// The same over FP32 K/V rows, ungated, for stateless encoders that keep no
// cache to narrow.
void BidirectionalAttention(const float* q, const float* k, const float* v,
                            float* out, int n, int heads, int kv_heads, int dim,
                            hipStream_t stream);
// Head dimension 64 or 128 with at most eight query heads per KV head.
bool SupportsBidirectionalAttention(int heads, int kv_heads, int dim);
// Packed row copies; preserves the tensor's encoded row bytes exactly.
void SelectRows(const Tensor& source, const std::uint32_t* rows, void* out,
                int count, hipStream_t stream);
// Rounds count values of k and v to F16.
void PackKV(const float* k, const float* v, Half* k16, Half* v16,
            std::size_t count, hipStream_t stream);
// Copies n F16 rows into the ring at position.
void StoreKV(const Half* k, const Half* v, Half* cache_k, Half* cache_v, int n,
             int stride, int position, int capacity, hipStream_t stream);
// One decode row: per-head Q/K RMSNorm, RoPE and the F16 K/V cache store in
// one launch; Q and K match the separate launches. Attention then reads the
// current row from the cache slot.
// The same for n prompt rows at position, packing K and V rows into k16
// and v16 (n rows each) instead of a ring.
void PromptHeads(float* q, float* k, const float* v, const float* q_norm,
                 const float* k_norm, const float* frequencies,
                 const float* factors, Half* k16, Half* v16, int n, int heads,
                 int kv_heads, int dim, int position, float eps,
                 hipStream_t stream);
void DecodeHeads(float* q, float* k, const float* v, const float* q_norm,
                 const float* k_norm, const float* frequencies,
                 const float* factors, Half* cache_k, Half* cache_v, int heads,
                 int kv_heads, int dim, int position, int capacity, float eps,
                 hipStream_t stream);

void LaunchBatchedRMSNorm(const float* x, const float* weight, float* out,
                          void* out_bf16, std::size_t batch_size,
                          std::size_t dim, float eps, hipStream_t stream);
void LaunchBatchedPerHeadRMSNorm(const float* x, const float* weight,
                                 float* out, std::size_t batch_size,
                                 std::uint32_t num_heads,
                                 std::uint32_t head_dim, float eps,
                                 hipStream_t stream);
void LaunchBatchedResidualAdd(const float* a, const float* b, float* out,
                              std::size_t batch_size, std::size_t dim,
                              hipStream_t stream);
void LaunchBatchedSwiGLUActivation(const float* gate, const float* up,
                                   float* out, void* out_bf16,
                                   std::size_t num_elements,
                                   hipStream_t stream);
void LaunchGEMV(const void* A, core::GgmlType a_type, const float* x, float* y,
                std::size_t M, std::size_t K, hipStream_t stream);
void LaunchBatchedQuantGEMMFp32(core::GgmlType type, const void* w,
                                const float* fp32_x, float* y,
                                std::size_t batch, std::size_t m, std::size_t k,
                                hipStream_t stream);
void LaunchExactBf16GEMMFp32SmallBatch(const void* A, const float* X, float* Y,
                                       std::size_t batch_size, std::size_t M,
                                       std::size_t K, hipStream_t stream);
}  // namespace gufo::models::lfm3::gpu
#endif
