#ifndef GUFO_MODELS_LFM3_VL_HIP_OPS_HPP_
#define GUFO_MODELS_LFM3_VL_HIP_OPS_HPP_
#include <hip/hip_runtime.h>
namespace gufo::models::lfm3_vl::gpu {
void SpatialConv(const float* p, const float* w, float* out, int h, int width,
                 int hidden, hipStream_t stream);
void AxialRope(float* x, int h, int w, int heads, hipStream_t stream);
// Bidirectional attention of n FP32 rows with head dimension 64 on the
// matrix cores, keeping FP32 precision through split F16 operands. Requires
// heads / kv_heads in {1, 2, 4, 8}.
void Attention64(const float* q, const float* k, const float* v, float* out,
                 int n, int heads, int kv_heads, hipStream_t stream);
}  // namespace gufo::models::lfm3_vl::gpu
#endif
