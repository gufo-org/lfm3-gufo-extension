// Q8_0 matrix products on the XDNA NPU through XRT (xrt/iron/gemm/q8gemm.py):
// y = W x for any row count, with weights resident on the device. x is
// quantized per row to int16; each 32-code block's integer dot is scaled by
// the block's weight scale in exact integer arithmetic (int16 fractions of the
// row's largest scale), so results match f32 activations to about 5e-5.
#ifndef GUFO_XRT_GEMM_H_
#define GUFO_XRT_GEMM_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct gufo_xrt_gemm gufo_xrt_gemm;
// dir holds q8gemm.xclbin and q8gemm_dispatch.so.
gufo_xrt_gemm* gufo_xrt_gemm_create(const char* dir);
void gufo_xrt_gemm_destroy(gufo_xrt_gemm* g);
const char* gufo_xrt_gemm_last_error(void);
// A matrix of n rows (outputs) x k columns: Q8_0 codes [n][k] and block scales
// [n][k / 32] (k a multiple of 32). Returns its id, or -1.
int gufo_xrt_gemm_upload(gufo_xrt_gemm* g, int n, int k, const int8_t* codes,
                         const float* scales);
// y [rows][n] = x [rows][k] times matrix `id` transposed. Returns 0, or -1.
int gufo_xrt_gemm_run(gufo_xrt_gemm* g, int id, const float* x, int rows,
                      float* y);
// Several independent products in one dispatch: y_i = x_i times matrix ids[i].
int gufo_xrt_gemm_run_batch(gufo_xrt_gemm* g, int count, const int* ids,
                            const float* const* xs, const int* rows,
                            float* const* ys);
#ifdef __cplusplus
}
#endif
#endif
