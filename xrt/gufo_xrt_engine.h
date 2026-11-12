// The NPU stack engine through XRT (xrt/iron/engine): Q8_0 products and row
// operations as phases of one dispatch. A program is a list of phases over
// regions of the scratch buffer S (f32 words); the host writes inputs into S,
// runs the program, and reads results back.
//
// Products take A rows quantized by a row operation (int16 codes; their row
// scale goes to the meta word of an x region row) and leave f32 tiles that a
// later row operation reads. x regions hold rows of k floats then a 128-word
// meta record (row stride k + 128); a row operation reading products reads the
// meta of its x region rows for their A scale.
#ifndef GUFO_XRT_ENGINE_H_
#define GUFO_XRT_ENGINE_H_
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct gufo_xrt_engine gufo_xrt_engine;

// dir holds engine.xclbin.
gufo_xrt_engine* gufo_xrt_engine_create(const char* dir);
void gufo_xrt_engine_destroy(gufo_xrt_engine* e);
const char* gufo_xrt_engine_last_error(void);

// A Q8_0 matrix of n outputs x k inputs (k a multiple of 32): codes [n][k],
// block scales [n][k / 32].
int gufo_xrt_engine_matrix(gufo_xrt_engine* e, int n, int k,
                           const int8_t* codes, const float* scales);
// Scratch regions of S, zeroed (words); returns the word offset or -1.
int64_t gufo_xrt_engine_alloc(gufo_xrt_engine* e, int64_t words);
float* gufo_xrt_engine_scratch(gufo_xrt_engine* e);  // host view of S
// Words of S the A rows / product tiles of matrix `id` over `rows` rows need.
int64_t gufo_xrt_engine_a_words(gufo_xrt_engine* e, int id, int rows);
int64_t gufo_xrt_engine_out_words(gufo_xrt_engine* e, int id, int rows);

// Row operation flags (engine.cc).
enum {
  GUFO_XRT_ROW_O =
      1,  // add the product rows (times the A row scale in the x region meta)
  GUFO_XRT_ROW_X = 2,  // the x region rows are input
  GUFO_XRT_ROW_BIAS =
      4,  // constants hold a bias (after the norm weight, if any)
  GUFO_XRT_ROW_SWIGLU = 8,  // products are [gate | up]: silu(gate) up
  GUFO_XRT_ROW_NORM = 16,   // RMS norm with the constants' weight
  GUFO_XRT_ROW_STORE_X =
      32,  // write the result (before the norm) back as x rows (+ meta)
  GUFO_XRT_ROW_F32 = 64,  // write f32 rows instead of A codes
};

// Program building: phases run in order within one dispatch.
int gufo_xrt_engine_begin(gufo_xrt_engine* e);
// A row operation over `rows` rows of width k. x_in: the x region read (rows of
// k floats and meta with ROW_X, else the meta only); x_out: the x region
// written (rows and meta with STORE_X, else the meta only). o_id/o_out: the
// matrix and tile region of the product read (ROW_O). consts/n_consts: the norm
// weight then the bias. a_id/a: the matrix and A region the codes are for, or
// with ROW_F32 the f32 rows at `a` with row stride `stride`.
int gufo_xrt_engine_row(gufo_xrt_engine* e, int flags, int rows, int k,
                        int64_t x_in, int64_t x_out, int o_id, int64_t o_out,
                        const float* consts, int n_consts, float eps, int a_id,
                        int64_t a, int stride);
// A product over `rows` rows by matrix `id`: A rows at `a`, f32 tiles to `out`.
int gufo_xrt_engine_gemm(gufo_xrt_engine* e, int id, int rows, int64_t a,
                         int64_t out);
// Finishes the program (instructions built once); returns its id.
int gufo_xrt_engine_end(gufo_xrt_engine* e);
int gufo_xrt_engine_run(gufo_xrt_engine* e, int program);
#ifdef __cplusplus
}
#endif
#endif
