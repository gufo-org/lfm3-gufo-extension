// C ABI over FastFlowLM's prebuilt LFM3 NPU engine (XRT build). Gufo loads
// this library with dlopen; it owns one engine and one NPU context.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct flm_lfm3 flm_lfm3;
// model_dir holds config.json and model.q4nx; asset_root holds xclbins/<model>.
// Returns NULL on failure; flm_lfm3_last_error() describes it.
flm_lfm3* flm_lfm3_create(const char* model_dir, const char* asset_root, int max_length);
void flm_lfm3_destroy(flm_lfm3* m);
const char* flm_lfm3_last_error(void);
int flm_lfm3_vocab(const flm_lfm3* m);
// Appends ids to the context and writes the last position's logits (vocab
// floats). Returns 0 on success.
int flm_lfm3_prefill(flm_lfm3* m, const int32_t* ids, int count, float* logits);
int flm_lfm3_decode(flm_lfm3* m, int32_t id, float* logits);
int flm_lfm3_length(flm_lfm3* m);
void flm_lfm3_clear(flm_lfm3* m);
#ifdef __cplusplus
}
#endif
