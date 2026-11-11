// C API of liblfm3_npu.so: compiles Loom programs for the XDNA NPU through
// HRX and runs them on device buffers. The extension loads it with dlopen, so
// its own build does not depend on HRX. Every call returns 0 on success; on
// failure lfm3_npu_error() describes the last error of that context.
#ifndef LFM3_NPU_H_
#define LFM3_NPU_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LFM3_NPU_EXPORT __attribute__((visibility("default")))

typedef struct lfm3_npu lfm3_npu;

#define LFM3_NPU_ABI_VERSION 1

LFM3_NPU_EXPORT int lfm3_npu_abi_version(void);
// |device| is an HRX device URI such as "xdna://0?columns=8".
LFM3_NPU_EXPORT int lfm3_npu_open(const char* device, lfm3_npu** out);
LFM3_NPU_EXPORT void lfm3_npu_close(lfm3_npu* npu);
LFM3_NPU_EXPORT const char* lfm3_npu_error(const lfm3_npu* npu);

// Compiles public kernel |root| (for example "@layer") from Loom source text
// and prepares it for dispatch. Returns a kernel id.
LFM3_NPU_EXPORT int lfm3_npu_compile(lfm3_npu* npu, const char* source, size_t source_length,
                     const char* source_name, const char* root, int* out_kernel);

// Device buffers, ids starting at 0.
LFM3_NPU_EXPORT int lfm3_npu_alloc(lfm3_npu* npu, uint64_t bytes, int* out_buffer);
LFM3_NPU_EXPORT int lfm3_npu_free(lfm3_npu* npu, int buffer);
LFM3_NPU_EXPORT int lfm3_npu_write(lfm3_npu* npu, int buffer, uint64_t offset, const void* data,
                   uint64_t bytes);
LFM3_NPU_EXPORT int lfm3_npu_read(lfm3_npu* npu, int buffer, uint64_t offset, void* data,
                  uint64_t bytes);

// Runs |kernel| with whole-buffer bindings in ordinal order and waits.
LFM3_NPU_EXPORT int lfm3_npu_dispatch(lfm3_npu* npu, int kernel, const int* buffers, int count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LFM3_NPU_H_
