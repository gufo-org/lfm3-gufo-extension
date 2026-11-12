#include "flm_lfm3_shim.h"

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "lm_config.hpp"
#include "models/lfm3/flm/aie2p/lfm3_npu.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"

namespace {
thread_local std::string last_error;
float BfToFloat(bf16 v) {
  std::uint16_t h;
  std::memcpy(&h, &v, 2);
  const std::uint32_t b = std::uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
}  // namespace

struct flm_lfm3 {
  xrt::device device{0};
  std::unique_ptr<npu_xclbin_manager> npu;
  std::unique_ptr<LM_Config> config;
  std::unique_ptr<lfm3_npu> engine;
  int vocab = 0;
  void Copy(buffer<bf16>& out, float* logits) const {
    for (int i = 0; i < vocab; ++i) logits[i] = BfToFloat(out[i]);
  }
};

extern "C" {

flm_lfm3* flm_lfm3_create(const char* model_dir, const char* asset_root, int max_length) {
  try {
    auto m = std::make_unique<flm_lfm3>();
    m->npu = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, &m->device, false);
    m->config = std::make_unique<LM_Config>();
    m->config->from_pretrained(model_dir, asset_root);
    m->vocab = m->config->get<int>("vocab_size");
    Q4NX q4nx(model_dir);
    m->engine = std::make_unique<lfm3_npu>(*m->config, m->npu.get(), max_length);
    m->engine->load_weights(q4nx);
    m->engine->clear_context();
    return m.release();
  } catch (const std::exception& e) {
    last_error = e.what();
    return nullptr;
  }
}

void flm_lfm3_destroy(flm_lfm3* m) { delete m; }
const char* flm_lfm3_last_error(void) { return last_error.c_str(); }
int flm_lfm3_vocab(const flm_lfm3* m) { return m->vocab; }

int flm_lfm3_prefill(flm_lfm3* m, const int32_t* ids, int count, float* logits) {
  try {
    std::vector<int> v(ids, ids + count);
    buffer<bf16> out = m->engine->prefill(v);
    if (logits) m->Copy(out, logits);
    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}

int flm_lfm3_decode(flm_lfm3* m, int32_t id, float* logits) {
  try {
    buffer<bf16> out = m->engine->forward(id);
    if (logits) m->Copy(out, logits);
    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}

int flm_lfm3_length(flm_lfm3* m) { return m->engine->get_current_context_length(); }
void flm_lfm3_clear(flm_lfm3* m) { m->engine->clear_context(); }
}
