// One-core check of decode.cc's Dot4: in = [x codes 2048 | x scales bf16 64 | pad 128 | record 8704].
#include "../decode.cc"
extern "C" void dot4_test(const int8_t* in, float* out) {
  alignas(64) float r[16];
  Dot4(in, in + 2048 + 128 + 128, r);
  for (int i = 0; i < 4; ++i) out[i] = r[i];
  // the raw block sums of block 0..3 for debugging
  for (int i = 0; i < 12; ++i) out[4 + i] = float(dot4_sums[i]);
}
