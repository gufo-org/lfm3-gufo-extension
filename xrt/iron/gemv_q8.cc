// y[r] = sum_b dot(w[r, b*32 : b*32+32], x[b*32 : ...]) * sw[r, b] * sx[b]
// for TM rows of one weight record. Codes are int8; scales are bf16 stored
// after the codes (x: K codes then K/32 scales; w: TM*K codes then TM*K/32).
#include <stdint.h>
#include <aie_api/aie.hpp>
#ifndef GEMV_K
#define GEMV_K 2048
#endif
#ifndef GEMV_TM
#define GEMV_TM 8
#endif
constexpr int K = GEMV_K, TM = GEMV_TM;
extern "C" void gemv_q8(const int8_t* __restrict x, const int8_t* __restrict w, float* __restrict y) {
  const bfloat16* xs = reinterpret_cast<const bfloat16*>(x + K);
  const bfloat16* ws = reinterpret_cast<const bfloat16*>(w + TM * K);
  constexpr int NB = K / 32;
  for (int r = 0; r < TM; ++r) chess_prepare_for_pipelining {
    aie::accum<accfloat, 16> acc = aie::zeros<accfloat, 16>();
    for (int b0 = 0; b0 < NB; b0 += 16) {
      alignas(64) int32_t sums[16];
      for (int j = 0; j < 16; ++j) {
        const int b = b0 + j;
        aie::vector<int8, 32> a = aie::load_v<32>(w + r * K + b * 32);
        aie::vector<int8, 32> v = aie::load_v<32>(x + b * 32);
        aie::accum<acc32, 32> m = aie::mul(a, v);
        sums[j] = aie::reduce_add(m.to_vector<int32>(0));
      }
      aie::vector<int32, 16> s = aie::load_v<16>(sums);
      aie::vector<float, 16> sf = aie::to_float(s, 0);  // exact below 2^24
      aie::vector<bfloat16, 16> sw = aie::load_v<16>(ws + r * NB + b0);
      aie::vector<bfloat16, 16> sx = aie::load_v<16>(xs + b0);
      aie::vector<float, 16> sc = aie::mul(sw, sx).to_vector<float>();  // exact
      acc = aie::mac(acc, sf, sc);
    }
    y[r] = aie::reduce_add(acc.to_vector<float>());
  }
}
