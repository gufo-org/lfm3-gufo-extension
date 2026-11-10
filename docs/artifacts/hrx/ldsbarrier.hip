// Scratch probe: a memory-streaming kernel (one wave per row, like a GEMV)
// with and without a tiny static __shared__ array and a __syncthreads().
// Separates the LDS allocation from the barrier on each runtime.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

template <bool kShared, bool kBarrier>
__launch_bounds__(128, 4) __global__ void Stream(const float4* __restrict__ src, float* out,
                                                 std::size_t row_vec) {
  const std::size_t wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const std::size_t row = blockIdx.x * 4 + wave;
  float4 acc{0, 0, 0, 0};
  for (std::size_t i = lane; i < row_vec; i += 32) {
    const float4 v = src[row * row_vec + i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  float s = acc.x + acc.y + acc.z + acc.w;
  for (int off = 16; off > 0; off >>= 1) s += __shfl_xor(s, off);
  if constexpr (kShared) {
    __shared__ float part[4];
    if (!lane) part[wave] = s;
    if constexpr (kBarrier) __syncthreads();
    if (!lane && wave == 0) out[blockIdx.x] = part[0] + part[1] + part[2] + part[3];
  } else {
    if constexpr (kBarrier) __syncthreads();
    if (!lane) out[row] = s;
  }
}

template <typename F>
static float Median(F launch) {
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  for (int i = 0; i < 3; ++i) launch();
  OK(hipDeviceSynchronize());
  std::vector<float> v;
  for (int r = 0; r < 15; ++r) {
    OK(hipEventRecord(a, nullptr));
    for (int i = 0; i < 10; ++i) launch();
    OK(hipEventRecord(b, nullptr)); OK(hipEventSynchronize(b));
    float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t * 1000 / 10);
  }
  std::sort(v.begin(), v.end());
  return v[7];
}

int main() {
  const std::size_t rows = 16384, row_vec = 2048 / 4;  // 16384 rows x 8 KiB = 128 MiB
  float4* src; OK(hipMalloc(&src, rows * row_vec * 16)); OK(hipMemset(src, 0, rows * row_vec * 16));
  float* out; OK(hipMalloc(&out, rows * 4));
  const int blocks = int(rows / 4);
  const double bytes = double(rows) * row_vec * 16;
  auto run = [&](const char* name, auto kernel) {
    const float t = Median([&] { kernel<<<blocks, 128>>>(src, out, row_vec); });
    std::printf("%-22s %8.1f us  %5.0f GB/s\n", name, t, bytes / t / 1e3);
  };
  run("plain", Stream<false, false>);
  run("__shared__ only", Stream<true, false>);
  run("__syncthreads only", Stream<false, true>);
  run("__shared__ + barrier", Stream<true, true>);
  return 0;
}
