// Scratch probe: how many workgroups of a small grid run in parallel. Each
// workgroup does a fixed, bounded amount of ALU work; if a grid of G
// workgroups takes as long as one, all G ran on separate compute units. Also
// records each workgroup's hardware ID (shader engine, CU) via s_getreg.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <vector>

#define OK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1);} } while (0)

__global__ void Work(int iters, unsigned* sink, unsigned* hw) {
  unsigned a = threadIdx.x ^ blockIdx.x;
  for (int i = 0; i < iters; ++i) a = a * 1664525u + 1013904223u;
  if (a == 0x9e3779b9u) sink[0] = a;
  if (threadIdx.x == 0) {
    // HW_REG_HW_ID1 on gfx11: wave, SIMD, WGP, SA and SE identifiers.
    hw[blockIdx.x] = __builtin_amdgcn_s_getreg((23 << 0) | (0 << 6) | (31 << 11));
  }
}

int main() {
  unsigned *sink, *hw; OK(hipMalloc(&sink, 4)); OK(hipMalloc(&hw, 4096 * 4));
  hipEvent_t a, b; OK(hipEventCreate(&a)); OK(hipEventCreate(&b));
  const int iters = 200000;
  float base = 0;
  for (int grid : {1, 2, 4, 8, 10, 16, 20, 32, 40, 80}) {
    Work<<<grid, 256>>>(iters, sink, hw); OK(hipDeviceSynchronize());
    std::vector<float> v;
    for (int r = 0; r < 7; ++r) {
      OK(hipEventRecord(a, nullptr)); Work<<<grid, 256>>>(iters, sink, hw);
      OK(hipEventRecord(b, nullptr)); OK(hipEventSynchronize(b));
      float t; OK(hipEventElapsedTime(&t, a, b)); v.push_back(t);
    }
    std::sort(v.begin(), v.end());
    if (grid == 1) base = v[3];
    std::vector<unsigned> ids(grid);
    OK(hipMemcpy(ids.data(), hw, grid * 4, hipMemcpyDeviceToHost));
    std::set<unsigned> wgps, ses;
    for (unsigned id : ids) {
      wgps.insert((id >> 10) & 0xff);  // WGP, SA, SE bits
      ses.insert((id >> 18) & 0x7);
    }
    std::printf("grid %3d: %7.3f ms (%.2fx one WG)  distinct WGP/SA/SE ids %zu, SE ids %zu\n",
                grid, v[3], v[3] / base, wgps.size(), ses.size());
  }
  return 0;
}
