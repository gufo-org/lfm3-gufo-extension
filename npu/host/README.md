# NPU host runtime

`liblfm3_npu.so` compiles Loom programs for the XDNA NPU in-process through HRX
and runs them on device buffers ([`lfm3_npu.h`](lfm3_npu.h)). The extension
loads it at run time ([`npu_runtime.hpp`](npu_runtime.hpp)), so its build does
not depend on HRX.

Build against a configured and built hrx-system tree that contains
`loom-test` (the library links the same objects):

```sh
npu/host/build.sh <hrx-system> <hrx-build> <output-dir>
```

Smoke test (eight-column BF16 GEMV against a CPU reference):

```sh
c++ -std=c++20 -O2 npu/host/npu_smoke.cpp npu/host/npu_runtime.cpp -ldl -o npu_smoke
LFM3_NPU_LIBRARY=<output-dir>/liblfm3_npu.so ./npu_smoke npu/npu_gemv.loom
```
