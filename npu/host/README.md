# NPU host runtime

`liblfm3_npu.so` compiles Loom programs for the XDNA NPU in-process through HRX
and runs them on device buffers ([`lfm3_npu.h`](lfm3_npu.h)). The extension
loads it at run time ([`npu_runtime.hpp`](npu_runtime.hpp)), so its build does
not depend on HRX.

Build against a configured and built hrx-system tree that contains
`loom-test` (the library links the same objects); qualified with Gufo's fork
pin `gufo-org/hrx-system` `d8345774e3`:

```sh
npu/host/build.sh <hrx-system> <hrx-build> <output-dir>
```

Smoke test (eight-column BF16 GEMV against a CPU reference):

```sh
c++ -std=c++20 -O2 npu/host/npu_smoke.cpp npu/host/npu_runtime.cpp -ldl -o npu_smoke
LFM3_NPU_LIBRARY=<output-dir>/liblfm3_npu.so ./npu_smoke npu/npu_gemv.loom
```

## Program cache

Compiling one Loom root takes seconds, so `lfm3_npu_compile` stores each
compiled executable on disk and loads it on later calls. Entries are keyed by
a hash of the library file, the device URI, the source text and the root, so a
rebuilt library or a changed kernel misses instead of loading a stale program.

- `LFM3_NPU_CACHE_DIR` selects the directory; `off` disables the cache.
- The default is `$XDG_CACHE_HOME/lfm3-npu`, or `~/.cache/lfm3-npu`.
- Entries are written to a temporary file and renamed, so concurrent processes
  share the directory safely. Deleting it only costs recompilation.
