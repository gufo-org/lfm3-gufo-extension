# LFM3 on the NPU through XRT

`GUFO_LFM3_DEVICE=xrt` runs the whole LFM3 text model on the Strix Halo
XDNA NPU through AMD's XRT runtime instead of HRX. The model runs on
FastFlowLM's prebuilt LFM3 engine (`src/lib/xrt/liblfm3_npu.so`, under
FastFlowLM's binary license: free to use and redistribute unmodified). A small
C shim (`flm_lfm3_shim.h`) wraps the engine, and Gufo loads the shim with
`dlopen`.

## Status

| LFM3-1.7B, Strix Halo NPU | Prefill, 1,010 tokens | Prefill, 2,110 tokens | Decode |
| --- | ---: | ---: | ---: |
| Gufo, `GUFO_LFM3_DEVICE=xrt` | 0.96 s | 1.93 s | 38.9–40.7 tok/s |
| FastFlowLM's server, same engine | 0.79 s | 1.68 s | 41.5–43.5 tok/s |
| Gufo, HRX path (`npu`) | — | 3.15 s | 17–18.6 tok/s |
| Halogen, 8-bit | about 0.27 s | — | 27.3 tok/s |
| Halogen, 5-bit | about 0.28 s | — | 38.3 tok/s |

The engine runs FastFlowLM's Q4_0 weights (`model.q4nx`). Their quality falls
short of our bar on the 185-position teacher-forced test against the FP32 GPU
reference:

| Backend | Top-5 KL mean | Top-1 agreement |
| --- | ---: | ---: |
| GPU, Q8_0 | 2.09e-3 | 183/185 |
| NPU, HRX path | 1.14e-2 | 175/185 |
| NPU, XRT engine (Q4_0) | 2.42e-1 | 128/185 |

XRT sessions take tokens only. They keep no continuation rows, so the server
caches no prefixes. They also take no embedding rows, so the audio and vision
inputs are not supported yet.

## Build

1. Build XRT and the amdxdna plugin as a matched pair (xdna-driver `main` and
   the XRT commit it pins). The derivations come from robcohen/nixpkgs
   (`vitis-ai` branch); the plugin maps its device heap with `MAP_POPULATE`
   instead of `MAP_LOCKED`, so no memlock limit is needed:

   ```sh
   GUFO_FLAKE=path:$HOME/gufo nix build --impure -f xrt/nix/default.nix combined -o xrt-amdxdna
   XILINX_XRT=$PWD/xrt-amdxdna/opt/xilinx/xrt xrt-amdxdna/opt/xilinx/xrt/bin/unwrapped/xrt-smi examine
   ```

2. Clone FastFlowLM (with its prebuilt engines and xclbins) and build the shim:

   ```sh
   git clone --depth 1 https://github.com/ROCm/FastFlowLM.git
   nix-shell -p gcc libuuid libdrm pkg-config --run \
     "xrt/build_shim.sh FastFlowLM xrt-amdxdna/opt/xilinx/xrt out"
   ```

3. Fetch the engine's weights from `huggingface.co/FastFlowLM/LFM3-1.7B-NPU2`
   (`config.json`, `model.q4nx`; LFM Open License v1.0).

## Run

```sh
GUFO_LFM3_DEVICE=xrt \
LFM3_XRT_LIBRARY=out/libflm_lfm3_shim.so \
LFM3_XRT_MODEL=/models/LFM3-1.7B-NPU2 \
LFM3_XRT_ASSETS=FastFlowLM/src \
XILINX_XRT=xrt-amdxdna/opt/xilinx/xrt \
gufo serve llm --model LFM3-1.7B-Q8_0.gguf
```

The GGUF supplies the tokenizer and chat template; the NPU engine uses the
Q4NX weights.

`out/flm_lfm3_bench <model-dir> <assets> <ids.txt> [tokens]` times the engine
alone on a file of token ids.

## Own kernels (mlir-aie)

`iron/gemv.py` is a Q8_0 matrix-vector product for decode, written with
mlir-aie's IRON API and compiled by Peano. It uses llama.cpp's Q8_0 semantics:
int8 weights and activations in 32-element blocks with per-block scales, and
F32 accumulation. All 32 cores take part. Each column streams its four cores'
weight records through one shim channel into the memory tile, which splits
them.

| 8192 × 2048 Q8_0 GEMV | Time | Weight bandwidth |
| --- | ---: | ---: |
| One run, waited | 389 µs | 45.8 GB/s |
| Eight runs queued | 349 µs | 51.1 GB/s |

The results match a double-precision reference to within 1.1e-3 relative.
51 GB/s is the NPU's DRAM ceiling. At that rate LFM3-1.7B's Q8 weights
(about 1.83 GB per token) take about 36 ms, which is Halogen's 27.3 tok/s.

```sh
python iron/gemv.py -M 8192 -K 2048 --xclbin-path gemv.xclbin --insts-path gemv.bin
host/gemv_bench $PWD/gemv.xclbin $PWD/gemv.bin 8192 2048 8
```
