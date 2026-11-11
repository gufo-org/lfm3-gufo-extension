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

`-P` chains dependent GEMVs in one XRT run. Each phase starts only after the
previous phase's outputs drained to DDR, as a decode layer's next projection
would. Each phase is a 2048 × 2048 GEMV:

| Phases in one run | Time | Per phase | Pure streaming |
| ---: | ---: | ---: | ---: |
| 1 | 113 µs | 113 µs | 87 µs |
| 24 | 2,229 µs | 92.9 µs | 87 µs |

So a phase boundary (every core synchronized, plus a DDR round trip) costs
about 6 µs. A whole decode token as about 100 dependent phases in one run
would add about 0.6 ms to its 36 ms of weight streaming.

```sh
python iron/gemv.py -M 8192 -K 2048 [-P 24] --xclbin-path gemv.xclbin --insts-path gemv.bin
host/gemv_bench $PWD/gemv.xclbin $PWD/gemv.bin 8192 2048 8 [24]
```

### FFN block

`iron/ffn.py` runs one LFM3 decode FFN block in one XRT run, on all 32 cores,
in two phases:

- **F1.** Each core computes the post-norm residual `h += rmsnorm(o) * w_post`,
  then `rmsnorm(h) * w_ffn`, quantized to Q8_0. It then runs its 240
  gate/up channels and writes `SiLU(gate) * up` as bf16.
- **F2.** Each core quantizes the 7,680 activations and runs its 64 down rows.

Every vector stays in natural order in DDR; the host packs the weight
records to match. Both shim MM2S channels of every column stream weights, and
each phase's input rides at the head of the weight streams.

`iron/ffn_host.py` (pyxrt) checks the block against float64 math on the
GGUF's own Q8_0 weights. On layer 0 of LFM3-1.7B:

| | Relative error |
| --- | ---: |
| Residual after the post-norm | 2.6e-8 |
| SwiGLU activations | 8.3e-3 |
| Block output after its post-norm | 1.2e-2 |

The error comes from quantizing activations to Q8_0, as llama.cpp's CPU path
also does. The block takes 1,242 µs per run, including the host's wait: 41 GB/s
over its 51 MB of weights. With the dot products stubbed out it takes 1,168 µs,
so the streaming side, not the arithmetic, sets the time.

### Convolution layer

`iron/decode.py` and `iron/decode.cc` run a whole LFM3 convolution layer
(operator norm, B/C/x projection, causal short convolution, out projection,
then the FFN block) in one XRT run, in four dependent phases. The convolution
state goes to and from DDR each token. The kernels use vector arithmetic
only (SiLU through the hardware tanh, `aie::invsqrt`, `aie::inv`), so no
soft-float routines are linked.

`iron/decode_host.py` checks two consecutive tokens of layer 0 against
float64 model math on the Q8_0 GGUF:

| Token | Convolution output | Residual | FFN output |
| ---: | ---: | ---: | ---: |
| 0 | 4.5e-3 | 3.7e-3 | 9.5e-3 |
| 1 | 7.8e-3 | 4.9e-3 | 1.3e-2 |

A layer takes 1,617 µs per run over 73.5 MB of records (45.5 GB/s).

### Whole decode token

`iron/lfm3_token.py` runs an entire LFM3-1.7B decode token in one XRT run:

- **Phases.** Embedding, 13 convolution layers (4 phases each), 11 attention
  layers (5 phases each) and the LM head: 109 phases.
- **Interpreter.** Every core runs a phase interpreter (`decode.cc`,
  `*_next`). The C++ generates the phase sequence, so the core program stays
  within its 16 KB.
- **Roles.** Rows 2–3 also attend (one head each, a two-pass-free online
  softmax over a bf16 KV cache). Rows 4–5 also convolve (128 channels each).
  Each role links its own kernel object.
- **Dispatch-time values.** The token, the position and the attention spans
  are `DispatchTime` values. The host builds each token's instructions with
  the design's dispatch library (`dispatch_generate`).

`iron/layout.py` holds the buffer layouts for the design and the host. Both
buffers stay under 2^31 bytes, because tap offsets are 32-bit. The f32
embedding table sits in the state buffer.

`iron/token_host.py` packs the Q8_0 GGUF into weight records, runs tokens and
compares the logits with a float32 model on the same dequantized weights.
LFM3-1.7B, positions 0–7:

| | |
| --- | ---: |
| Time per token | 42.7 ms (23.4 tok/s) |
| Top-1 agreement | 8 of 8 |
| Full-vocabulary KL | 9e-3 to 2e-2 (6.7e-2 at position 0) |
| Logit relative error | 6e-2 to 1e-1 |
