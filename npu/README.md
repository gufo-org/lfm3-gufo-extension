# LFM3 on the XDNA NPU

`--npu` runs LFM3 on the Strix Halo XDNA NPU through Gufo's NPU runtime
(`gufo/ext/experimental/npu.hpp`) with our own IRON kernels. It applies to
LFM3-1.7B and LFM3-VL-2B (`serve`, `prompt`, `chat`, `bench`), LFM3-Audio-2B
(`serve asr`, `serve tts`) and d1-Omni-2B (`serve decision`, `decide`).

The GPU stays off for every supported model:

- **Backbone:** runs on the NPU programs below.
- **Towers:** the audio encoder, depth decoder and codec, and the VL vision encoder, do their matrix products on Gufo's `npu::Q8Gemm` (`npu/dense.cpp`, `npu/stack.cpp`).
  - Q8_0 weights keep their codes.
  - Other weights (the BF16 vision projector) are two Q8_0 planes, a code and its residual, about 2^-15 of each block's largest weight.
  - Norms, attention, convolutions, sampling and mel/iSTFT run on the CPU (`npu/host_ops.hpp` and the `npu` paths of `audio/model.cpp` and `vl/vision.cpp`).

```sh
GUFO_SOURCE=~/gufo npu/iron/build.sh /models/LFM3-1.7B/LFM3-1.7B-Q8_0.gguf  # writes LFM3-1.7B-Q8_0.npu
gufo serve --model /models/LFM3-1.7B/LFM3-1.7B-Q8_0.gguf --npu
gufo serve tts --model /models/LFM3-Audio-2B --npu
```

## d1 decision models

Each d1 prompt is one bidirectional prefill from position 0 (`npu/d1.cpp`).

- **Prefill programs:** `prefill/model.py <out> d1 <M>`, for M = 128, 512 and 1024.
  - Full-attention layers see the whole prompt, with question groups: state rows see and are seen by all rows; a question row sees the state rows and its own group.
  - Convolutions and sliding attention stay causal.
  - A per-round meta object carries the prompt length and the groups.
  - Every key block streams, block 0 first: the first state row it holds sets a finite softmax offset for every row.
- **Readout:** the answer rows' final hidden states go through the LM head on `npu::Q8Gemm`.
- **Limits:** prompts of up to 1024 rows; a packed prompt must start with a state row.
- **Quality:** against the f32 reference, the hidden state is within about 4%, as for text prefill. In `tests/functional/d1.py` against the GPU, every case passes the 0.02 probability gate except one audio question, which differs by 0.04.

## Bundle

`--npu` loads `<model>.npu`, the directory beside the GGUF that
`npu/iron/build.sh` writes (about 6 GB). `build.sh` needs an mlir-aie (IRON)
Python with Peano and XRT's `pyxrt`. It compiles the programs once per machine
into `~/.cache/gufo/lfm3-npu`, which takes about 15 minutes, then repacks the
model's weights.

| File | Content |
|---|---|
| `decode.xclbin`, `decode_dispatch.so` | One decode token on all 32 cores (`lfm3_token.py`, `decode.cc`). The dispatch library writes each token's instruction stream. |
| `prefill{,512,128}.xclbin`, `*.insts` | Prefill of up to 1024, 512 or 128 rows (`prefill/model.py`): bf16 × bfp16 GEMMs, norms, convolutions and attention as phases of one dispatch. |
| `decode_*.bin`, `prefill_*.bin` | The repacked Q8_0 weights and the initial state (`bundle.py`). |
| `bundle.txt` | The offsets the runtime needs (`npu/runtime.cpp`). |
| `gemm/` | Gufo's Q8_0 GEMM program (`src/core/npu/kernels/q8gemm`), for the towers and the d1 head. |
| `d1_prefill*` | (d1 GGUFs, instead of the files above) the bidirectional prefill programs, their scratch and weights. |

## Runtime

- `npu/runtime.cpp` prefills a fresh prompt of up to 1024 rows in one dispatch. It then copies the KV caches and convolution states into the decode state and decodes one token per dispatch.
- Each token's decode stream is generated while the previous token runs, then patched with the token id (`gufo::npu::DiffInstructions`).
- `npu/engine.cpp` shares the runtime between sessions. Inputs that need no logits wait for the next pass that needs logits. A long continuation prefills the whole history again when that is cheaper than decoding it row by row.

Measured on LFM3-1.7B Q8_0:

| Workload | Time |
|---|---:|
| Prefill, 1024 tokens | 245 ms |
| Prefill, up to 128 tokens | 62 ms |
| Decode | 27.4 tok/s |

Against the GPU Q8 model, teacher-forced top-5 KL is 3.2e-2, and the top token matches in 163 of 185 positions.

LFM3-Audio-2B with the GPU off (`HIP_VISIBLE_DEVICES=-1`):

| Workload | Time |
|---|---:|
| TTS first audio | 96–108 ms |
| TTS real-time factor | 0.82 |
| ASR, 3.5 s clip | 397 ms (encoder 86 ms) |
| Live transcription, first text | ~430 ms |
| Voice, first audio | ~860 ms |

The limits are about 84 µs per GEMM dispatch, the CPU row operations between dispatches, and a ~3 ms switch between the backbone and GEMM programs at each audio frame.
