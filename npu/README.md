# LFM3 on the XDNA NPU

`--npu` runs the LFM3 backbone on the Strix Halo XDNA NPU through Gufo's NPU
runtime (`gufo/ext/experimental/npu.hpp`) with our own IRON kernels. It applies
to LFM3-1.7B and LFM3-VL-2B (`serve`, `prompt`, `chat`, `bench`) and to the
LFM3-Audio-2B backbone (`serve asr`, `serve tts`). Vision and audio towers stay
on the GPU. The NPU runs chat models only, not d1 decision models.

```sh
npu/iron/build.sh /models/LFM3-1.7B/LFM3-1.7B-Q8_0.gguf  # writes LFM3-1.7B-Q8_0.npu
gufo serve --model /models/LFM3-1.7B/LFM3-1.7B-Q8_0.gguf --npu
gufo serve tts --model /models/LFM3-Audio-2B --npu
```

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
