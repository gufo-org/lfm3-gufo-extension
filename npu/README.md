# LFM3 on the XDNA NPU

`--npu` runs LFM3 on the Strix Halo XDNA NPU with Gufo's installed NPU
programs (`gufo/ext/experimental/npu.hpp`). It needs nothing beside the GGUF:
no build step and no per-model bundle. It applies to LFM3-1.7B and LFM3-VL-2B
(`serve`, `prompt`, `chat`, `bench`), LFM3-Audio-2B (`serve asr`,
`serve tts`) and d1-Omni-2B (`serve decision`, `decide`).

```sh
gufo serve --model /models/LFM3-1.7B/LFM3-1.7B-Q8_0.gguf --npu
gufo serve tts --model /models/LFM3-Audio-2B --npu
```

The GPU stays off for every supported model. Every matrix product runs on the
NPU; norms, short convolutions, rotary embeddings, attention, sampling and
mel/iSTFT run on the CPU.

## Backbone

`npu/backbone.cpp` reads the GGUF's weights at load time (about 1.2 s with a
warm page cache) into two of Gufo's programs:

- **`Q8Gemm`** (`q8gemm`): the Q8_0 codes as they are. It runs products of up
  to 16 rows: decode, short continuations and the LM head. A layer is four
  dispatches: the convolution's or attention's input projection (q, k, v and
  the gate stacked), its output projection, the FFN's gate and up (stacked),
  and down.
- **`BfpGemm`** (`bfpgemm`): bfp16 copies of the same matrices, for prompts
  of more than 16 rows. A prompt of more than 512 rows splits into chunks of
  about 512 rows, each on its own thread one step behind the previous one, so
  one chunk's CPU work overlaps another's NPU products. A chunk's convolution
  and attention wait for the previous chunk's state and keys of that layer.

The state is one sequence: convolution histories and F32 keys and values. A
prompt or continuation of any length prefills at its position. `npu/engine.cpp`
shares the backbone between sessions; a session whose history is not resident
replays it.

d1 prompts are one forward pass from position 0 (`npu/d1.cpp`). Full
attention is bidirectional with question groups: a state row sees and is seen
by every row; a question row sees the state rows and its own group
(`npu/grouped_attention.hpp`). Convolutions and sliding attention stay causal.
The answer rows' final hidden states go through the LM head.

## Towers

The audio encoder, depth decoder and codec, and the VL vision encoder, do
their matrix products on `Q8Gemm` (`npu/dense.cpp`, `npu/stack.cpp`):

- Q8_0 weights keep their codes.
- Other weights (the BF16 vision projector) are two Q8_0 planes, a code and
  its residual, about 2^-15 of each block's largest weight.

## Performance

As of November 2026, on Strix Halo with Gufo's NPU programs:

| Workload | Time |
| --- | ---: |
| LFM3-1.7B Q8_0, prefill of 1024 tokens | 255 ms (4021 tok/s) |
| LFM3-1.7B Q8_0, decode | 22.4 tok/s |
| LFM3-Audio-2B, ASR of a 3.5 s clip | 494 ms |
| LFM3-Audio-2B, TTS first audio | 93–110 ms |
| LFM3-Audio-2B, TTS real-time factor | 0.83 |

Against the GPU Q8_0 model, teacher-forced top-5 KL over 185 positions is
8.2e-3, and the top token matches in 180. d1 passes every case of
`tests/functional/d1.py` against the GPU within the 0.02 probability gate.

`GUFO_LFM3_NPU_TIMING=1` prints the upload time and, per forward pass, its
products, attention and convolution time.
