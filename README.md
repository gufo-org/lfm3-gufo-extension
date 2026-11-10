# LFM3 1.7B

Native text inference for LiquidAI LFM3-1.7B on Linux x86-64 / Strix Halo
`gfx1151`. The same extension serves the other checkpoints built on the LFM3
backbone:

- LFM3-VL-2B text and still-image input, through the same `lfm3` text
  provider ([LFM3-VL](docs/vl/README.md)).
- LFM3-Audio-2B speech recognition, synthesis, live transcription, realtime
  spoken conversations and `gufo speak` ([LFM3-Audio](docs/audio/README.md)).
- d1-Omni-2B decisions over text, images and audio, through `gufo decide` and
  `gufo serve decision` ([d1](docs/d1/README.md)).

The extension needs a Gufo with extension API 0.2.20 or newer (Gufo
[#1110](https://github.com/gufo-org/gufo/pull/1110), shared WAV decoding).
Functional suites are in [tests/functional](tests/functional/README.md).

The 24-layer backbone mixes 13 gated width-seven convolutions, five sliding
attention layers (256 tokens) and six global attention layers. Attention has
Q/K RMSNorm, a sigmoid gate per head, and full-head paired RoPE with masked
frequencies on global layers. Both operator and FFN branches have sandwich
norms. Embedding and vocabulary projection share weights.

## Precision and acquisition

Use **Q8_0 for deployment** and **BF16 for numerical reference**. BF16 is the
original checkpoint precision. F16 is accepted for compatibility; it offers no
weight-memory advantage over BF16. Q4/Q5/Q6, multimodal and decision variants
are not supported by this initial backend.

Weights stay packed on the GPU. Activations, accumulation and convolution
history remain FP32 outside the matrix cores; attention K/V caches are F16,
llama.cpp's default (the checkpoint itself is BF16). Q8 prompt projections
scale each activation row by a power of two and round it to FP16 (11
significant bits per element, far below the Q8_0 weight rounding), with one
FP32 WMMA accumulator per Q8 K32 block. Against the BF16 reference this
matches the earlier 22-bit split activations in KL, top-1 and NLL
([quality](docs/QUALITY.md)). Decode keeps FP32 activations in the Strix Halo
quantized GEMV kernels. BF16 prefill shares weights across rows using the
existing exact BF16/FP32 projection kernels. Prompt attention runs on the
matrix cores: F16 Q, K, V and probabilities, FP32 scores, softmax and
accumulation. Decode attention reads the F16 cache with FP32 Q, probabilities
and accumulation. Cache-free attention (vision encoders,
audio stacks, decision prompts) reads the same F16 K/V rows.

Download from the [official GGUF repository](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf):

```sh
hf download LiquidAI/LFM3-1.7B-GGUF LFM3-1.7B-Q8_0.gguf \
  --revision b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf \
  --local-dir models/lfm3
```

Replace the filename with `LFM3-1.7B-BF16.gguf` for reference weights. Model
weights use the [LFM Open License](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE)
and are downloaded separately.

## Build with Gufo

Pass this repository to Gufo via `GUFO_EXTENSION_DIRS` (CMake) or `override { extensions = [ ... ]; }` (Nix).

### From source (CMake)

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/path/to/lfm3-gufo-extension
cmake --build --preset release --parallel 4
./build/release/gufo diagnose --section extensions
```

### With Nix

```sh
nix build --impure --expr '
  let flake = builtins.getFlake (toString ./.);
  in flake.packages.x86_64-linux.default.override {
    extensions = [ /path/to/lfm3-gufo-extension ];
  }
'
./result/bin/gufo diagnose --section extensions
```

Or in a flake-based configuration:

```nix
gufo.packages.x86_64-linux.default.override {
  extensions = [ inputs.lfm3-gufo-extension ];
}
```

## Usage

```sh
build/release/gufo prompt --model models/lfm3/LFM3-1.7B-Q8_0.gguf \
  --prompt 'What is the capital of France?' --think off
build/release/gufo chat --model models/lfm3/LFM3-1.7B-Q8_0.gguf --think off
build/release/gufo bench --model models/lfm3/LFM3-1.7B-Q8_0.gguf \
  -p 2048 -n 128 -r 3
build/release/gufo serve --sessions 2 llm \
  --model models/lfm3/LFM3-1.7B-Q8_0.gguf --context 4096
```

The CLI reserves 8192 tokens; HTTP accepts `--context` up to the native 262144.
The native limit is a loader limit, not a qualification of maximum-depth quality
or speed. HTTP supports Chat Completions, Responses and Messages, custom stops,
structured JSON and native tool calls. Sessions have independent FP32 state;
the scheduler batches up to eight autoregressive decode rows. Beside decoding
sessions, a prompt advances in the engine's 512-token passes, one decode round
between passes; the scheduling profile asks for no layer split, since one such
pass takes about 0.1 s at depth zero. RAM checkpoints,
prefix reuse, cancellation recovery and restart-safe disk checkpoints use the
component cache: full-attention rows are shared between checkpoints, while
convolution state and sliding windows are private. Speculative decoding and image inputs remain
unsupported. CLI chat retains its live prefix when framing permits.

The full release API matrix still fails: several prompts produce unfinished
reasoning or fail exact-answer/tool expectations. See [quality](docs/QUALITY.md) for
retained failures and the bounded checks that pass; API wiring is not a claim
that this checkpoint meets the release quality bar.

Thinking defaults on; use `--think off` or the HTTP
`chat_template_kwargs.enable_thinking=false` control to disable it. Defaults are
temperature 0.1, top-k 50 and repetition penalty 1.1; explicit sampling controls
override them. Historical assistant thoughts are stripped as in the official
text-only template. The tokenizer uses LFM2's Unicode split and byte-level BPE,
including three-digit number groups.

[Quality](docs/QUALITY.md) · [Benchmarks](docs/BENCHMARKS.md) · [Experiments](docs/EXPERIMENTS.md) · [LFM3-VL](docs/vl/README.md) · [LFM3-Audio](docs/audio/README.md) · [d1](docs/d1/README.md)

## License

The extension source is released under the [MIT License](LICENSE). Model
weights are not part of this repository; each model keeps its own license
(the LFM3 family uses the [LFM Open License](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/main/LICENSE)).
