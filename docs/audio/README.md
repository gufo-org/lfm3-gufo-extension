# LFM3-Audio 2B

Native speech recognition and speech synthesis for LiquidAI LFM3-Audio-2B on
Linux x86-64 / Strix Halo `gfx1151`, served by the `lfm3-audio-asr` and
`lfm3-audio-tts` providers of this extension. Implementation:
[audio/](../../audio), on top of the [LFM3 backbone](../../README.md).

The checkpoint pairs the LFM3 1.7B backbone with three audio towers stored in
the mmproj GGUF:

- **Encoder** (16 kHz in): log-mel features (128 Slaney bins, 10 ms hop), three
  stride-2 convolutions over the (mel, time) plane, a causal 17-layer LFM2
  stack (short convolutions and 32-token sliding attention) and a projector
  into the backbone. One audio token covers 80 ms.
- **Depth decoder**: a bidirectional 3-layer LFM3 stack over the 12 codebook
  slots of one speech frame. Codebooks are revealed in a random order over
  three MaskGIT passes of four.
- **LiquidCodec decoder** (24 kHz out): FSQ latents, an 8-layer causal LFM3
  stack at 75 Hz with 30-frame sliding attention, and an inverse STFT
  (1280/320). One speech frame is 80 ms (1920 samples).

Each backbone step consumes one text token or one speech frame. Prompts
ending in `<|audio_start|>` speak; `<|mixed_start|>` alternates blocks of six
text tokens and twelve speech frames until `<|text_end|>`, then finishes the
speech. This follows the official `generate` state machine.

## Precision and acquisition

Use the **Q8_0 backbone with the BF16 mmproj**. BF16 is the original
precision of both files. The Q8_0 mmproj is accepted: it speeds up the depth
decoder by about a quarter, but moves codec audio by up to 7% of full scale
against the BF16 towers (see [quality](QUALITY.md)). Activations, attention,
convolution history, the inverse FFT and all accumulation are FP32. Speech
codes are sampled on the device; the frame embedding returned to the backbone
is the exact sum of the codebook embeddings.

```sh
REV=02355c1a141e464a064f2ab2c276638dbcb7f38a
hf download LiquidAI/LFM3-Audio-2B-GGUF \
  LFM3-Audio-2B-Q8_0.gguf mmproj-LFM3-Audio-2B-BF16.gguf \
  --revision $REV --local-dir models/lfm3-audio
```

Weights use the [LFM Open License](https://huggingface.co/LiquidAI/LFM3-Audio-2B/blob/main/LICENSE)
and are downloaded separately.

`--model` names the backbone GGUF, or a directory holding exactly one LFM3
GGUF. The audio towers are the single `mmproj*.gguf` with LFM3-Audio towers
beside it; when there are several, the one named `mmproj-<backbone file
name>` is used and any other choice is an error. The provider API takes no
`--mmproj`, so to pair the Q8_0 backbone with the BF16 mmproj keep only that
mmproj beside it (a directory of symbolic links works).

## Usage

```sh
D=models/lfm3-audio-serve   # LFM3-Audio-2B-Q8_0.gguf + mmproj-LFM3-Audio-2B-BF16.gguf
build/release/gufo serve tts --model $D/LFM3-Audio-2B-Q8_0.gguf \
  --served-model-name lfm3-audio-2b
build/release/gufo serve asr --model $D/LFM3-Audio-2B-Q8_0.gguf \
  --served-model-name lfm3-audio-2b
```

Without `--served-model-name` the services keep their Qwen default names, and
requests must use those as `model`.

`serve tts` exposes `/v1/audio/speech` (WAV, PCM, streamed audio and SSE)
and `/v1/audio/voices` (`female`, `male`);
`language` takes a code (`en`, `en-gb`, `fr`, …) or English name.
Without request sampling the model card's speech sampling is used
(temperature 1.0, top-k 128, repetition penalty 1.1 over 4096 frames) with
the request `seed`; disabling sampling selects the most likely speech codes.
The Qwen-specific sampling fields do not apply to this model. Concurrent
speech requests (up to eight) share decoding steps: every step runs one
backbone pass and one depth-decoder pass for all of them, so their weights
are read once (see [benchmarks](BENCHMARKS.md)); a lone request takes the
single-stream path. Each request samples with its own seed; batched
arithmetic may round differently from a lone request. Reference-voice
cloning is rejected. `serve asr` exposes `/v1/audio/transcriptions` (JSON,
text, verbose JSON, SSE), transcribing recordings in 30 s windows;
`language` pins the transcript language.

Inputs are PCM16/24/32 or float32 WAV, mixed to mono and resampled
(windowed sinc) to 16 kHz. Streaming synthesis emits the first 80 ms frame as
soon as it is decoded, then chunks of four frames; the codec of one chunk runs
while the backbone computes the next frame.

## Not in this extension

The in-tree port on the Gufo `lfm3` branch (`2ac4eb9e`) also had `gufo
transcribe --live`, `gufo speak` (voice chat, text replies, function tools),
`/v1/realtime` WebSocket transcription and conversation sessions with server
turn detection, and the `/voice` page. Those surfaces live in Gufo's CLI and
server and have no extension API, so they are not provided here; the live
transcriber stays in [generation.hpp](../../audio/generation.hpp). Also not
supported: the Omni and VL checkpoints (separate models) and Q4 backbones.
Digital silence transcribes as a short word ("Okay."), as in the FP32
reference.

[Quality](QUALITY.md) · [Benchmarks](BENCHMARKS.md) · [Experiments](EXPERIMENTS.md)
