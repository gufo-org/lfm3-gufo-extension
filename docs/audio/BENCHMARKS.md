# LFM3-Audio 2B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Official `Q8_0` backbone
and BF16 projector (`02355c1a`). Gufo `12f8d8ab` with this extension
(November 8, 2026), Nix development build. One process per CLI measurement,
so first requests include lazy kernel loading. Positive gain favors Gufo.
**TODO** means unmeasured: the llama.cpp reference (b12205 or newer, which
serves LFM3-Audio transcription) has not been run yet.

## Single stream

| Workload | Gufo | llama.cpp | Gain |
| --- | ---: | ---: | ---: |
| ASR, 10.0 s English clip, mean of 5 after 1 warmup | 284.7 ms (35.1× real time) | TODO | TODO |
| ASR audio encoder (mel, towers, projector) | 17.2 ms | TODO | TODO |
| TTS, 104-character sentence, seeds 1–3 | 6.47–6.49× real time | N/A | N/A |
| TTS time to first audio (prompt to first 80 ms frame) | 36.3–36.5 ms | N/A | N/A |
| TTS backbone step per frame | 8.59–8.60 ms | N/A | N/A |
| TTS depth decoder per frame (3 MaskGIT passes, sampling) | 3.51–3.52 ms | N/A | N/A |
| Voice chat, 2.24 s spoken question: first text / first audio | 69.0 / 138.4 ms | N/A | N/A |

llama-server has no speech-synthesis endpoint, so the TTS and voice chat
rows have no reference. The voice-chat question is the model's own seed-7
reading of "What is the capital of France?". The interleaved reply starts
with six text tokens, so its first audio follows the first text block.

```sh
gufo transcribe --model $M --mmproj $P --language en --repeat 5 --warmup 1 \
  --format json tests/functional/fixtures/audio/clear-english-voice.wav
gufo speak --model $M --mmproj $P --seed 1 --format json --output tts.wav \
  --text 'The quick brown fox jumps over the lazy dog. Then it runs back home through the forest, happy and tired.'
gufo speak --model $M --mmproj $P --mode chat --audio question.wav \
  --output reply.wav --format json
```

## Concurrent synthesis

Concurrent requests share decoding steps. Same sentence, seeds 100 and up,
after one warmup batch, measured by `lfm3_audio_batch_bench` (`gpu-test`
build). HTTP throughput of `serve tts` is not measured.

| Streams | Aggregate | Per stream | Depth / backbone per step |
| ---: | ---: | ---: | ---: |
| 1 | 6.5× | 6.5× | 3.50 / 8.46 ms |
| 2 | 9.9× | 5.0× | 6.03 / 9.06 ms |
| 4 | 13.9× | 3.5× | 10.84 / 10.53 ms |
| 8 | 20.7× | 2.6× | 13.75 / 13.73 ms |

The backbone's eight-row Q8_0 projections stay near the one-row bandwidth
floor; the rest of its step is per-row attention, RoPE and convolution
launches. The depth decoder becomes compute bound as streams grow, so
batches stop at eight.

## Ceiling

Every speech frame needs one backbone step and then the depth decoder, in
sequence. The Q8_0 backbone reads about 1.55 GB of weights per step at close
to the measured ~240 GB/s large-buffer ceiling, so one step cannot drop much
below 6.6 ms. A single stream is bounded near **10× real time**. Concurrent
streams share steps instead.

Realtime latencies (live transcription, conversation turns, barge-in) are not
measured on this build; [realtime_voice.py](../../tests/functional/realtime_voice.py)
covers their behaviour. See [experiments](EXPERIMENTS.md) for retained and
rejected optimizations.
