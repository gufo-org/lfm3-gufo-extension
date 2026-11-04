# Functional suites

Model-owned HTTP and CLI checks for this extension, from the Gufo `lfm3`
branch (`500df164`) with a reduced harness ([run.py](run.py) keeps only the
owned-server lifecycle and provenance of Gufo's `tests/functional/run.py`).
They need a Gufo built with this extension (extension API 0.1.19 or newer),
real weights and a GPU; missing weights are failures, not skips. Run them
with the Gufo development shell's Python (`nix develop -c python3 ...` in a
Gufo checkout). Each creates a fresh evidence directory.

```sh
B=build/gpu-test/gufo
A=models/lfm3-audio V=models/lfm3-vl D=models/d1
python3 realtime_voice.py --binary $B --model $A/LFM3-Audio-2B-Q8_0.gguf \
  --mmproj $A/mmproj-LFM3-Audio-2B-BF16.gguf --output out/realtime \
  [--browser /path/to/chromium]
python3 lfm3_audio_context.py --binary $B --model $A/LFM3-Audio-2B-Q8_0.gguf \
  --mmproj $A/mmproj-LFM3-Audio-2B-BF16.gguf --output out/context32k
python3 model_services.py asr --binary $B --model $A/LFM3-Audio-2B-Q8_0.gguf \
  --mmproj $A/mmproj-LFM3-Audio-2B-BF16.gguf --output out/asr
python3 model_services.py tts --binary $B --model $A/LFM3-Audio-2B-Q8_0.gguf \
  --mmproj $A/mmproj-LFM3-Audio-2B-BF16.gguf --output out/tts \
  --asr-binary $B --asr-model models/Qwen3-ASR-1.7B
python3 lfm3_vl.py --binary $B --model $V/LFM3-VL-2B-Q8_0.gguf \
  --mmproj $V/mmproj-LFM3-VL-2B-BF16.gguf --output out/vl
# d1 suites talk to already running `gufo serve decision` servers.
python3 d1_packed.py --url http://127.0.0.1:18886 --image cats.jpg \
  --audio fixtures/audio/clear-english-voice.wav --output out/d1-packed.json
python3 d1.py --candidate http://127.0.0.1:18886 \
  --reference http://127.0.0.1:18887 --image cats.jpg \
  --audio fixtures/audio/clear-english-voice.wav --output out/d1.json
```

- `realtime_voice.py`: `/v1/realtime` live transcription, server turn
  detection, spoken and typed conversation turns, reply speech transcribed
  back, a spoken tool call, cancellation, reasoning and rejections; with
  `--browser`, the `/voice` page in headless Chromium.
- `lfm3_audio_context.py`: 32K conversation history, audio after long text,
  eviction and an input near the 32768-slot boundary.
- `model_services.py` with `asr_service.py`, `tts_service.py` and
  `service_http.py`: the transcription and speech APIs (formats, streaming,
  replay, concurrency, rejections, cancellation); TTS intelligibility is
  judged by a separate Qwen3-ASR server.
- `lfm3_vl.py`: image chat over the text APIs, cache replay and restart.
- `d1_packed.py`: packed sets, boxes, audio placement and validation
  (`cats.jpg` must show two cats, for example COCO `val2017/000000039769.jpg`).
  `d1.py`: parity with a reference `/v1/systemone` server (llama.cpp
  `8642541b`, or another Gufo build) within a 0.02 probability gate.

[fixtures/audio](fixtures/audio/README.md) holds the English speech fixture.
