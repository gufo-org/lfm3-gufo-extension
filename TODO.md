# TODO: Expand `lfm3-gufo-extension` with LFM3-Audio Providers (Option B)

This guide explains how to expand `lfm3-gufo-extension` in-place using **Option B** (multiple providers declared in the single top-level `extension.json`), porting the remaining LFM3-Audio models (`speech_recognition` and `speech_synthesis`) from the `lfm3` branch in the main `gufo` repository.

---

## 1. Source Material on the `lfm3` Branch

In `/home/mixer/gufo`, the `lfm3` branch (`commit 2ac4eb9e`) contains the complete `lfm3_audio` implementation layered on top of `gufo_lfm3`:

- **Model & HIP Kernels (`src/models/lfm3_audio/`)**:
  - `weights.hpp`, `weights.cpp`
  - `audio.cpp`
  - `model.hpp`, `model.cpp`
  - `generation.hpp`, `generation.cpp`
  - `hip/ops.hpp`, `hip/ops.hip`
- **Backbone Additions (`src/models/lfm3/`)**:
  - Inspect `jj diff -r c888fe1a..2ac4eb9e src/models/lfm3/` for any helper methods added to `engine.hpp` / `engine.cpp` and `tokenizer.hpp` / `tokenizer.cpp` for audio token injection, streaming audio input, and batched depth-decoder steps.
- **Serving Runner (`src/cli/serve/`)**:
  - `lfm3_audio_runner.hpp`, `lfm3_audio_runner.cpp` (`FindLfm3AudioMmproj`, `Lfm3AudioRuntime::Transcribe`, `Lfm3AudioRuntime::Synthesize`).
- **Tests & Docs**:
  - `tests/models/lfm3_audio/probe.cpp`, `tests/models/lfm3_audio/audio_test.cpp`, `tests/models/lfm3_audio/batch_bench.cpp`
  - `docs/models/lfm3-audio/{README.md,QUALITY.md,BENCHMARKS.md,EXPERIMENTS.md}`

Extract files from `lfm3` with:
```sh
jj file show -r lfm3 <path>
```

---

## 2. How Option B Works (`extension.json`)

A single `extension.json` can register multiple providers across different capabilities (`text_generation`, `speech_recognition`, `speech_synthesis`) and declare model-specific CLI options (`--mmproj`) if needed.

### Target `extension.json` Layout

```json
{
  "schema": 1,
  "name": "lfm3",
  "api_version": ">= 0.1.8",
  "version": "0.1.0",
  "depends": [],
  "platforms": ["linux"],
  "entry": "lfm3_register",
  "registration_target": "gufo_lfm3",
  "targets": {
    "gufo_lfm3": { "requires": [] },
    "gufo_lfm3_hip": { "requires": ["hip"] },
    "lfm3_config_test": { "requires": ["testing"], "role": "test" },
    "lfm3_ops_test": { "requires": ["hip", "testing"], "role": "test" },
    "lfm3_session_test": { "requires": ["hip", "testing"], "role": "test" },
    "lfm3_probe": { "requires": ["testing"], "role": "test" },
    "lfm3_audio_test": { "requires": ["hip", "testing"], "role": "test" },
    "lfm3_audio_probe": { "requires": ["hip", "testing"], "role": "test" }
  },
  "providers": [
    {
      "id": "lfm3",
      "capability": "text_generation",
      "keys": ["gguf:lfm3"],
      "model_ids": ["LFM3-1.7B", "lfm3-1.7b"],
      "target": "gufo_lfm3_hip",
      "entry": "lfm3_provider_register"
    },
    {
      "id": "lfm3-audio-asr",
      "capability": "speech_recognition",
      "keys": ["gguf:lfm3-audio-asr"],
      "model_ids": ["LFM3-Audio-1.5B", "lfm3-audio-1.5b"],
      "target": "gufo_lfm3_hip",
      "entry": "lfm3_asr_provider_register"
    },
    {
      "id": "lfm3-audio-tts",
      "capability": "speech_synthesis",
      "keys": ["gguf:lfm3-audio-tts"],
      "model_ids": ["LFM3-Audio-1.5B", "lfm3-audio-1.5b"],
      "voices": ["female", "male"],
      "target": "gufo_lfm3_hip",
      "entry": "lfm3_tts_provider_register"
    }
  ]
}
```

> **Note on `entry` functions**:
> `cmake/GufoExtension.cmake` deduplicates or calls each provider's `entry` function when `target` (`gufo_lfm3_hip`) is built, and always calls the top-level `"entry": "lfm3_register"` when `gufo_lfm3` is built. You can either use the same `"entry": "lfm3_provider_register"` across all three providers or separate `lfm3_asr_provider_register` / `lfm3_tts_provider_register` functions in `provider.cpp`.

---

## 3. Step-by-Step Implementation Plan

### Step 1: Sync Backbone Additions from `lfm3` Branch
1. Diff `src/models/lfm3/{engine.hpp,engine.cpp,tokenizer.hpp,tokenizer.cpp}` between commit `c888fe1a` (LFM3 text) and `2ac4eb9e` (LFM3-Audio tip):
   ```sh
   jj diff --from c888fe1a --to 2ac4eb9e src/models/lfm3/
   ```
2. Port those additions into `lfm3-gufo-extension/engine.{hpp,cpp}` and `lfm3-gufo-extension/tokenizer.{hpp,cpp}`.

### Step 2: Port `src/models/lfm3_audio/` into `lfm3-gufo-extension/audio/`
1. Copy `weights.{hpp,cpp}`, `audio.cpp`, `model.{hpp,cpp}`, `generation.{hpp,cpp}`, and `hip/ops.{hpp,hip}` into `lfm3-gufo-extension/audio/`.
2. Replace internal `#include "src/core/..."` headers with public `<gufo/ext/...>` headers:
   - `src/core/gguf_reader.hpp` -> `<gufo/ext/gguf_reader.hpp>`
   - `src/core/json.hpp` -> `<gufo/ext/json.hpp>`
   - `src/core/quant.hpp` -> `<gufo/ext/quant.hpp>`
   - `src/core/ggml_dequant.hpp` -> `<gufo/ext/experimental/ggml_dequant.hpp>`
   - `src/models/lfm3/...` -> `"../..."`
   - `src/models/lfm3_audio/...` -> `"..."`
3. Ensure all HIP compile flags in `extension.cmake` use single-option generator expressions (`$<$<COMPILE_LANGUAGE:HIP>:-O3>`, etc.) to satisfy `_gufo_validate_external_target` in `cmake/GufoExtensionDiscovery.cmake`.

### Step 3: Adapt `Lfm3AudioRuntime` to `<gufo/ext/asr.hpp>` and `<gufo/ext/tts.hpp>`
1. Create `lfm3-gufo-extension/audio/runner.{hpp,cpp}` based on `src/cli/serve/lfm3_audio_runner.{hpp,cpp}`.
2. Implement `gufo::ext::AsrRunner` (`<gufo/ext/asr.hpp>`):
   - `Transcribe(const ext::TranscriptionRequest&, const ext::CancellationCheck&, ext::TranscriptionResult*, std::string* error)`
   - Note: `ext::TranscriptionRequest` passes raw WAV bytes (`request.wav`) or `request.pcm16k`. If `request.wav` is provided, decode the WAV header/PCM in the extension (or via `sndfile` / lightweight WAV parser) before passing 16 kHz mono float samples to `Lfm3AudioRuntime`.
3. Implement `gufo::ext::TtsRunner` (`<gufo/ext/tts.hpp>`):
   - `Synthesize(const ext::SynthesisRequest&, const ext::CancellationCheck&, ext::SynthesisResult*, std::string* error)`
   - `Voices()` returning `{"female", "male"}`
   - `Variant()` returning `ext::TtsVariant::kCustomVoice`
4. Implement pure artifact probes in `probe.{hpp,cpp}`:
   - `ProbeLfm3Audio(std::string_view model_root)`: returns `true` when `model_root` is an LFM3 GGUF (or directory containing an LFM3 GGUF) with a companion `mmproj*.gguf` whose `clip.audio.projector_type == "lfm3a"` and `clip.gen.audio.projector_type == "lfm3a_gen"`.
   - Update `ProbeLfm3(std::string_view model_root)` if needed so a directory/file with an LFM3-Audio `mmproj` is unambiguously claimed by ASR/TTS when probing `serve asr` / `serve tts`.

### Step 4: Register Providers in `extension.cpp` and `provider.cpp`
1. In `extension.cpp` (CPU / always-linked target `gufo_lfm3`):
   - When `!defined(ENGINE_ENABLE_HIP)`, register CPU probe-only entries via `registry.AddAsrProvider(...)` and `registry.AddTtsProvider(...)` with `.create = nullptr` so HIP-free CPU builds can probe LFM3-Audio artifacts and report `"requires hip"`.
2. In `provider.cpp` (HIP target `gufo_lfm3_hip`):
   - Call `registry.AddProvider(...)` for `lfm3`, `lfm3-audio-asr`, and `lfm3-audio-tts` descriptors.
   - Under `#if defined(ENGINE_ENABLE_HIP)`, register:
     - `registry.AddTextProvider(...)` (`id = "lfm3"`)
     - `registry.AddAsrProvider(...)` (`id = "lfm3-audio-asr"`)
     - `registry.AddTtsProvider(...)` (`id = "lfm3-audio-tts"`)

### Step 5: Wire Build Targets & Tests in `extension.cmake`
1. Add `audio/weights.cpp`, `audio/audio.cpp`, `audio/model.cpp`, `audio/generation.cpp`, `audio/runner.cpp` to `gufo_lfm3_hip` `SOURCES`, and `audio/hip/ops.hip` to `gufo_extension_hip_sources(gufo_lfm3_hip SOURCES ...)`.
2. Add `lfm3_audio_test` and `lfm3_audio_probe` test targets with `gufo_extension_target(...)` (including `tests/gguf_test_shim.hpp` in standalone test binaries so they do not link prohibited private `gufo_core`).
3. Copy `docs/models/lfm3-audio/*` into `lfm3-gufo-extension/docs/audio/`.

---

## 4. Verification Checklist

```sh
# 1. Configure & build GPU targets
nix develop -c cmake --preset gpu-test -DGUFO_EXTENSION_DIRS=/home/mixer/gufo/lfm3-gufo-extension
nix develop -c cmake --build --preset gpu-test --target gufo lfm3_config_test lfm3_ops_test lfm3_session_test lfm3_probe lfm3_audio_test lfm3_audio_probe --parallel 8

# 2. Check extension & provider diagnostics
./build/gpu-test/gufo diagnose --section extensions --json

# 3. Run unit, operator, session, and audio tests
./build/gpu-test/extensions/build/lfm3/build/lfm3_audio_test /home/mixer/gufo/models/lfm3-audio

# 4. Run live HTTP conformance for ASR and TTS
./build/gpu-test/gufo serve asr -m /home/mixer/gufo/models/lfm3-audio/... --port 18090
./build/gpu-test/gufo serve tts -m /home/mixer/gufo/models/lfm3-audio/... --port 18091

# 5. Format & docs checks
nix shell --inputs-from /home/mixer/gufo nixpkgs#clang-tools -c clang-format --dry-run --Werror $(find /home/mixer/gufo/lfm3-gufo-extension -name '*.cpp' -o -name '*.hpp' -o -name '*.hip')
python3 /home/mixer/gufo/tools/ci/check-docs.py
```
