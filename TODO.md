# TODO

What the in-tree port on the Gufo `lfm3` branch (PR #886, head `500df164`)
has that this extension does not, and what each piece needs. Everything the
extension API can carry is already here: LFM3 text, LFM3-VL text and images,
and LFM3-Audio transcription and synthesis. The kernels match the branch;
its last five commits add d1 paths only and change no existing kernel.

The items below need Gufo core changes first. A change to `include/gufo/ext/`
or its `SURFACE` closure needs an `ext-api` PR label, an increased extension
API version and an `include/gufo/ext/CHANGES.md` entry (see Gufo's
`AGENTS.md`).

## LFM3-Audio surfaces owned by Gufo core

The model side of each exists in [audio/](audio): `Generator` keeps a
persistent context (`Append`, `Continue`, `Replace`), `LiveTranscriber`
streams 16 kHz audio, and `SpeechBatch` batches synthesis.

### `--mmproj` for `serve asr` and `serve tts`

- Now: `AsrRunnerFactory` and `TtsRunnerFactory` receive only `model_root`
  and a context size, so the extension pairs the mmproj by file name
  ([probe.cpp](probe.cpp)).
- Needs: a factory options argument in `asr.hpp`/`tts.hpp` that carries
  the projector path, or the provider's parsed `ParsedOptions` from
  `options.hpp`. Then declare `--mmproj` (or reuse the core spelling `serve
  llm` already has) and pass it to `FindLfm3Audio`.
- Also: speech services default `model_id` to the Qwen names unless
  `--served-model-name` is given. The default should come from the provider
  (`ProviderInfo::model_ids` or a runner method).

### `gufo transcribe --live`

- Needs: a live-session method on `ext::AsrRunner` (the branch's
  `server::LiveTranscription`, in `src/cli/serve/realtime_audio.hpp`:
  `Push` samples and `Finish` with a text callback, and the transcript so
  far), plus a `transcribe` CLI path that uses it.
- Extension: wrap `lfm3_audio::LiveTranscriber`, the same way
  `src/cli/serve/lfm3_audio_runner.cpp` `LiveStream` does on the branch.

### `/v1/realtime` WebSockets

Live transcription (`?intent=transcription`) and spoken conversations, with
`server_vad` turn detection, barge-in, `conversation.item.truncate`, tools
and reasoning.

- Core: move the branch's `voice_websocket.cpp`, `voice_activity.hpp`,
  `asr_websocket.cpp` changes and `realtime_audio.hpp` interfaces onto
  ext contracts. Turn detection and the WebSocket protocol stay in core.
- Ext API: the live-transcription method above, and a voice-conversation
  contract (the branch's `server::VoiceConversation`: `Configure`,
  `AppendAudio`/`CommitAudio`/`ClearAudio`, `AddUserText`, `AddToolOutputs`,
  `Respond` streaming text and 24 kHz audio, `TruncateReply`). It is a new capability kind or an
  optional ASR runner extension; capability kinds are core contracts.
- Extension: port the branch's `Conversation` class from
  `lfm3_audio_runner.cpp` and restore `ReplyParser` (removed from
  `audio/generation.*`: it splits `<think>`, content and `<tool_call>`, and
  needs a Qwen tool-call scanner). The scanner is
  `tokenization::ScanQwenToolCall` in core; expose it in the ext API or
  give the extension its own.
- Branch fixes to carry with it: `3b88f18c` (32768-slot voice context, keep
  the current input when the reply budget exceeds the room) and `26977118`
  (reply echo and short noise do not start turns; adds
  `GenerateOptions::suppress_tokens`).

### `/voice` browser page

- Core only (`voice_page.cpp`, subprotocol API keys). It follows the
  realtime contract above.

### `gufo speak`

TTS, voice chat (`--mode chat`), text replies (`--mode text`, `--reasoning`)
and `--tools`.

- Needs: a CLI driver hook on speech providers, like `TextProvider`'s
  `prompt_driver`/`chat_driver`, or a core `speak` command over the
  `TtsRunner` and the conversation contract above.

## d1 decision models (`LiquidAI/d1-Omni-2B-GGUF`)

`gufo decide` and `gufo serve decision` (`/v1/systemone`) over text, images,
audio and spoken questions: `noul`, `choice`, ordinal `score`, `box` and
`boxes`, packed question sets of 1-128.

### Gufo core

- A `decision` capability: `Capability::kDecision`, a `DecisionProvider`
  (`probe`, `resolve`, `create`), a `DecisionRunner` (request to response, or
  typed questions to typed answers) and `Registry::AddDecisionProvider`;
  `capability: "decision"` in the metadata schema and
  `cmake/GufoExtension.cmake`.
- `gufo decide` (`--request`, `--jsonl`) and `gufo serve decision` resolve
  a provider through the registry instead of calling `models::d1` directly
  (branch: `src/cli/decide/`, `src/cli/serve/decision_service.hpp`,
  `http_server.cpp`, `serve.cpp`).
- `include/gufo/ext/` change: label, API version, `CHANGES.md` entry.

### This extension

- Backbone: `Config::decision` (accept non-causal `lfm3-d1-omni` GGUFs;
  [config.cpp](config.cpp) rejects them now), `Session::ForwardDecision`
  (both overloads), and the `rows` scratch size in `engine.cpp`
  (`431efcc7`, `f1683b74`).
- Kernels: `TiledAttn<D, bidirectional, packed>` with question groups,
  `BidirectionalAttention` and `SelectRows` in [hip/ops.hip](hip/ops.hip).
- Audio encoder: accept an encoder-only mmproj (`clip.has_audio_generator`
  false) in `audio/weights.cpp` and `audio/model.cpp` (`431efcc7`).
- Model: port `src/models/d1/` (`model`, `prompt`, `packing`, `boxes`); it
  uses the LFM3-VL vision tower in [vl/](vl) and the audio encoder.
- A decision provider in `extension.json` and its registration.
- Tests and tools: `tests/models/d1/`, `tools/d1/reference.py` and
  `README.md`, the branch functional suites `d1.py` and `d1_packed.py`, and
  `docs/models/d1/`.

## Other gaps

- Extension tests are not registered with Gufo's CTest; run the binaries in
  `build/<preset>/extensions/build/lfm3/build/`.
- The Gufo functional suites used for these models (`asr_service.py`,
  `model_services.py` LFM3-Audio cases, `realtime_voice.py`, `lfm3_vl.py`,
  `lfm3_audio_context.py`) exist only on the `lfm3` branch.
- The LFM3-VL reference ([tests/vl_reference](tests/vl_reference)) needs
  transformers 5.21.0 and the pinned safetensors snapshot; the development
  shell has 5.17.0, so the extension port was checked against the in-tree
  build instead.
