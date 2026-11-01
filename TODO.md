# TODO

Everything the in-tree port on the Gufo `lfm3` branch (PR #886, head
`500df164`) provides now lives in this extension: LFM3 text, LFM3-VL images,
LFM3-Audio transcription, synthesis, live transcription, realtime
conversations, `gufo speak` and `--mmproj`, and d1-Omni decisions. The
speech and decision surfaces need Gufo's extension API 0.1.12 (the realtime,
`speak`, projector and `decision` contracts). What is left:

- Register the extension's tests with Gufo's CTest. They build with the
  extension, but run them by hand from
  `build/<preset>/extensions/build/lfm3/build/`.
- Move the functional suites used for these models into this repository:
  `asr_service.py`, the LFM3-Audio cases of `model_services.py`,
  `realtime_voice.py`, `lfm3_audio_context.py`, `lfm3_vl.py`, `d1.py` and
  `d1_packed.py` exist only on the `lfm3` branch, with its shared harness
  (`run.py`, `service_http.py`).
- Rerun the numerical teachers. The LFM3-VL reference
  ([tests/vl_reference](tests/vl_reference)) needs transformers 5.21.0 and
  the pinned safetensors snapshot; the d1 reference
  ([tests/d1_reference](tests/d1_reference)) needs the pinned LiquidAI
  source. The development shell has transformers 5.17.0, so the ports were
  checked against the in-tree build instead (identical VL responses; d1
  within 1.3e-6).
- Decide whether the text backbone keeps `-ffast-math`. The in-tree port
  compiled it with `-O3` only; the audio, VL and d1 towers here do too, and
  the backbone's fast math is the source of the last-digit d1 differences.
