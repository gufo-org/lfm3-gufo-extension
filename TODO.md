# TODO

Nothing is left from the in-tree port on the Gufo `lfm3` branch (PR #886,
head `500df164`). This extension serves LFM3 text, LFM3-VL images,
LFM3-Audio transcription, synthesis, live transcription, realtime
conversations, `gufo speak` and `--mmproj`, and d1-Omni decisions; its
CTest tests, functional suites ([tests/functional](tests/functional/README.md))
and independent numerical teachers ([tests/vl_reference](tests/vl_reference),
[tests/d1_reference](tests/d1_reference)) live here and pass.

It depends on Gufo
[#942](https://github.com/gufo-org/gufo/pull/942) (extension API 0.1.12:
realtime, `speak`, projector and `decision` contracts, and CTest for
extension tests) until that merges.
