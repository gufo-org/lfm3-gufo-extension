# LFM3-Audio experiments

Recorded on the in-tree port (Gufo `lfm3` branch, `2ac4eb9e`), whose
`gufo transcribe` and `gufo speak` commands this extension does not provide;
the extension compiles the same model code.

October 29, 2026. Timings are per TTS frame (80 ms of speech), Q8_0 backbone
with the BF16 mmproj, unless noted.

| Change | Result | Decision |
| --- | --- | --- |
| First port: host sampling, three synchronized depth passes, synchronous codec | 6.2× real time; depth 3.8 ms, backbone 8.4 ms, codec 0.55 ms | Baseline |
| Whole frame on the device: reveal passes, top-k/temperature/repetition sampling, window counts and next frame embedding in one captured HIP graph, one sync per frame | Depth 3.8 → 3.5 ms; greedy frames still identical to the reference | Kept: launch gaps were not the main cost |
| Codec decode in a worker on its own stream, overlapping the next backbone step | Codec leaves the critical path | Kept |
| Profile of the depth decoder | 12-row BF16 projections stream at about 80 GB/s (88 and 32 workgroups for 2816- and 1024-row weights); they are about 2.2 ms of the 3.5 ms | Open |
| Custom 12-row GEMM, one wave per row with LDS-staged activations | 2–3× slower; then accumulators spilled to scratch on data-dependent loop exits | Rejected |
| Same with four rows per wave, weight prefetch, exact-size templates | No spills, still slower than the stock kernel | Rejected |
| Barrier-free split-K variant (eight waves per four rows) | Paired gate/up 155 µs vs 2 × 69 µs stock; activation re-reads saturate L1/L2 at 12 rows | Rejected for 12 rows |
| Same kernel for the one-row 2048 → 12288 depth input | 263 µs vs 320 µs stock (190 GB/s) | Kept as `WideGemv` |
| Q8_0 mmproj | Depth 3.5 → 2.5 ms (7.1× real time), but codec audio moves 7% | Accepted input, not the default |

Concurrent synthesis, eight streams of the same sentence (aggregate real
time; per step depth / backbone):

| Change | Result | Decision |
| --- | --- | --- |
| Batched backbone step (`ForwardBatchInputs`, frame embeddings from the device, no logits) and one depth pass over all voices | 17.4×; depth 20.1 ms, backbone 14.1 ms: the BF16 small-batch GEMM reads weights once per 16 rows, so 96 rows cost six reads; greedy codes identical to a lone stream | Kept |
| 64×64-tile FP32 GEMM over BF16 weights for 32+ rows | Depth 19.1 ms at eight, but slower than the 16-row kernel at four (14.5 vs 10.5 ms); ASR encoder 17.7 → 17.1 ms | Kept for the encoder and codec rows; depth uses hipBLAS |
| One launch per frame kernel over a device table of voices | 18.8×; depth 16.8 ms | Kept |
| hipBLAS SGEMM on FP32 copies of the depth projections (about 236 MB) | 20.8×; depth 13.6 ms (rocBLAS 32×32 tiles, about 2.7 TFLOPS) | Kept for batches of 32+ rows |
| Sixteen streams | 23.5×; backbone 24.3 ms, depth 23.5 ms (per-row launches, linear depth FLOPs) | Rejected: batches stop at eight |

Remaining levers: one attention/RoPE launch for all rows of a backbone batch
(about 5 ms of the eight-stream step) and a faster FP32-activation GEMM for the
depth decoder (BF16 WMMA would round activations).

The backbone step is near its bandwidth floor (see [benchmarks](BENCHMARKS.md)).
Its remaining overhead is about 1 ms of launch gaps; graphing it needs
position-independent attention/RoPE/cache launches in the shared LFM3 engine.

November 9, 2026, extension port, HTTP on a release build. The audio encoder
shares the VL tower's tiled bidirectional attention and split-BF16
projections ([VL experiments](../vl/EXPERIMENTS.md)): the 10.0 s ASR clip
270 → 261 ms with the same transcript. Speech synthesis is byte-identical
(its depth decoder keeps the small-batch BF16 path) at 6.67× real time. A
profile of one sentence puts every frame at its bandwidth floor: backbone
Q8_0 GEMVs at about 220 GB/s and the 267 MB BF16 depth decoder streamed
once per MaskGIT pass; the GPU is busy for the whole request.

## November 10 pass: transcription latency

Gufo `bd6b0fdd`, release build, against extension `18ca844`. First text is
the median of five streamed HTTP requests after a warmup; transcripts of all
14 clips (the fixture, four LibriSpeech utterances and nine 16, 24 and
48 kHz clips from audio.cpp) are identical to the previous build. The 10.0 s
fixture took 51 ms to its first logits: 8.4 ms of encoder, then the system
turn (15 tokens, 14.7 ms), the 125 audio rows (17.1 ms) and the 5-token
assistant header (9.4 ms) in three passes, and 1.2 ms for the head.

| Change | Result | Decision |
| --- | --- | --- |
| Prompt runs (text, audio rows, text) in shared passes (`Session::ForwardInputs`) instead of one pass per run: short runs fell to the exact FP32 small-batch kernels, which read every weight per pass | Fixture first text 54.4 → 35.1 ms; prompt passes 41.2 → 21.8 ms. A live chunk (its pending `<|text_end|>` and 12 audio rows) 20.5 → 15.1 ms. Versus the separate passes (`lfm3_asr_probe`), greedy tokens are identical on all clips, KL ≤ 1.5e-6 per step (mean ≤ 1.1e-7), largest log-probability difference 0.017: the text rows take the FP16 prompt activations already used for the audio rows | Kept |
| Next text token's logits computed by its own forward pass instead of a separate head launch and stream synchronization | Same tokens; part of the totals below | Kept |
| Resampler filter tabulated per output phase (2 for 24 kHz, 1 for 48 kHz, 160 for 44.1 kHz), outputs at exact integer positions, eight partial sums, and one-shot calls spread over up to eight threads | 28 s at 24 kHz 473 → 2.6 ms, at 48 kHz 952 → 3.9 ms; 24, 48 and 8 kHz output bit-identical, 44.1/22.05/11.025 kHz within one float ulp on 0.1% of samples; streamed pieces still equal one call. 24 kHz clips: first text 156–571 → 27–69 ms | Kept |
| Encoder BF16 projections (`Split3Gemm`) with smaller tiles for inputs of up to 256 tokens, chosen by the fewest waves of about 56 resident blocks times the tile's work; 64 × 128 tiles filled 12–36 of the device's slots for a 10 s clip | Encoder 8.5 → 7.4 ms (10 s), first text −0.2 to −1.5 ms; output bit-identical (per-output sums are the same for every tile). Within 5% of the best measured shape per launch; 64 × 128 stays above 256 tokens, where it measured as good or better | Kept |
| `RegQ8Gemm` skips the products of 16-token sub-tiles past the prompt in the last token block | 145-row pass 21.3 → 20.9 ms, bit-identical | Kept |
| Ragged exact small-batch passes: 9–11 rows in one 12-row pass instead of 8 rows and a second pass over the weights | Identical text logprobs over HTTP | Kept |
| 13–15 rows (a live chunk with its pending `<|text_end|>`) in one ragged 16-row pass | 25.1 ms per chunk against 15.1 ms for 12 and 1 rows: these kernels are compute bound, about 0.8 ms per row over the backbone | Rejected |
| FP16 prompt activations (`RegQ8Gemm`) from 9 rows instead of 32, for live chunks | Live chunk 15.1 → 14.2 ms, for FP16 activations in every 9–31-row prefill of all LFM3 models | Rejected |

Totals (HTTP, transcript complete): 16 kHz clips −4 to −22% (the fixture
255.8 → 235.1 ms), resampled clips −33 to −49%. Live transcription of the
28 s clip: compute 2732 → 1977 ms, slowest piece 70.5 → 62.8 ms. Voice chat
first text / first audio 67 / 133 → 33 / 99 ms; greedy TTS stays
byte-identical. Text replies remain the bulk of a transcription: each token
costs 9.1 ms, with the backbone and head GEMVs at about 220 GB/s.

## November 10 pass: speech synthesis latency

Gufo `bd6b0fdd`, release build, against extension `9506e27`. First audio
over HTTP (median of three seeded streamed requests after a warmup) was
40.3, 28.7 and 30.5 ms for 12-, 104- and 246-character inputs. A profile of
the 104-character request: a 49-row prompt pass (14.8 ms), the first depth
frame (3.3 ms), the one-frame codec chunk (1.2 ms), then the first audio
waited for the next backbone step (8.7 ms) before it was handed out.

| Change | Result | Decision |
| --- | --- | --- |
| Join the codec of the first chunk before the next (blocking) backbone step instead of after it | First audio −7 to −8 ms; the backbone starts about 1 ms later once per reply | Kept |
| FP16 prompt activations (`RegQ8Gemm`) from 16 rows instead of 32 | 16–31-row prompt passes cost 14.8–26.1 ms in the exact small-batch kernels and 12.6–13.1 ms now; the 12-character TTS prompt (29 tokens) 27.5 → 13.1 ms. Against the BF16 reference (`lfm3_prefix_probe`, every prefix of three texts) the 16–31-row rows' KL is 2.31e-3 vs 2.35e-3 before, top-1 unchanged; all other rows bit-identical. Greedy TTS is byte-identical; short sampled outputs change, all 24 short-sentence samples transcribe exactly (Qwen3-ASR) before and after | Kept |
| FP16 from 13 rows | About 1 ms per 13–15-row pass | Rejected: the live ASR chunk (13 rows) stays exact |
| Four-wave `RegQ8Gemm` blocks for one-token-block prompts (twice the blocks), split-K for the SwiGLU projection (forced 2 or 3 parts) | 49-row pass −2%; SwiGLU 265 → 248 µs at best | Rejected: at 49 rows the kernel is neither occupancy- nor K-latency-bound |
| Depth decoder 12-row projections (`K = 1024`): activations staged in LDS once per block, waves stream weight rows without per-tile barriers, one persistent block per compute unit | Counters showed the stock kernel's waves waiting 92% of their cycles, one weight round trip per 256 columns. 2816 × 1024 71 → 36 µs, 1024 × 1024 27 → 20 µs in isolation; depth 3.55 → 2.80 ms per frame. Same lane order and reduction: audio byte-identical | Kept |
| Other shapes of the stock kernel: 8 × 1, 4 × 2, 8 × 2, 16 × 1 waves × rows, plain loads, one-tile prefetch, four- and eight-tile load groups | All byte-identical, none faster (the register copy of a prefetched tile waits for it) | Rejected |

Totals: HTTP first audio 19.8 / 20.4 / 21.9 ms (−51%, −29%, −28%), real time
6.62–6.69× → 7.08–7.14× for the two longer inputs. The prompt pass is now
most of the first audio: at 16–64 rows `RegQ8Gemm` reads the backbone at
about 130 GB/s, about twice its bandwidth floor.

