# LFM3 retained experiments

October 28, 2026, Strix Halo `gfx1151`, GCC 16.2 / ROCm 10.1 HIP Clang 24.
Keep packed weights resident and reuse the existing quantized GEMV / exact
BF16 projection machinery. Every retained route is the default.

| Mechanism | Decision and evidence |
| --- | --- |
| Fuse width-seven convolution and history update | Retained. One owner per channel consumes and updates the six delayed B*x products; FP64 operator and continuation checks pass. |
| FP32 sliding KV ring | Retained. Only the last 256 positions persist; attention reads the previous ring plus current chunk before cache publication. Independent attention and wrap tests pass. |
| Full-head paired, masked-frequency RoPE | Required for correctness. Generic shortened-head partial RoPE changes the pairing. FP64-formula tests cover the untouched tail and a high position. |
| Double evaluation of the RoPE frequency power, rounded back to FP32 | Retained. Device float pow produced phase drift at position 12001; the corrected frequency passes the unchanged tolerance. |
| Shared-weight exact BF16 prefill | Retained. Removes one full weight read/launch per prompt token; 300-token batched/serial logits remain identical. Matched end-to-end measurements are in BENCHMARKS.md. |
| Three separately launched residual-corrected Q8 WMMA passes | Superseded by fused fixed-point encoding after passing the unchanged quality gates. The earlier version reached 1291 tok/s. |
| Two residual passes | Rejected. The independent projection test at 32 rows failed its unchanged 0.0001 tolerance (observed error about 0.00012). |
| Wave32 64x64 Q8 tiles | Retained for LFM3's smaller matrices. More blocks occupy the 40 CUs than the generic 128x128 tile. Ragged projection and full-model checks pass. |

Separate `tools/prof/prof.py` traces are retained locally as
`artifacts/lfm3/profile-bf16-{before,after}` and
`artifacts/lfm3/profile-q8-wmma`. Timings exclude model loading; full trace
utilization includes uploads and must not be presented as inference utilization.
The Q8 profile identifies matrix projections and attention as the remaining
major costs. Upstream's ordinary activation-quantized prefill remains faster;
this implementation preserves a higher-precision activation contract.

## Prefill target: 5000 tokens/s

The following experiments use Q8_0 pp2048/tg128. Exploratory speeds are
single-run observations; retained repeated measurements belong in BENCHMARKS.md.

| Mechanism | Decision and evidence |
| --- | --- |
| One wave per attention query/head | Retained. Removes per-key CTA barriers while preserving the four dot partials and online-softmax order. |
| Prefetch eight attention K/V rows in registers | Retained. Keeps sequential softmax updates and hides global load latency; independent FP64 attention checks pass. |
| 512-token prefill chunks | Retained in CLI and HTTP. More matrix reuse than 128; 2048-token chunks regressed (2239 versus 2361 tok/s). State tests cross the 512-token boundary. |
| Fuse three Q8 digit products, scheduling barrier after each product | Retained. Stage packed weights once; the scheduling barrier reduced register pressure. A naive fused version was slower and was replaced. |
| One-pass power-of-two fixed-point activation encoding | Retained. Three signed base-256 digits encode about 22 bits relative to the row maximum. Row scales move outside the K reduction. Independent FP64 projections pass the unchanged 0.0001 gate, including 512 rows. This is a bounded approximation, not lossless FP32. |
| Repacked prefill weights | Rejected in standalone measurements: extra layout/copy provided no useful speed gain. |
| Combine digit products per K block | Rejected in standalone measurements: integer reconstruction and FP32 merging did not improve the three-accumulator route. |
| Shared KV attention tile across four queries and four heads | Rejected. Operator/state checks passed, but complete prefill was 2323 versus 2361 tok/s. |

The standalone matrix experiment (`tools/lfm3/prefill_bench.hip`) rotates
four weight buffers beyond MALL capacity and compares sampled outputs with an
independent FP64 product. It supports tile exploration; its kernel time is not
a complete-model throughput claim. Raw rejected trials remain in local artifacts.
The initial optimization profile attributed roughly 49% of GPU time to Q8
projections and 34% to attention. Reaching 5000 remains an open performance target.

## Serving state and batching

The same dense projection kernels now accept independent autoregressive rows
from up to eight sessions; convolution and attention retain private histories,
positions and ring buffers. C2/C3/C4/C8 full-logit comparisons with independent
serial sessions are exactly equal for Q8 and BF16. This establishes correctness,
not an unmeasured concurrency throughput gain.

Snapshots retain executed KV rows, convolution history, the normalized hidden
frontier and logits. Partial-prefill snapshots compute their head lazily before
serialization, so restore can immediately select from the correct frontier.
The native template advertises a stable checkpoint before assistant framing;
the common cache also attempts a complete-prompt retry checkpoint. Existing
history priorities and bounded-budget admission are unchanged.

Native GGUF control-token discovery and uninterrupted ordinary BPE fix tool
history reuse without parsing client literal tags as framing. API schemas use
the existing native Qwen-style tool syntax because LFM3 uses those exact tool
delimiters. Independent Jinja and tokenizer fixtures remain model-private.

## October 29 pass

Q8_0, pp2048/tg128 unless stated; quality evidence in QUALITY.md.

| Mechanism | Decision and evidence |
| --- | --- |
| Split activations into FP16 high/low pairs, two F16 WMMA chains per K32 block | Retained. Replaces the three-digit int8 encoding: lower FP64 projection error and 1.5x faster GEMMs (about 31 raw F16 TFLOPS). pp2048 3010 → 4430. |
| Register-prefetched, LDS double-buffered stages; 256x64 tiles (64x64 under 1024 rows) | Retained. The loop with no global loads, commits or scales still reaches only 39 raw TFLOPS: barriers and LDS fragments set the ceiling. |
| Register-resident weight tiles without LDS | Rejected: 256 VGPRs with spills, 15–25% slower. |
| Split-K for 2048-row projections | Rejected: 1152 versus 1024 µs with its reduction; block count is not the limit. |
| 1024 / 2048 / 256-token chunks | Rejected: 4100 / 3940 / 4600 versus 4720 at 512; larger chunks leave the 32 MiB MALL. |
| Shared input split for q/k/v/gate and gate/up; SwiGLU in the gate/up epilogue | Retained, byte-identical: 4430 → 4660. |
| Fused post-norm, residual add, RMSNorm and split per row | Retained, byte-identical (same reduction order): 4660 → 4720. |
| 16-byte output stores after a lane-half swap | Retained, byte-identical: stores cost 5–12% of GEMM time; 4800 → 4930. |
| Down projection splits FP32 inputs while staging, using SwiGLU row maxima | Retained, byte-identical: about +1%, about 5000. The same for conv and attention outputs was neutral and was dropped. |
| Prompt attention: lanes own keys, queries broadcast by DPP row_share FMAs | Retained: 156 → 26 ms per pp2048, lower FP64 error. |
| DPP broadcast of P for P·V instead of readlane | Rejected: DPP FMAs cannot dual-issue, 52 → 64 ms. |
| Scalar-loaded queries for dual-issued QK | Rejected: hoisted loads spill 914 SGPRs. |
| Row-parallel convolution; RoPE frequency table | Retained, byte-identical: convolution 452 → about 30 µs per 512 rows. |
| Split-K decode attention, 64-key tiles staged in LDS, at most 64 spans | Retained: tg128 at 4096 36.5 → 98.9, at 128K 38.7 (about 88% of the K/V+weight bandwidth floor). |
| Prompt attention split into ≥8192-key spans scheduled together | Retained: lower error over long histories; pp512 at 32K 1891 → 2073. Compute (about 5.8 TFLOPS FP32) remains the limit. |

The next prompt-depth lever is attention arithmetic. The current FP32 SIMT
kernel reaches about 40% of the issue rate; matrix-core attention would need a
precision decision (for example split FP16 products).

## November 8 pass: decode launches and depth

Q8_0, `gufo bench` CLI (not the HTTP tables in BENCHMARKS.md), Gufo
`6d793814`. Attention timings are per global layer and 512-row chunk from a
standalone operator benchmark with 16 heads, 4 KV heads and D = 128.

| Mechanism | Decision and evidence |
| --- | --- |
| One launch for one-row q/k/v/gate GEMVs; gate/up GEMV with SwiGLU; per-head Q/K RMSNorm, RoPE and KV store in one kernel; in-place one-row convolution | Retained. Decode dropped from 322 to about 180 launches per token (about 2 µs idle each). Same expressions and reduction order: greedy decode logits are byte-identical for one session and a three-session batch. tg128 105.6 → 110.4 at depth 0, 73.7 → 76.2 at 32K. |
| Prompt key spans of at least 2048 instead of 8192 | Retained. Balances causal blocks at mid depths: 8K 6.62 → 5.71 ms, 16K 11.7 → 11.0 ms. Lower FP64 error; a 9000-token prompt changes logits by at most 1.4e-5 (argmax unchanged). |
| Split-FP16 WMMA prompt attention (Q, K, V and P as high/low pairs, three products) | Rejected. 86.5 → 68.9 ms at 128K, but about 10× the FP32 kernel's error and fails the 2e-6 operator gate. |
| Register prefetch of the next K/V tile (prompt and decode) | Rejected. Prompt unchanged (latency already hidden by resident blocks, 208 → 252 VGPRs); decode 2.70 → 3.27 ms at 128K. |
| 512-thread prompt blocks (twice the queries per K/V tile) | Rejected: slightly slower; not bound by K/V traffic. |
| QK with scalar-register queries or readlane broadcasts instead of DPP | Rejected: SGPR/VGPR spills (2.5×–10× slower) and 5× slower. Ablation attributes about 59% of prompt-attention time to the DPP QK FMAs, which cannot dual-issue. |
| Decode attention tile/split sizes (32/64 keys, 64–256 splits) | Neutral. Decode attention reads the FP32 KV cache at about 199 GB/s; lower KV precision is the remaining depth lever. |
| Batched decode GEMM with 4 or 8 rows per wave | Rejected: slower than 2. |

## November 8 pass: F16 K/V caches

Every attention K/V cache in the extension is now F16, llama.cpp's default
and Gufo's standard (Liquid AI ships BF16 weights and specifies no FP32 cache).
The FP32 cache route was removed, not kept as an option. Q8_0 `gufo bench`
pp2048 / tg128 against the committed FP32-cache build:

| Depth | FP32 K/V | F16 K/V |
| ---: | --- | --- |
| 0 | 5019 / 110.3 | 5179 / 110.3 |
| 8K | 3749 / 98.0 | 4677 / 103.7 |
| 16K | 2958 / 88.9 | 4187 / 98.1 |
| 32K | 2124 / 76.2 | 3582 / 88.7 |
| 128K | 784 / 39.4 | 2040 / 60.3 |

| Mechanism | Decision and evidence |
| --- | --- |
| Matrix-core prompt attention: S^T = K Q^T and O^T = V^T P^T, so lane L owns query L % 16 and the softmax needs one cross-half exchange; F16 K, V and probabilities, FP32 scores and accumulation | Retained. One global layer, 512 rows at 128K: 86.5 → 23.6 ms (with split Q below). |
| Q as one F16 value | Rejected for quality. The 513-token chunked/serial session gate measured 0.0025 against 0.002; FP64 operator error about 3e-4. |
| Q as F16 high part plus F16 remainder (two QK products) | Retained. FP64 error about 5× lower (6.7e-5 at depth 0, 1.1e-5 at depth); session gate 0.00136 Q8_0, 0.00175 BF16. Costs about 40% of attention time; the kernel then issues WMMA at about 60% of the F16 peak, 256 VGPRs. |
| Loading each K/V tile at its use instead of a register prefetch | Rejected: 6.4 → 7.3 ms at 32K. |
| Decode attention over the F16 ring with the V tile kept F16 in LDS | Retained. LDS 39 KB → 23 KB per block doubles resident blocks: 128K layer 1.59 → 1.32 ms (about 204 GB/s). |
| Decode spans: 64 up to 64K keys, 128 beyond | Retained: 128K 1.32 → 1.26 ms; 128 spans hurt short histories. 32 or 256 spans and 128-key tiles were neutral or slower. |
| Sliding rings | Full-attention chunks are packed straight into the cache; sliding chunks go through an F16 staging buffer and are written to the ring after attention reads the history. |

Cache-free attention reads the same F16 rows: decision prompts (bidirectional
TiledAttn), audio stacks (rings and stateless) and the VL encoder. Checks:
operator FP64 references over the stored F16 values, session, continuation,
D1 and VL contract tests, the VL (19 checks) and D1 packed (26 checks)
functional suites, LFM3-Audio ASR of the English fixture (exact transcript), a
TTS round trip transcribed back word for word, and a 41.5K-token needle
retrieval. Snapshot version 2 and continuation layout version 2 reject FP32
state.

## November 8 pass: prompt processing

Q8_0 `gufo bench` pp2048 / tg128 against the F16-cache commit (`0f6c662`).
Microbenchmarks ran at model shapes with 512-row chunks.

| Depth | Before | After |
| ---: | --- | --- |
| 0 | 5179 / 110.3 | 6282 / 110.4 |
| 8K | 4677 / 103.7 | 5731 / 103.5 |
| 16K | 4187 / 98.1 | 5064 / 97.9 |
| 32K | 3582 / 88.7 | 4419 / 88.6 |
| 128K | 2040 / 60.3 | 2569 / 60.3 |

| Mechanism | Decision and evidence |
| --- | --- |
| Register-resident weights (`RegQ8Gemm`): each wave owns its rows, so weights load straight into WMMA registers and only activations pass through LDS; products transposed so each lane scales by its own row's Q8 scale | Retained. Byte-identical to `SplitQ8Gemm` without splits; 16 KB LDS fits 3 blocks per WGP (60 slots) instead of 2. |
| Tail split-K: tiles of the last partial wave of resident blocks split across K, last part sums partials in part order | Retained. Deterministic; FP64 projection error unchanged. Projections per layer-chunk: out 294 → 240 µs, FFN 2804 → 2587 µs. A part costs about four K blocks, so the planner skips in_proj. |
| q/k/v/gate in one wide launch (waves past a narrow job's rows skip products) | Retained, byte-identical: 460 → 375 µs. |
| Trailing-row pruning: only the frontier row leaves the last layer; after the last full-attention layer's K/V, its remaining work and the final convolution layer run on the last 7 (then 1) rows | Retained. Exact by construction (decision and batched rows excluded); 1500-token prompt + 24 decode steps: max logit change 3e-6, argmax identical. About +6.6% at depth 0, +9.7% at 32K. |
| Prompt Q/K norms, RoPE and F16 K/V packing in one launch (`DecodeHeadsKernel` over rows); convolution sliding a 7-tap window down 16 rows | Retained, byte-identical logits: +2.6% at depth 0. |
| Attention: skip masking on interior tiles, skip the accumulator rescale when every alpha is exactly 1, fold the softmax scale into Q before its split | Retained. Mask/rescale skips byte-identical; scale fold keeps the FP64 error (8.4e-6 at 128K). About 8% faster attention. |
| At most 8 prompt key spans (was 16) | Retained: 32K layer 5.95 → 5.62 ms, error unchanged. |
| 768 / 1024-row chunks | Rejected again: 5630 / 5530 versus 5725 at 512. |
| 4×2 waves, 128-token tiles, 4-wave blocks, two K blocks per barrier, half-row weight loads with `v_permlanex16` | Rejected: neutral or slower (128-token tiles spill). |
| Q high/remainder packed into one register set | Impossible: gfx1151 WMMA lanes 0-15 compute even output rows and 16-31 odd rows, each from its own full operand copy (a garbage-filled probe changes results). |
| Attention LDS double buffer (one barrier per tile); more key spans or group-fastest grid for few-row tails | Rejected: neutral. A 7-row tail at 128K reads 268 MB of K/V at about 215 GB/s, so it is DRAM-bound. |

Pure WMMA reaches about 51 TFLOPS on this device; the split GEMM runs at
about 37 raw TFLOPS. Ablation: weight loads about 9%, activation loads 5%,
barrier 3%. Attention is about 60% WMMA time, 10% K/V loads. The remaining
large levers lower precision (F16-only activations or Q) and need a decision.

## November 10 pass: FP16 prompt activations

Q8_0 `gufo bench` pp2048 / tg128, Gufo `bd6b0fdd`, against extension
`23efc07`. Quality evidence for the precision change is in QUALITY.md.

| Depth | Before | After |
| ---: | --- | --- |
| 0 | 6281 / 110.6 | 9399 / 111.8 |
| 8K | 5713 / 103.8 | 8392 / 105.1 |
| 32K | 4428 / 88.9 | 6276 / 90.8 |
| 128K | 2548 / 60.3 | 3432 / 60.9 |

The HTTP text suites (`all`, provider preset, 32K context) pass and fail the
same cases as before with the same responses; prefill time per request fell
by a median of 31% (89 matched requests, none slower) and decode by 1%. The VL
(19 cases), ASR, d1 packed (26 checks) and d1 parity suites pass; d1 answers
move by at most 2.5e-4 in probability against the previous build.

| Mechanism | Decision and evidence |
| --- | --- |
| Prompt projection activations as one row-scaled FP16 plane instead of the high/remainder pair; two WMMAs per K32 block instead of four | Retained (precision decision). KL against BF16 unchanged (1.252e-3 → 1.255e-3), KL to the previous path 2.4e-6, identical top-1. pp2048 6150 → about 8450. |
| FFN hidden state converted to FP16 once (row maxima from the SwiGLU epilogue) instead of while staging the down projection | Retained, byte-identical logits: each of the 8 row tiles had re-read and re-converted the FP32 rows. Down projection 750 → 562 µs per layer and chunk; pp2048 about +8%. The FP32 staging route was removed. |
| Single F16 query in matrix-core prompt attention | Retained (same precision decision). On a 27K-token text, KL against BF16 9.79e-4 → 9.72e-4, KL to the split query 1.5e-6, identical top-1. pp512 at 32K 5888 → 6232. |
| One decode GEMV row per wave (four for the 7680-column down projection); gate and up rows of a SwiGLU block in separate waves joined through LDS | Retained, byte-identical (each row keeps its reduction order). tg128 109.8 → 111.4. The GEMVs now read at 92–96% of the measured 240.8 GB/s DRAM ceiling. |
| Barrier-free decode attention: a wave per KV group takes batches of 8 keys, lane L owns dimensions 4L..4L+3 of every head, scores reduce by DPP, waves merge in LDS | Retained: 64-key tiles with six block barriers each reached 87% of the cache layout's read ceiling (about 220 GB/s for 256-byte group slices of 1 KiB rows), now 92%. tg128 at 32K 89.3 → 90.4. Unrolled span merge: 108 → 56 µs per token at 32K. |
| HIP graph capture of the decode step | Rejected: +0.5%. The about 1.9 µs between dependent kernels is GPU-side, not launch overhead. |
| 256 / 1024-row prefill chunks | Rejected again: 8430 / 9240 versus 9200 at 512 (1024 within noise, more memory and scheduling latency). |
| Down projection tiles (16-row token tiles, 4-wave blocks), split-K off | Rejected: 5–25% slower than 32-token, 8-wave tiles with tail split-K. |
| Wave-tree RowSquares in the residual norm (two barriers instead of twenty) | Rejected: tg unchanged; the single-block norm waits on loads, not barriers. |
| Dword-aligned weight loads in decode GEMVs; 2–8 waves per block | Rejected: neutral. |

Remaining decode cost is the 1.84 GB weight read per token (7.6 ms at the
ceiling, 8.9 ms measured) plus 184 launches with GPU-side gaps. Prefill gate/up
projections run at about 29 effective TFLOPS (54% of the WMMA peak) with one
FP32 FMA per accumulator and K block for the Q8 scale. The next precision lever
would fold the Q8 scale into FP16 weights (rounding each q·d to 11 bits).

## November 10 pass: model loading

Cold model files (evicted from the page cache and verified non-resident) to
HTTP readiness, Gufo `bd6b0fdd`, against extension `033a9e8`; text at 262144
tokens of context, VL at 16384, d1 at 8192.

| Model | Before (s) | After (s) |
| --- | ---: | ---: |
| LFM3 1.7B | 3.73–3.84 | 0.92–1.16 |
| LFM3-VL 2B | 4.22–4.25 | 0.97 |
| LFM3-Audio 2B (ASR) | 4.30–4.34 | 0.68 |
| d1-Omni 2B | 4.45–4.46 | 0.73–0.77 |

Every HTTP response (greedy text and VL logprobs, ASR, d1 probabilities) is
byte-identical to the previous build: weights are still copied verbatim.

| Mechanism | Decision and evidence |
| --- | --- |
| Tensor payloads through Gufo's `hip::WeightUpload` (16 unbuffered readers, pinned staging) instead of `hipMemcpy` from the mapping | Retained. Serial page faults read the file at about 0.6 GB/s; the readers reach 5.3 GB/s against 5.7 GB/s measured for 16 parallel `O_DIRECT` reads of this disk. Resident memory at the session phase falls from 2057 to 305 MiB. |
| One allocation per 128 MiB of payloads, in file order, instead of one per tensor | Retained: reads start after the first segment; `hipMalloc` maps zeroed pages at about 28 GB/s regardless of threads, so later segments allocate while the disk is busy. |
| Tokenizer parsing and the first session's K/V caches (3 GiB at 262144 tokens) allocated while the weights stream | Retained: removes 60 ms and 110 ms from the serial path. |
| Vision and audio projectors loaded beside the backbone | Retained: VL 1.18 → 0.97 s, ASR 0.90 → 0.68 s, d1 0.86 → 0.73 s. |

Remaining text readiness is about 70 ms before the extension is called, about
110 ms of HIP initialization and staging setup, 350 ms of disk reads at the
ceiling, and Gufo's 2 GiB RAM-cache reserve, prepared before readiness
(about 300 ms of page population). With the file already in the page cache,
loading takes 0.82–0.88 s against 0.80–0.95 s before: unbuffered reads skip
the cache, as for Gufo's own models.

## November 10 pass: short prompts

Gufo `bd6b0fdd`, against extension `9506e27`. Q8_0 `gufo bench`, prompt
pass time:

| Rows | 12 | 16 | 20 | 24 | 29 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before (ms) | 11.3 | 14.8 | 18.7 | 21.3 | 26.1 | 13.4 |
| After (ms) | 11.3 | 12.7 | 12.9 | 13.0 | 13.1 | 13.1 |

| Mechanism | Decision and evidence |
| --- | --- |
| FP16 prompt activations from 16 rows instead of 32 (`kPromptRows`): the exact small-batch kernels cost about 0.8 ms per row and passed the FP16 route at 16 | Retained. KL against BF16 for 16–31-token prefixes 2.350e-3 → 2.314e-3, same top-1 (QUALITY.md); other rows bit-identical. 13–15 rows would save about 1 ms and stay exact. |


## November 10 pass: GPU and NPU efficiency on HRX

The question is speed per watt, not peak speed: a slower device that produces
more tokens for the same energy can be the better choice for small models.
Strix Halo `gfx1151`, Gufo `3edc2eff` with this extension at `debba20`,
LFM3-1.7B Q8_0. HRX is AMD's alternative HIP runtime on its native libamdf
driver (hrx-system `dd9572875f`, Gufo's experimental `.#hrx` package; see
Gufo's `docs/internal/hrx-migration.md`). Power is the amdgpu `power1_average`
socket sensor, which covers the whole APU. With no Gufo process and a cooled
GPU the socket draws 9-12 W.

### GPU, HIP against HRX

`gufo serve llm` with one session, single-stream HTTP requests: prefill is
about 2,000-token prompts with `max_tokens=1`, decode 256 greedy tokens; each
phase runs 20 s with its own power average.

| Runtime | Phase | Speed (tok/s) | Socket power | Energy per token |
| --- | --- | ---: | ---: | ---: |
| HIP | prefill | 7,821.3 | 140.6 W | 18.0 mJ |
| HIP | decode | 111.8 | 100.4 W | 898 mJ |
| HRX | prefill | 7,054.4 | 139.5 W | 19.8 mJ |
| HRX | decode | 95.6 | 96.6 W | 1,011 mJ |
| HRX, queue pool | prefill | 7,489.8 | 140.5 W | 18.8 mJ |
| HRX, queue pool | decode | 95.8 | 97.2 W | 1,014 mJ |

`gufo bench` agrees on the speeds: HIP pp2048 9,063-9,196 and tg128
111.3-112.0, HRX 8,720-8,843 and 96.1-96.2 tok/s. HRX decodes LFM3 about 14%
slower, more than on Qwen3.8-Flash-Next, consistent with its higher per-launch
cost on small kernels.

The HRX build used here gave every created stream its own hardware queue.
LFM3 creates about 70 streams, and with that many queues mapped the GPU stays
100% busy at 2.9 GHz with no work queued: an idle server drew 39-41 W against
11-12 W on HIP. A pool of at most four queues idles normally (0% busy,
11-12 W); sixteen spins again. The two "HRX" rows above used the queue per
stream; the "HRX, queue pool" rows use the pooled package
(gufo-org/gufo#1147), with which an idle LFM3 server draws 13 W at 0% busy.
The pool lifts prefill by 6% and leaves decode unchanged.

### NPU

There is no Loom NPU program for LFM3 yet. The closest working NPU model is
Loom's Qwen3-Embedding-0.6B on the XDNA NPU (hrx-loom-library `8ef2e49`,
eight columns by four core rows, original BF16 weights), measured with the
same sensor:

| Load | Socket power | Throughput | Energy per token |
| --- | ---: | ---: | ---: |
| 8 inputs of 15 tokens per call | 20.1 W | 2,047 tok/s | 9.8 mJ |
| 1 input of about 120 tokens per call | 20.2 W | 2,144 tok/s | 9.4 mJ |

Every 128-slot cohort takes about 57 ms whatever it holds, consistent with
streaming the 1.2 GB of weights at about 21 GB/s per cohort.

Projection for LFM3 on the NPU, if that 21 GB/s is the NPU's sustained weight
bandwidth: decode reads about 1.8 GB per token, so about 12 tok/s at about
20 W, or about 1.7 J per token, worse than the GPU's 0.9 J. Prefill at the
embedding's arithmetic rate would be several hundred tok/s at about 20 W,
around 25-40 mJ per token, also short of the GPU's 18 mJ. The NPU wins on
efficiency only if its weight bandwidth is well above what the embedding
program reaches; it wins on concurrency regardless, since it leaves the GPU
free.

Next: measure the NPU's DRAM-to-tile bandwidth directly, then port one LFM3
SwiGLU block to the NPU in Loom and measure tokens per joule against the
GPU's 898 mJ decode figure.
