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
