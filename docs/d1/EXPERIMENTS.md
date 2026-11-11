# d1 implementation decisions

| Decision | Evidence / outcome |
| --- | --- |
| Reuse the LFM3 backbone with complete bidirectional prompts | Independent HTTP probabilities and FP64 attention checks pass; convolution and sliding attention remain causal. |
| Compile a separate bidirectional attention specialization | Keeps the causal specialization's mask and arithmetic; existing LFM3 operator checks pass. |
| Reuse the LFM3-Audio input encoder without requiring generation tensors | Official d1 mmproj has audio input and vision, but no depth decoder/codec. Audio and spoken-instruction reference checks pass. Voice creation explicitly rejects this input-only projector. |
| Add native vision convolution, axial RoPE and spatial merging | Image and mixed-media reference checks pass. No Python/reference runtime dependency. |
| Packed sets of 16 by default | Matches LiquidAI's recorded training mode; exact layout goldens and FP32 teacher logits pass, including a 17-question partial tail. |
| Sparse output head for packed/box slots | Projects only the union of vocabulary entries read by the answer slots; all selected logits agree with the independent teacher within `0.000063`. |
| Preserve the original single-slot primitive path | A compact replacement (the multi-row head GEMM) shifted legacy probabilities by up to `1.3e-7`. The one-row GEMV over only the read vocabulary (November 11, below) returns the full head's values exactly. |
| Calibrated single/multiple boxes | Median thousandth edges, presence tables, fixed Halton confidence and reading order match the pinned Python reference; 1/4/16 blocks and absence pass. |
| Files before images | Initial image-first ordering failed the mixed-media probability gate; reference ordering passes without changing tolerance. |
| Reuse the decision session and its readout buffers across prompts (grow only when a prompt exceeds its rows) | A new session per prompt allocated every activation, a full-context K/V cache and a stream, then freed them: about 13 ms of idle GPU per prompt. Responses stay bit-identical (all 26 packed/box checks and every `d1.py` text case); `set_size=1` with 17 questions 519 → 316 ms, text 94 → 58 ms. |
| Share the VL tower's November 9 kernels ([VL experiments](../vl/EXPERIMENTS.md)) | Image 328 → 213 ms, mixed media 385 → 262 ms; probabilities move at most 2.4e-4 against the previous build (gate 0.02), discrete answers and boxes unchanged. |
| Run every pack of a request in one pass, up to `--context` rows: the projections read the weights once for all packs, while convolution, RoPE positions and attention run per pack from empty state | Each pack's sums are identical to its own pass (checked with split-K off: batched and separate packs byte-identical). `set_size=1` text 47 → 26 ms, 17 questions 84 → 66 ms, the same 17 at `set_size=1` 259 → 86 ms. |
| No split-K in decision passes | Split-K splits the last partial wave of tiles, so a pack's sums depended on the packs sharing its pass (`d1_packed.py` caught the 17-question tail moving 1e-4 from its standalone answer). Without it, every prompt's answer is independent of the others in its request. Answers move at most 6.3e-4 from the previous build, the same as reordering that build's own split; discrete answers are unchanged. |
| 128-row, four-wave projection tiles when an unsplit launch would have fewer than 56 tiles of 256 rows | Same sums as the 256-row tiles (byte-identical responses, LFM3 TTS audio included). Recovers the small passes split-K used to fill: 51-token prompts 17.1 → 15.6 ms. The down projection (K = 7680) remains below its split speed at 100–250 rows. |
| Project only the read vocabulary for single-slot questions too | The one-row GEMV is per output row, so the compact head returns the full head's values bit for bit and skips a 278 MB read (1.2 ms) per question. |
| Keep the vision tower's activations between pictures up to 4096 patches, and take 64 × 128 tiles for long inputs when 128 × 128 ones would fill fewer waves | Byte-identical embeddings. One picture 124 → 107 ms (about 3 ms of allocation, 4 ms of GEMM tiles). |
| Normalize patches through a 256-entry table and resize pictures on several threads with 32-bit sums (as Pillow) | Byte-identical pixels and patches; a 12-megapixel photograph 764 → 700 ms. |
| Rejected: deeper weight prefetch in `RegQ8Gemm` | At 16–124 rows the projections stream weights at 100–150 GB/s of 240. Prefetching two to four K blocks ahead (with the activation loads issued first so the LDS commit does not drain the ring) spilled at 256 VGPRs and was not faster; fewer accumulators (32-token tiles) did not raise occupancy above one block per unit. |
| Rejected: fixed in-block K groups for the down projection | Four wave groups adding their K quarters in order keep sums independent of the pass and win 10–20% below 64 rows, but are twice as slow at 2048 rows; the group count cannot vary with the pass without changing sums. |

Current measurements are in [BENCHMARKS.md](BENCHMARKS.md); numerical and API
limits are in [QUALITY.md](QUALITY.md).
