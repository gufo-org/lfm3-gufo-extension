# LFM3-VL integration decisions

- Share the identical LFM3 vision architecture with d1; preserve d1's own
  pixel budget. The existing d1 pixel/operator goldens and HTTP suite remain.
- Keep native static-image preprocessing and its full pixel range. Low
  detail uses a smaller aspect-ratio-aware budget with the native minimum.
- Feed image rows through causal backbone embedding prefill. The decision
  model's bidirectional forward path is not used for VL generation.
- Bind supplemental cache identity to processed image inputs and the full
  projector identity. Persistent compatibility includes the image ABI;
  identical token placeholders alone never identify a photo.
- Keep the Python teacher and raw tensor probes outside production. Source
  hashes and strict tensor mapping make their provenance reviewable.

## November 9 pass: vision tower

Q8_0 backbone, BF16 projector, release build of Gufo `365cc5a5`; prompt
milliseconds over HTTP against the extension at `810892b`. Accuracy is the
image-embedding error against an FP64 evaluation of the same encoder from
the GGUF weights and the probe's own patches (gradient 1024×1024, a
2048×2048 photograph, a 784×1168 PNG).

| Mechanism | Decision and evidence |
| --- | --- |
| Tiled FP32 bidirectional attention (the d1 kernel, generalized to D = 64 and no gate) instead of one wave per query streaming every key | Retained. Operator FP64 error at or below the old kernel's (3.8e-9 vs 4.5e-9 at 1200 rows). 1024×1024: 1173 → 722 ms. |
| BF16 weights on the matrix cores: FP32 activations split into three exact BF16 terms, each K32 block accumulated from zero then added in FP32 | Retained. Relative error 6.5e-8 to 1.3e-7 against FP64, below a sequential FP32 dot product's 2.8e-7 to 4.0e-7; one chain over all K was 10× worse (WMMA accumulation). About 10 effective TFLOPS versus 3.8 for the FP32 tile. hipBLAS BF16 reaches 5 raw TFLOPS here and rounds activations: rejected. |
| Summed patchifier through the tiled FP32 GEMM instead of 4096 per-row GEMVs | Retained: 25 ms → 1.7 ms at 1024×1024. |
| FP32 K/V in the tower (it keeps no cache) instead of F16 | Retained. Embedding error against FP64 drops 5–50× (2048×2048: max 7.1e-4 → 2.0e-5, rms 1.05e-5 → 3.3e-7) at no speed cost. |
| 16 (head, query) rows per wave for D = 64 | Retained, bit-identical: halves K/V tile traffic; 2048×2048 3.85 → 2.79 s. |
| Weight-row-major GEMM grid when FP32 activations exceed the 32 MiB MALL | Retained, bit-identical: 2048×2048 4.5 → 3.85 s. Smaller inputs keep token-major order (row-major was 2–3% slower at 1024×1024). |
| Matrix-core attention: Q, K, P and V as F16 high/low pairs, three WMMAs per product with the small terms in their own accumulators | Retained. Embedding error against FP64 below the tiled FP32 kernel's on all three images (rms 9.9e-7 vs 1.6e-6 at 1024×1024); 1024×1024 496 → 443 ms, 2048×2048 2.79 → 2.3 s. One accumulator chain per score was slower and less exact. |
| Pre-split K/V planes read as WMMA fragments straight from L2 | Rejected: 2.3 → 2.53 s at 2048×2048 without LDS staging and prefetch. |
| Split-K for small prompt projections and two-block-deep Q8 weight prefetch (shared backbone) | Rejected: pp32–pp2048 1–14% slower. |

Matched end to end (fastest of three, base `810892b` → this pass):
1024×1024 1163 → 442 ms, 2048×2048 JPEG 14253 → 2380 ms, 4032×3024 JPEG
185226 → 15073 ms (one request each). Greedy answers are unchanged. The
remaining attention time is VALU-bound (693 instructions for 48 WMMAs per
tile) with waves locked by two barriers per tile.

## November 11 pass: shared tower overheads

From the d1 decision work ([d1 decisions](../d1/EXPERIMENTS.md)); every change
keeps embeddings byte-identical (LFM3-VL greedy replies on a synthetic PNG and
the COCO two-cat photograph are unchanged).

| Mechanism | Decision and evidence |
| --- | --- |
| Reuse encode activations and stream across pictures up to 4096 patches | Retained: a 640×480 picture allocated about 80 MB per encode (about 3 ms with the free). Larger pictures still allocate their own. |
| 64 × 128 tiles for inputs over 1024 patches when they need fewer waves than 128 × 128 | Retained, same sums: 1200 patches 50.7 → 45 ms of GEMM; 1024×1024 and larger pictures keep 128 × 128. |
| Threaded separable resize, patch normalization by table | Retained, identical pixels: 12-megapixel input 764 → 700 ms end to end (d1). |

