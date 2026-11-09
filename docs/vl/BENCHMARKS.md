# LFM3-VL 2B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Official `Q8_0` backbone and
BF16 projector (`b36554cf`). HTTP, greedy, thinking off. Gufo `365cc5a5` with
this extension (November 9, 2026), release build. Positive gain favors Gufo. **TODO** means
unmeasured: the llama.cpp reference (b12205 or newer) has not been run yet.
[Quality](QUALITY.md) · [Benchmark configuration](artifacts/bench.json)

## Image encoder

Prompt time of a request carrying one gradient PNG and a one-line question,
one output token, prompt cache off: image decoding and resizing, the vision
tower, the projector and the prefill of the image and text tokens. One warmup,
then the mean of three requests per size.

<!-- bench:image-encoder -->
| LFM3-VL 2B Q8_0 vision<br>RGB image | Merged tokens | Gufo ms | llama.cpp ms | Gain |
| --- | ---: | ---: | ---: | ---: |
| 256×256 | 64 | 71.2 ± 0.5 | TODO | TODO |
| 512×512 | 256 | 134.8 ± 0.9 | TODO | TODO |
| 768×768 | 576 | 258.3 ± 2.1 | TODO | TODO |
| 1024×1024 | 1024 | 438.3 ± 7.0 | TODO | TODO |
<!-- /bench -->

![Image encoder](artifacts/charts/image-encoder.svg)

## Photographs

The same request with a photograph at its native pixel budget, one warmup,
then the fastest of three (one request for 4032×3024). These images are not
resized, so the vision tower sees 16384 and 47376 patches.

| JPEG | Merged tokens | Gufo ms |
| --- | ---: | ---: |
| 2048×2048 | 4113 | 2380 |
| 4032×3024 | 11861 | 15073 |

At 4032×3024, vision attention takes 11.0 s of the request and the backbone
prefill about 2.1 s. Text-only speed of this backbone and throughput beside
other sessions are not measured.
