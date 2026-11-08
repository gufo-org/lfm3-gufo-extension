# LFM3 1.7B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Official `Q8_0` GGUF
(`b3f2ae5d`). HTTP, greedy, thinking off. Gufo `12f8d8ab` with this extension
(November 8, 2026). Positive gain favors Gufo. **TODO** means unmeasured: the
llama.cpp reference (b12205 or newer, the first release with LFM3) has not
been run yet.
[Quality and measurement details](QUALITY.md) · [Benchmark configuration](artifacts/bench.json)

## Single user, autoregressive

pp2048 / tg128; depth is the cached prefix in tokens.

<!-- bench:single-ar -->
| LFM3 1.7B Q8_0 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4714.18 | TODO | TODO | 101.47 | TODO | TODO |
| 4,096 | 3914.64 | TODO | TODO | 95.89 | TODO | TODO |
| 8,192 | 3310.11 | TODO | TODO | 91.46 | TODO | TODO |
| 12,288 | 2889.67 | TODO | TODO | 87.38 | TODO | TODO |
| 16,384 | 2613.98 | TODO | TODO | 83.84 | TODO | TODO |
| 32,768 | 1828.69 | TODO | TODO | 71.93 | TODO | TODO |
| 65,536 | 1151.18 | TODO | TODO | 55.42 | TODO | TODO |
| 131,072 | 657.22 | TODO | TODO | 38.51 | TODO | TODO |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
Every session is prefilled before timed decoding; throughput sums individual
request decode rates.

<!-- bench:multi-ar -->
| LFM3 1.7B Q8_0 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 101.53 | TODO | TODO |
| 2 | 186.95 | TODO | TODO |
| 4 | 314.39 | TODO | TODO |
| 6 | 411.24 | TODO | TODO |
| 8 | 475.90 | TODO | TODO |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Loading time

C1, context capacity 262144, AR. Cold model file (evicted from the page
cache and verified non-resident) to HTTP readiness.

<!-- bench:loading -->
| LFM3 1.7B Q8_0<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q8_0 | 4.42 | TODO | TODO |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak device memory in use reported by HIP.

<!-- bench:memory -->
| LFM3 1.7B Q8_0 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 11.98 | TODO | TODO |
| 16K prefix, pp4096 + tg128 | 12.61 | TODO | TODO |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)
