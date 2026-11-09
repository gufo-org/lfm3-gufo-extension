# d1-Omni-2B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Official `Q8_0` backbone and
BF16 projector (`c53ae0b1`), context 8192. Gufo `365cc5a5` with this
extension (November 9, 2026), release build, `gufo serve decision`.
Each cell is one warm HTTP request, including transport, media processing,
forward passes and readout; output tokens are always zero and loading is
excluded. Positive gain favors Gufo. **TODO** means unmeasured: the llama.cpp
reference (b12205 or newer, `/v1/systemone`) has not been run yet.

## One question per set

`set_size=1`: every question is its own prompt. Input tokens sum the
separate question prompts. Fixtures from
[d1.py](../../tests/functional/d1.py), with COCO `val2017/000000039769.jpg`
as the picture; the fastest of three warm requests.

| Request | Input tokens | Gufo ms | llama.cpp ms | Gain |
| --- | ---: | ---: | ---: | ---: |
| text | 156 | 58.1 | TODO | TODO |
| image | 665 | 212.8 | TODO | TODO |
| audio | 326 | 85.1 | TODO | TODO |
| json-unicode | 57 | 20.2 | TODO | TODO |
| literal-controls | 51 | 19.5 | TODO | TODO |
| long-state | 734 | 137.0 | TODO | TODO |
| mixed-media | 928 | 262.4 | TODO | TODO |
| spoken-instructions | 158 | 47.4 | TODO | TODO |

## Packed sets and boxes

Default packing reads questions in sets of 16. Grouping changes the trained
function as well as the repeated state work, so these rows are not an
identical-logit comparison with `set_size=1`. Fixtures from
[d1_packed.py](../../tests/functional/d1_packed.py), with
COCO `val2017/000000039769.jpg` as the picture; the second of two runs.

| Request | Input tokens | Gufo ms |
| --- | ---: | ---: |
| 17 text questions, default packing | 436 | 116.9 |
| The same 17, `set_size=1` | 646 | 308.7 |
| 16 text questions, one set | 398 | 83.7 |
| One-picture `boxes` | 571 | 189.3 |
| Audio state | 180 | 51.8 |
| Spoken instructions | 181 | 50.8 |
| Picture, audio and text | 698 | 228.5 |

Throughput, high-resolution pictures and maximum context are not measured.
