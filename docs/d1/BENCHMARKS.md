# d1-Omni-2B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Official `Q8_0` backbone and
BF16 projector (`c53ae0b1`), context 8192. Gufo `bd6b0fdd` with this
extension (November 11, 2026), release build, `gufo serve decision`.
Each cell is one warm HTTP request, including transport, media processing,
forward passes and readout; output tokens are always zero and loading is
excluded. Positive gain favors Gufo. **TODO** means unmeasured: the llama.cpp
reference (b12205 or newer, `/v1/systemone`) has not been run yet.

## One question per set

`set_size=1`: every question is its own prompt. Input tokens sum the
separate question prompts. Fixtures from
[d1.py](../../tests/functional/d1.py), with COCO `val2017/000000039769.jpg`
as the picture; the fastest of five warm requests.

| Request | Input tokens | Gufo ms | llama.cpp ms | Gain |
| --- | ---: | ---: | ---: | ---: |
| text | 156 | 25.5 | TODO | TODO |
| image | 665 | 143.7 | TODO | TODO |
| audio | 326 | 55.7 | TODO | TODO |
| json-unicode | 57 | 15.8 | TODO | TODO |
| literal-controls | 51 | 15.6 | TODO | TODO |
| long-state | 734 | 89.1 | TODO | TODO |
| mixed-media | 928 | 186.9 | TODO | TODO |
| spoken-instructions | 158 | 35.7 | TODO | TODO |

## Packed sets and boxes

Default packing reads questions in sets of 16. Grouping changes the trained
function as well as the repeated state work, so these rows are not an
identical-logit comparison with `set_size=1`. Fixtures from
[d1_packed.py](../../tests/functional/d1_packed.py), with
COCO `val2017/000000039769.jpg` as the picture; the second of two runs.

| Request | Input tokens | Gufo ms |
| --- | ---: | ---: |
| 17 text questions, default packing | 436 | 66.2 |
| The same 17, `set_size=1` | 646 | 85.5 |
| 16 text questions, one set | 398 | 54.6 |
| One-picture `boxes` | 571 | 138.3 |
| Audio state | 180 | 38.6 |
| Spoken instructions | 181 | 37.8 |
| Picture, audio and text | 698 | 168.9 |

Throughput, high-resolution pictures and maximum context are not measured.
