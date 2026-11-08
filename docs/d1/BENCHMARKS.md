# d1-Omni-2B benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Official `Q8_0` backbone and
BF16 projector (`c53ae0b1`), context 8192. Gufo `12f8d8ab` with this
extension (November 8, 2026), Nix development build, `gufo serve decision`.
Each cell is one warm HTTP request, including transport, media processing,
forward passes and readout; output tokens are always zero and loading is
excluded. Positive gain favors Gufo. **TODO** means unmeasured: the llama.cpp
reference (b12205 or newer, `/v1/systemone`) has not been run yet.

## One question per set

`set_size=1`: every question is its own prompt. Input tokens sum the
separate question prompts. Fixtures from
[d1.py](../../tests/functional/d1.py); a repeated identical request stayed
within 1.9 ms of each value.

| Request | Input tokens | Gufo ms | llama.cpp ms | Gain |
| --- | ---: | ---: | ---: | ---: |
| text | 156 | 111.8 | TODO | TODO |
| image | 665 | 351.6 | TODO | TODO |
| audio | 326 | 138.4 | TODO | TODO |
| json-unicode | 57 | 38.3 | TODO | TODO |
| literal-controls | 51 | 37.8 | TODO | TODO |
| long-state | 734 | 172.1 | TODO | TODO |
| mixed-media | 928 | 418.9 | TODO | TODO |
| spoken-instructions | 158 | 82.2 | TODO | TODO |

## Packed sets and boxes

Default packing reads questions in sets of 16. Grouping changes the trained
function as well as the repeated state work, so these rows are not an
identical-logit comparison with `set_size=1`. Fixtures from
[d1_packed.py](../../tests/functional/d1_packed.py), with
COCO `val2017/000000039769.jpg` as the picture.

| Request | Input tokens | Gufo ms |
| --- | ---: | ---: |
| 17 text questions, default packing | 436 | 157.4 |
| The same 17, `set_size=1` | 646 | 618.7 |
| 16 text questions, one set | 398 | 108.4 |
| One-picture `boxes` | 571 | 309.5 |
| Audio state | 180 | 88.1 |
| Spoken instructions | 181 | 81.7 |
| Picture, audio and text | 698 | 354.5 |

Throughput, high-resolution pictures and maximum context are not measured.
