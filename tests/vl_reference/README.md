# Independent LFM3-VL reference

`reference.py` verifies patches, grids, template token IDs, vision/projector
embeddings and full next-token logits against unmodified LiquidAI classes.
It needs a local snapshot of `LiquidAI/LFM3-VL-2B` at
`ba44bd51db20a2753f28c273a1571ccaa210d954`, transformers 5.21.0, Torch,
torchvision, Pillow and llama.cpp's `gguf-py`. Reference source files are
hash checked. GGUF tensors are strictly mapped into FP32 teacher modules;
no Gufo numerical operators execute in the teacher.

```sh
cmake --build --preset gpu-test --target lfm3_vl_probe  # with -DGUFO_EXTENSION_DIRS
python3 lfm3-gufo-extension/tests/vl_reference/reference.py \
  --model models/lfm3-vl/LFM3-VL-2B-Q8_0.gguf \
  --mmproj models/lfm3-vl/mmproj-LFM3-VL-2B-BF16.gguf \
  --reference-dir /path/to/pinned-LFM3-VL-2B-snapshot \
  --probe build/gpu-test/extensions/build/lfm3/build/lfm3_vl_probe \
  --image /path/to/reviewed-cats.png --output artifacts/lfm3-vl/reference
```

The named image should be the reviewed two-cat fixture when reproducing the
retained report. Other images create a new qualification history. Missing
weights, source, packages or image files fail; they are not quality passes.
The native probe, Python teacher and their tensors stay outside hosted CI.
