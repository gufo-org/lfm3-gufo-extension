#!/usr/bin/env bash
# Writes the NPU bundle `--npu` loads for an LFM3-family GGUF (LFM3-1.7B,
# LFM3-VL-2B, LFM3-Audio-2B backbones): <model>.npu beside the GGUF.
#
# Usage: npu/iron/build.sh <model.gguf> [build dir]
#
# Needs an mlir-aie (IRON) Python with Peano, and XRT's Python bindings
# (pyxrt) on PYTHONPATH. The programs (xclbins) depend only on the LFM3
# shapes, so the build dir caches them across models; compiling them the
# first time takes about 15 minutes. Each bundle holds about 6 GB of
# repacked weights.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
gguf=$(realpath "$1")
build=$(realpath -m "${2:-${XDG_CACHE_HOME:-$HOME/.cache}/gufo/lfm3-npu}")
out="${gguf%.gguf}.npu"
mkdir -p "$build"

if [ ! -f "$build/decode.xclbin" ]; then
  (cd "$here" && python3 lfm3_token.py --xclbin-path "$build/decode.xclbin") | tee "$build/decode.log"
  so=$(sed -n 's/^dispatch library: //p' "$build/decode.log" | tail -1)
  cp "$so" "$build/decode_dispatch.so"
fi
# prefill programs: rows M with the LM head over the rows from `head`
for spec in "h0 1024 0" "h512 1024 512" "s512 512 0" "s128 128 0"; do
  set -- $spec
  [ -f "$build/$1.xclbin" ] || (cd "$here/prefill" && python3 model.py "$build/$1" 24 "$2" "$3")
done

tmp="$out.tmp"
rm -rf "$tmp"
python3 "$here/bundle.py" "$gguf" "$tmp" "$build/decode.xclbin" "$build/decode_dispatch.so" \
  "$build/h0.xclbin" "$build/h0.bin" "$build/h512.bin" \
  --small "$build/s512.xclbin" "$build/s512.bin" --tiny "$build/s128.xclbin" "$build/s128.bin"
rm -rf "$out"
mv "$tmp" "$out"
echo "NPU bundle: $out"
