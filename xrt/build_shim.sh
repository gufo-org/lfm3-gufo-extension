#!/usr/bin/env bash
# Builds libflm_lfm3_shim.so and flm_lfm3_bench against a FastFlowLM checkout
# (prebuilt XRT engines in src/lib/xrt) and an XRT install.
# Usage: build_shim.sh <fastflowlm-checkout> <xrt-prefix> <output-dir>
# Needs g++, libuuid, libdrm and pkg-config on the path.
set -euo pipefail
F=$(realpath "$1"); X=$(realpath "$2"); OUT=$(realpath -m "$3")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
g++ -O2 -std=gnu++20 -mavx -mavx2 -fPIC -shared -DDISABLE_ABI_CHECK=1 \
  -DCMAKE_INSTALL_PREFIX="\"$F/src\"" -DCMAKE_XCLBIN_PREFIX="\"$F/src\"" \
  -I"$F/src/include" -I"$X/include" -isystem "$(pkg-config --variable=includedir libdrm)" \
  "$HERE/flm_lfm3_shim.cpp" "$F/src/common/utils.cpp" \
  -L"$F/src/lib/xrt" -llfm3_npu -lq4_npu_eXpress -lgemm -lmha -ldequant -llm_head \
  -L"$X/lib" -lxrt_coreutil -Wl,-rpath,"$F/src/lib/xrt:$X/lib" -o "$OUT/libflm_lfm3_shim.so"
g++ -O2 -std=c++20 "$HERE/flm_lfm3_bench.cpp" -L"$OUT" -lflm_lfm3_shim -Wl,-rpath,"$OUT" -o "$OUT/flm_lfm3_bench"
echo "$OUT/libflm_lfm3_shim.so"
