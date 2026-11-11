#!/usr/bin/env bash
# Socket power of a Loom NPU benchmark against the idle socket, both measured
# with no model process and a cooled GPU. Usage:
#   npupower.sh <file.loom> <@benchmark> <columns> <tag> [seconds]
set -uo pipefail
SP=/tmp/claude-1000/-home-mixer-gufo/8e84d6e4-1c1c-4d14-ba0b-69045e8d2aca/scratchpad
T=$SP/hrx-pin-build/loom/src/loom/tools
SRC=$1; BENCH=$2; COLS=$3; TAG=$4; SECS=${5:-20}
P=/sys/class/hwmon/hwmon1/power1_average
temp() { cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input | head -1; }
if pgrep -x .gufo-wrapped >/dev/null || pgrep -x loom-serve >/dev/null; then
  echo "model process running; refusing" >&2; exit 2
fi
for _ in $(seq 1 120); do [ "$(temp)" -le 38000 ] && break; sleep 5; done
sleep 30
sample() {  # seconds -> mean W
  python3 - "$1" "$P" <<'PY'
import sys, time
n = int(float(sys.argv[1]) * 4); s = []
for _ in range(n):
    s.append(int(open(sys.argv[2]).read()) / 1e6); time.sleep(0.25)
print(f"{sum(s) / len(s):.2f}")
PY
}
idle=$(sample 15)
systemd-run --user --scope -q -p MemoryMax=16G -- "$T/loom-benchmark/loom-benchmark" "$SRC" \
  --device="xdna://0?columns=$COLS" --benchmark="$BENCH" --measure=dispatch_complete \
  --batch-size=1 --input-ring-count=1 --min-time-ms=$((SECS * 1000 + 5000)) --max-batches=1000000 \
  --output="$SP/res/$TAG.json" > "$SP/res/$TAG.bench.log" 2>&1 &
pid=$!
sleep 4  # compile and warm up
load=$(sample "$SECS")
wait $pid
p50=$(python3 -c "import json;d=json.load(open('$SP/res/$TAG.json'));print(d['work_items'][0]['measurement']['batch_timing_ns']['p50'])")
echo "$TAG idle ${idle} W load ${load} W p50 ${p50} ns" | tee "$SP/res/$TAG.log"
