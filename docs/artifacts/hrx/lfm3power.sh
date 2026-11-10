#!/usr/bin/env bash
# LFM3-1.7B Q8 on the GPU through `gufo serve`, stock HIP or HRX: per-phase
# speed and socket power via httppower.py. Usage: lfm3power.sh <hip|hrx> <tag>
set -uo pipefail
SP=/tmp/claude-1000/-home-mixer-gufo/8e84d6e4-1c1c-4d14-ba0b-69045e8d2aca/scratchpad
RT=$1; TAG=$2
G=$SP/result-gufo-lfm3/bin/gufo
M=/persist/models/store/LFM3-1.7B/LFM3-1.7B-Q8_0.gguf
if pgrep -x .gufo-wrapped >/dev/null || pgrep -x loom-serve >/dev/null; then
  echo "model process already running; refusing" >&2; exit 2
fi
temp() { cat /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input | head -1; }
for _ in $(seq 1 120); do [ "$(temp)" -le 38000 ] && break; sleep 5; done
E=(); [ "$RT" = hrx ] && E=(LD_LIBRARY_PATH=$SP/result-hrx-pool/lib)
unit=lfm3-power-$TAG
systemd-run --user --scope -q --unit="$unit" -p MemoryMax=32G -p TasksMax=256 -- \
  env "${E[@]}" "$G" serve --port 18096 --sessions 1 llm --model "$M" --context 4096 \
  > "$SP/res/$TAG.serve.log" 2>&1 &
model=$(python3 - <<'EOF'
import json, time, urllib.request
t = time.time()
while time.time() - t < 300:
    try:
        d = json.loads(urllib.request.urlopen('http://127.0.0.1:18096/v1/models', timeout=2).read())
        print(d['data'][0]['id']); break
    except Exception:
        time.sleep(1)
EOF
)
echo "model id: $model" > "$SP/res/$TAG.log"
sleep 15
python3 "$SP/httppower.py" http://127.0.0.1:18096 "$model" 2000 256 20 >> "$SP/res/$TAG.log" 2>&1
systemctl --user stop "$unit.scope"
for _ in $(seq 1 30); do pgrep -x .gufo-wrapped >/dev/null || break; sleep 1; done
cat "$SP/res/$TAG.log"
