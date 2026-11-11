#!/usr/bin/env bash
# Starts gufo serve with the NPU backbone, sends one chat request, stops.
SP=/tmp/claude-1000/-home-mixer-gufo/8e84d6e4-1c1c-4d14-ba0b-69045e8d2aca/scratchpad
G=$SP/gufo-npu-build/gufo; M=${MODEL:-/persist/models/store/LFM3-1.7B/LFM3-1.7B-Q8_0.gguf}
systemd-run --user --scope -q --unit=lfm3-npu-serve -p MemoryMax=24G -p TasksMax=512 -E GUFO_LFM3_DEVICE=npu -E LFM3_NPU_LIBRARY=$SP/npu-host/liblfm3_npu.so -- $G serve --port 18096 --sessions ${SESSIONS:-1} llm --model $M --context 4096 > $SP/res/npu-serve.log 2>&1 &
python3 - "$@" <<'PY'
import json, sys, time, urllib.request
t=time.time()
while time.time()-t<300:
    try:
        d=json.loads(urllib.request.urlopen('http://127.0.0.1:18096/v1/models',timeout=2).read()); print('ready', round(time.time()-t,1), d['data'][0]['id']); break
    except Exception: time.sleep(1)
prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France? Answer in one sentence."
n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
body={"model":d['data'][0]['id'],"max_tokens":n,"temperature":0,"stream":True,"messages":[{"role":"user","content":prompt}]}
t=time.perf_counter(); stamps=[]; text=''; reasoning=''
req=urllib.request.Request('http://127.0.0.1:18096/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=900) as r:
    for raw in r:
        line=raw.decode().strip()
        if not line.startswith('data:') or line[5:].strip()=='[DONE]': continue
        for c in json.loads(line[5:]).get('choices',[]):
            dd=c.get('delta',{})
            if dd.get('content') or dd.get('reasoning_content'):
                stamps.append(time.perf_counter()); text+=dd.get('content') or ''; reasoning+=dd.get('reasoning_content') or ''
ttft=stamps[0]-t if stamps else 0
rate=(len(stamps)-1)/(stamps[-1]-stamps[0]) if len(stamps)>1 else 0
print(f'ttft {ttft:.2f}s tokens {len(stamps)} decode {rate:.2f} tok/s')
print('reasoning:', repr(reasoning[:400])); print('content:', repr(text[:600]))
PY
systemctl --user stop lfm3-npu-serve.scope
