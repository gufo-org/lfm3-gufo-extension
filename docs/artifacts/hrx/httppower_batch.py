#!/usr/bin/env python3
"""Aggregate decode speed and socket power with C concurrent streams.

Usage: httppower_batch.py <base-url> <model> <gen-tokens> <seconds> <streams>
Each stream loops greedy streamed requests of G tokens; every streamed token
inside the measured window counts, over the window's wall time.
"""
import glob
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
G, SECONDS, C = int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
SENSOR = glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*/power1_average")[0]


def watts():
    return int(open(SENSOR).read()) / 1e6


def decode_once(i=None):
    body = {"model": MODEL, "max_tokens": G, "temperature": 0, "stream": True,
            "messages": [{"role": "user", "content":
                          "Write a long, detailed essay about rivers and cities. " + str(time.time_ns())}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for c in json.loads(line[5:]).get("choices", []):
                d = c.get("delta", {})
                if (d.get("content") or "") + (d.get("reasoning_content") or ""):
                    n += 1
                    if i is not None:
                        counts[i] += 1
    return n


decode_once()
samples = []
for _ in range(int(SECONDS / 0.05)):
    samples.append(watts()); time.sleep(0.05)
idle = statistics.mean(samples)
stop = threading.Event()
counts = [0] * C


def worker(i):
    while not stop.is_set():
        decode_once(i)


threads = [threading.Thread(target=worker, args=(i,)) for i in range(C)]
for t in threads:
    t.start()
time.sleep(3)  # all streams in flight
base = sum(counts)
power = []
start = time.time()
while time.time() - start < SECONDS:
    power.append(watts()); time.sleep(0.05)
elapsed = time.time() - start
tokens = sum(counts) - base
stop.set()
for t in threads:
    t.join()
p = statistics.mean(power)
rate = tokens / elapsed
print(f"decode x{C} {rate:8.1f} tok/s  socket {p:5.1f} W  (+{p - idle:5.1f} W over idle {idle:.1f} W)  "
      f"{p / rate * 1000:7.1f} mJ/token total  {(p - idle) / rate * 1000:7.1f} mJ/token above idle")
