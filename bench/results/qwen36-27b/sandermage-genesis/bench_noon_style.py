#!/usr/bin/env python3
"""Bench in noonghunna style: streaming + enable_thinking:False + decode_TPS metric."""
import json, sys, time, urllib.request, statistics as s

URL = sys.argv[1]
MODEL = sys.argv[2]
PROMPT = sys.argv[3]
MAX_TOKENS = int(sys.argv[4])
WARMUPS = int(sys.argv[5])
RUNS = int(sys.argv[6])

def run_once():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_send = time.time()
    ttft = None
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode("utf-8", errors="ignore").rstrip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ttft is None:
                # First content chunk
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    ttft = time.time() - t_send
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", 0)
    wall = time.time() - t_send
    return wall, ttft if ttft else 0, completion_tokens

print(f"Warming up ({WARMUPS}x)...", flush=True)
for _ in range(WARMUPS):
    run_once()

walls, ttfts, tokens_list = [], [], []
print(f"Measured runs ({RUNS}x):", flush=True)
for i in range(RUNS):
    w, t, c = run_once()
    walls.append(w); ttfts.append(t); tokens_list.append(c)
    wall_tps = c / w if w > 0 else 0
    decode_tps = c / (w - t) if (w - t) > 0 else 0
    print(f"  run {i+1}: wall={w:.2f}s ttft={t:.3f}s tokens={c} wall_tps={wall_tps:.2f} decode_tps={decode_tps:.2f}", flush=True)

mean_wall = s.mean(walls)
mean_ttft = s.mean(ttfts)
mean_tokens = s.mean(tokens_list)
mean_wall_tps = mean_tokens / mean_wall
mean_decode_tps = mean_tokens / (mean_wall - mean_ttft)

print(f"\nSummary ({RUNS} runs, {int(mean_tokens)} avg tokens):")
print(f"  mean wall:   {mean_wall:.2f}s")
print(f"  mean ttft:   {mean_ttft:.3f}s")
print(f"  wall_TPS:    {mean_wall_tps:.2f}")
print(f"  decode_TPS:  {mean_decode_tps:.2f}  <-- noonghunna's headline metric")
