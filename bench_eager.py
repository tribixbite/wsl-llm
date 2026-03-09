#!/usr/bin/env python3
"""Benchmark vLLM enforce-eager mode."""
import requests, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "http://localhost:8080"
MODEL = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"

def bench(prompt, max_tokens, label):
    start = time.perf_counter()
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": f"/no_think {prompt}"}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
    })
    r.raise_for_status()
    elapsed = time.perf_counter() - start
    toks = r.json()["usage"]["completion_tokens"]
    tps = toks / elapsed
    print(f"{label}: {toks} tok in {elapsed:.2f}s = {tps:.1f} t/s")
    return tps

def stream_bench(prompt, max_tokens, label):
    start = time.perf_counter()
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": f"/no_think {prompt}"}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "stream": True,
    }, stream=True)
    r.raise_for_status()
    first_token = None
    tokens = 0
    for line in r.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                chunk = json.loads(decoded[6:])
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                if content:
                    if first_token is None:
                        first_token = time.perf_counter()
                    tokens += 1
    elapsed = time.perf_counter() - start
    ttft = (first_token - start) * 1000 if first_token else 0
    tps = tokens / elapsed if elapsed > 0 else 0
    print(f"{label}: {tokens} tok in {elapsed:.2f}s = {tps:.1f} t/s | TTFT: {ttft:.0f}ms")
    return {"tps": tps, "ttft": ttft}

print("=== TP=1 enforce-eager, ctx=16k, prefix_caching=on ===")
print("(No CUDA graph warmup needed)")
print()

# No warmup needed with enforce-eager!
bench("Write a haiku about coding.", 128, "128tok")
bench("Explain hash tables briefly.", 256, "256tok")
bench("Write merge sort in Python with detailed comments.", 512, "512tok")
bench("Write a comprehensive guide to Python decorators.", 1024, "1024tok")

print()
print("Streaming TTFT tests:")
stream_bench("Say hello", 64, "stream_64")
stream_bench("Explain recursion", 256, "stream_256")

print()
for n in [5, 10]:
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(bench, f"Tell joke #{i}", 128, f"c{n}_{i}") for i in range(n)]
        [f.result() for f in as_completed(futs)]
    wall = time.perf_counter() - start
    print(f">> {n} concurrent: aggregate {128*n/wall:.1f} t/s in {wall:.2f}s")
