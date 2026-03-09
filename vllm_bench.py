#!/usr/bin/env python3
"""vLLM server benchmark script - measures TPS for various request types."""

import json
import time
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:8080"

def timed_completion(prompt, max_tokens=128, temperature=0.7, top_p=0.8, stream=False):
    """Send a chat completion request and measure performance."""
    payload = {
        "model": "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

    start = time.perf_counter()

    if stream:
        resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True)
        resp.raise_for_status()
        tokens = 0
        first_token_time = None
        text = ""
        for line in resp.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        tokens += 1
                        text += content
        elapsed = time.perf_counter() - start
        ttft = (first_token_time - start) if first_token_time else elapsed
        return {
            "tokens": tokens,
            "elapsed": elapsed,
            "tps": tokens / elapsed if elapsed > 0 else 0,
            "ttft": ttft,
            "text": text[:100],
        }
    else:
        resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.perf_counter() - start
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        text = data["choices"][0]["message"]["content"]
        return {
            "tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "elapsed": elapsed,
            "tps": completion_tokens / elapsed if elapsed > 0 else 0,
            "text": text[:100],
        }

def benchmark_single(label, prompt, max_tokens, **kwargs):
    """Run a single benchmark and print results."""
    print(f"\n--- {label} ---")
    result = timed_completion(prompt, max_tokens=max_tokens, **kwargs)
    print(f"  Tokens: {result['tokens']} | Time: {result['elapsed']:.2f}s | TPS: {result['tps']:.1f}")
    if 'ttft' in result:
        print(f"  TTFT: {result['ttft']*1000:.0f}ms")
    print(f"  Preview: {result['text']}...")
    return result

def benchmark_concurrent(n_requests, prompt, max_tokens):
    """Run concurrent requests and measure aggregate throughput."""
    print(f"\n--- {n_requests} concurrent requests ({max_tokens} tokens each) ---")

    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=n_requests) as executor:
        futures = [
            executor.submit(timed_completion, prompt, max_tokens=max_tokens)
            for _ in range(n_requests)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    total_elapsed = time.perf_counter() - start
    total_tokens = sum(r["tokens"] for r in results)
    aggregate_tps = total_tokens / total_elapsed if total_elapsed > 0 else 0
    avg_tps = sum(r["tps"] for r in results) / len(results)

    print(f"  Total tokens: {total_tokens} | Wall time: {total_elapsed:.2f}s")
    print(f"  Aggregate TPS: {aggregate_tps:.1f} | Avg per-request TPS: {avg_tps:.1f}")
    return {"aggregate_tps": aggregate_tps, "avg_tps": avg_tps, "total_tokens": total_tokens, "elapsed": total_elapsed}

def main():
    # Check server is up
    try:
        resp = requests.get(f"{BASE_URL}/v1/models")
        resp.raise_for_status()
        models = resp.json()
        print(f"Server ready. Models: {[m['id'] for m in models['data']]}")
    except Exception as e:
        print(f"Server not ready: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("vLLM Benchmark - Qwen3.5-35B-A3B-GPTQ-Int4")
    print("="*60)

    results = {}

    # Warmup
    print("\nWarming up...")
    timed_completion("Hello", max_tokens=16)

    # Single request benchmarks
    results["short_64"] = benchmark_single(
        "Short generation (64 tokens)",
        "Write a haiku about programming.",
        max_tokens=64
    )

    results["medium_256"] = benchmark_single(
        "Medium generation (256 tokens)",
        "Explain how a hash table works in 200 words.",
        max_tokens=256
    )

    results["long_512"] = benchmark_single(
        "Long generation (512 tokens)",
        "Write a Python function to implement merge sort with detailed comments.",
        max_tokens=512
    )

    # Streaming benchmark
    results["stream_256"] = benchmark_single(
        "Streaming (256 tokens)",
        "Explain recursion with examples.",
        max_tokens=256,
        stream=True
    )

    # Concurrent benchmarks
    results["concurrent_5"] = benchmark_concurrent(
        5, "Tell me a short joke.", max_tokens=128
    )

    results["concurrent_10"] = benchmark_concurrent(
        10, "Tell me a fun fact.", max_tokens=128
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Test':<35} {'Tokens':>7} {'Time':>7} {'TPS':>8}")
    print("-"*60)
    for key in ["short_64", "medium_256", "long_512", "stream_256"]:
        r = results[key]
        print(f"{key:<35} {r['tokens']:>7} {r['elapsed']:>6.2f}s {r['tps']:>7.1f}")
    for key in ["concurrent_5", "concurrent_10"]:
        r = results[key]
        print(f"{key:<35} {r['total_tokens']:>7} {r['elapsed']:>6.2f}s {r['aggregate_tps']:>7.1f}")

if __name__ == "__main__":
    main()
