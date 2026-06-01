#!/usr/bin/env python3
"""Unified streaming decode-TPS bench for any OpenAI-compatible /v1/chat/completions endpoint.

Methodology (matches project standard, see MEMORY.md):
  - streaming, measure TTFT (time to first content token)
  - decode_TPS = completion_tokens / (wall - ttft)
  - chat_template_kwargs:{enable_thinking:False}
  - Qwen non-thinking coding sampling: temp=0.6 top_p=0.95 top_k=20 min_p=0

Usage:
  python stream_bench.py --url http://localhost:8081 --model qwen3.6-27b --label genesis-mtp
  python stream_bench.py --url http://localhost:8000 --model qwen3.6-27b --label luce-dflash --tsv out.tsv
"""
import argparse, json, time, sys, urllib.request

PROMPTS = {
    "prose": "Write a concise 800-token explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs.",
    "code": "Write a complete TypeScript implementation of a binary search tree with insert, delete, search, and in-order traversal methods. Include unit tests for each method. Aim for 800 tokens of code.",
    "json": "Output a JSON array of 40 fictional employee records, each with fields: id, name, department, salary, hire_date, manager_id, skills (array of 3 strings). Valid JSON only, no prose.",
}

def run_one(url, model, key, prompt, max_tokens, no_think):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "stream": True, "stream_options": {"include_usage": True},
    }
    if no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=data,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    t0 = time.time(); ttft = None; ntok = 0; usage_completion = None; chars = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            ch = obj.get("choices") or []
            if ch:
                delta = ch[0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    ntok += 1
                    chars += len(piece)
            if obj.get("usage"):
                usage_completion = obj["usage"].get("completion_tokens")
    wall = time.time() - t0
    completion = usage_completion or ntok
    decode_wall = max(wall - (ttft or 0), 1e-6)
    return {
        "ttft": ttft or 0.0, "wall": wall, "tokens": completion,
        "decode_tps": completion / decode_wall, "chars": chars,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default="x")
    ap.add_argument("--label", default="run")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--prompts", default="prose,code,json")
    ap.add_argument("--no-think", action="store_true", default=True)
    ap.add_argument("--think", dest="no_think", action="store_false")
    ap.add_argument("--tsv", default=None)
    ap.add_argument("--warmup", action="store_true", default=True)
    args = ap.parse_args()

    if args.warmup:
        try:
            run_one(args.url, args.model, args.key, "Say hi.", 8, args.no_think)
        except Exception as e:
            print(f"warmup failed: {e}", file=sys.stderr)

    rows = []
    for name in args.prompts.split(","):
        name = name.strip()
        if name not in PROMPTS:
            continue
        r = run_one(args.url, args.model, args.key, PROMPTS[name], args.max_tokens, args.no_think)
        rows.append((name, r))
        print(f"[{args.label}/{name}] decode={r['decode_tps']:.2f} t/s  "
              f"ttft={r['ttft']*1000:.0f}ms  tokens={r['tokens']}  wall={r['wall']:.2f}s")

    if rows:
        avg = sum(r["decode_tps"] for _, r in rows) / len(rows)
        print(f"[{args.label}] AVG decode = {avg:.2f} t/s over {len(rows)} prompts")
    if args.tsv:
        with open(args.tsv, "a") as f:
            for name, r in rows:
                f.write(f"{args.label}\t{name}\t{r['decode_tps']:.2f}\t{r['ttft']*1000:.0f}\t{r['tokens']}\t{r['wall']:.2f}\n")

if __name__ == "__main__":
    main()
