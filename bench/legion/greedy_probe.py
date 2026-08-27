#!/usr/bin/env python3
"""Emit a fixed greedy completion set from an OpenAI-compatible endpoint.

Used to test whether two server configurations produce identical output at
temperature 0. Any difference between two runs of the SAME config is engine
nondeterminism; a difference between configs (e.g. speculative decoding on vs
off) is only meaningful once same-config determinism is established.
"""
import argparse
import json
import urllib.request

SEP = "\n<<<===PROMPT-SEP===>>>\n"

PROMPTS = [
    "Write a Python function to merge two sorted lists. Code only.",
    "Explain in exactly three sentences what a B-tree is.",
    "Write a TypeScript type-guard that narrows unknown to Record<string,unknown>.",
    "List the first 12 prime numbers separated by commas.",
    "Write a bash one-liner that finds the 5 largest files under /var/log.",
]


def complete(url: str, prompt: str, max_tokens: int) -> str:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1337,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        obj = json.loads(resp.read())
    return obj["choices"][0]["message"].get("content") or ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=300)
    args = ap.parse_args()

    outputs = [complete(args.url, p, args.max_tokens) for p in PROMPTS]
    with open(args.out, "w") as fh:
        fh.write(SEP.join(outputs))
    print(f"{args.out}: {[len(o) for o in outputs]}")


if __name__ == "__main__":
    main()
