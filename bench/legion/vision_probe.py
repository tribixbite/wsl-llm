#!/usr/bin/env python3
"""Verify multimodal serving through the OpenAI-compatible endpoint, and time it.

Uses images with known ground truth so a pass means the model actually read the
pixels rather than produced a plausible-sounding guess. Each case declares the
strings that must appear in the answer.

Also reports prefill/decode throughput, since an image costs a large number of
prompt tokens and that is the part people are surprised by.
"""

import argparse
import base64
import json
import mimetypes
import time
import urllib.request
from pathlib import Path

CASES = [
    {
        "image": "probe.png",
        "prompt": ("Read this image. Reply with exactly three lines:\n"
                   "1) the large heading text\n"
                   "2) the serial number\n"
                   "3) the shapes present, with their colours"),
        "must_include": ["LEGION", "5080", "QX-7741"],
        "should_include": ["red", "blue", "green", "circle", "square", "triangle"],
    },
    {
        "image": "chart.png",
        "prompt": ("This is a bar chart of decode tokens/sec. List each bar's label and its "
                   "value, then say which is highest."),
        "must_include": ["75"],
        "should_include": ["38", "44", "MTP", "exllamav3", "llama.cpp"],
    },
]


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def ask(url: str, image: Path, prompt: str, max_tokens: int, think: bool):
    body = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri(image)}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": bool(think)},
    }
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        obj = json.loads(r.read())
    wall = time.time() - t0
    msg = obj["choices"][0]["message"]
    return (msg.get("content") or ""), obj.get("usage", {}), wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--images", default="/tmp/vision")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sustained", action="store_true",
                    help="also time a long generation, since short answers are dominated by overhead")
    args = ap.parse_args()

    results, ok_all = [], True
    for case in CASES:
        img = Path(args.images) / case["image"]
        if not img.exists():
            print(f"SKIP {case['image']} (missing)")
            continue
        text, usage, wall = ask(args.url, img, case["prompt"], args.max_tokens, args.think)
        low = text.lower()
        missing = [k for k in case["must_include"] if k.lower() not in low]
        soft = [k for k in case["should_include"] if k.lower() in low]
        ok = not missing
        ok_all &= ok
        pt = usage.get("prompt_tokens", 0); ct = usage.get("completion_tokens", 0)
        print(f"\n=== {case['image']} — {'PASS' if ok else 'FAIL'} ===")
        print(f"  prompt_tokens={pt} (image dominates)  completion_tokens={ct}  wall={wall:.1f}s"
              f"  decode~{ct/max(wall,1e-6):.1f} t/s")
        if missing:
            print(f"  MISSING required: {missing}")
        print(f"  matched optional {len(soft)}/{len(case['should_include'])}: {soft}")
        print("  --- answer ---")
        print("  " + text.strip().replace("\n", "\n  ")[:800])
        results.append({"image": case["image"], "pass": ok, "missing": missing,
                        "matched_optional": soft, "prompt_tokens": pt,
                        "completion_tokens": ct, "wall_s": round(wall, 2),
                        "answer": text})

    if args.sustained:
        img = Path(args.images) / "chart.png"
        prompt = ("Describe this chart in exhaustive detail: axes, every bar, colours, "
                  "layout, spacing, and what an engineer should conclude. Be verbose; "
                  "write at least 400 words.")
        text, usage, wall = ask(args.url, img, prompt, 700, args.think)
        ct = usage.get("completion_tokens", 0); pt = usage.get("prompt_tokens", 0)
        print(f"\n=== sustained vision decode ===")
        print(f"  prompt_tokens={pt}  completion_tokens={ct}  wall={wall:.1f}s"
              f"  decode~{ct/max(wall,1e-6):.1f} t/s")
        results.append({"image": "chart.png(sustained)", "pass": True,
                        "prompt_tokens": pt, "completion_tokens": ct,
                        "wall_s": round(wall, 2), "answer": text})

    print(f"\n=== VISION {'OK' if ok_all else 'FAILED'} ===")
    if args.out:
        json.dump({"pass": ok_all, "results": results}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
