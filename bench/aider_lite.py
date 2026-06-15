#!/usr/bin/env python3
"""Lightweight Aider-style polyglot bench (Python subset) against a local OpenAI endpoint.

NOT the official Aider harness (which uses diff format + 2 attempts + Docker across 6
languages). This is a faithful *whole-file, single-attempt* approximation on the Python
exercises only — pure inference + local pytest, low system load. Use for a representative
pass@1 datapoint and to compare models on the same fixed exercise set.

Usage:
  python aider_lite.py --url http://localhost:8080 --model qwen3.6-35b-a3b \
      --key "$(cat /tmp/llama_key.txt)" --n 12 --out /tmp/aider_35b.json
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request, glob

PRACTICE = os.path.expanduser("~/polyglot-benchmark/python/exercises/practice")

def read(p):
    try:
        with open(p) as f: return f.read()
    except Exception: return ""

def build_prompt(ex_dir, name):
    slug = name.replace("-", "_")
    instr = read(os.path.join(ex_dir, ".docs", "instructions.md"))
    instr += "\n" + read(os.path.join(ex_dir, ".docs", "instructions.append.md"))
    stub_path = os.path.join(ex_dir, f"{slug}.py")
    stub = read(stub_path)
    return slug, stub_path, (
        f"{instr}\n\n"
        f"Below is the stub file `{slug}.py`. Implement it so all unit tests pass. "
        f"Keep the public function/class names and signatures. Return the COMPLETE "
        f"contents of `{slug}.py` in a SINGLE ```python code block, nothing else.\n\n"
        f"```python\n{stub}\n```"
    )

def extract_code(text):
    # strip a leading deepseek-style reasoning block if present
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()

def call(url, model, key, prompt, max_tokens=3000, timeout=600, think=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": bool(think)}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d["choices"][0]["message"]["content"]

def run_tests(ex_dir, name, code, py):
    slug = name.replace("-", "_")
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(ex_dir, td, dirs_exist_ok=True)
        with open(os.path.join(td, f"{slug}.py"), "w") as f:
            f.write(code)
        testfile = os.path.join(td, f"{slug}_test.py")
        if not os.path.exists(testfile):
            cand = glob.glob(os.path.join(td, "*_test.py"))
            testfile = cand[0] if cand else testfile
        try:
            r = subprocess.run([py, "-m", "pytest", "-q", "-x", testfile],
                               cwd=td, capture_output=True, text=True, timeout=120)
            return r.returncode == 0, (r.stdout + r.stderr)[-400:]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, repr(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default="x")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--out", default=None)
    ap.add_argument("--exercises", default=None, help="comma-separated names; else first N alphabetically")
    ap.add_argument("--think", action="store_true", help="enable_thinking:true (needs server reasoning-budget>0)")
    ap.add_argument("--max-tokens", type=int, default=0, help="0=auto (3000, or 20000 with --think)")
    args = ap.parse_args()
    max_tokens = args.max_tokens or (20000 if args.think else 3000)

    all_ex = sorted(d for d in os.listdir(PRACTICE) if os.path.isdir(os.path.join(PRACTICE, d)))
    names = args.exercises.split(",") if args.exercises else all_ex[:args.n]

    results = []
    t0 = time.time()
    for i, name in enumerate(names, 1):
        ex_dir = os.path.join(PRACTICE, name)
        slug, _, prompt = build_prompt(ex_dir, name)
        try:
            ts = time.time()
            out = call(args.url, args.model, args.key, prompt, max_tokens=max_tokens, think=args.think)
            gen_s = time.time() - ts
            code = extract_code(out)
            ok, tail = run_tests(ex_dir, name, code, args.py)
        except Exception as e:
            ok, tail, gen_s = False, f"CALL_ERR {e!r}", 0
        results.append({"name": name, "pass": ok, "gen_s": round(gen_s, 1)})
        print(f"[{i}/{len(names)}] {name:24} {'PASS' if ok else 'FAIL'}  ({gen_s:.0f}s)"
              + ("" if ok else f"  …{tail.splitlines()[-1][:70] if tail.strip() else ''}"), flush=True)
        # crash-resilient incremental write (this box dies often)
        if args.out:
            with open(args.out + ".jsonl", "a") as f:
                f.write(json.dumps({"name": name, "pass": ok, "gen_s": round(gen_s, 1)}) + "\n")

    n = len(results); p = sum(r["pass"] for r in results)
    print(f"\n=== {args.model}: pass@1 = {p}/{n} = {100*p/n:.1f}%  (whole-file, single-attempt, python subset)")
    print(f"    wall={time.time()-t0:.0f}s")
    if args.out:
        json.dump({"model": args.model, "pass": p, "n": n, "results": results}, open(args.out, "w"), indent=2)
        print(f"    wrote {args.out}")

if __name__ == "__main__":
    main()
