#!/usr/bin/env python3
"""Lightweight Aider-style polyglot bench (Python subset) against a local OpenAI endpoint.

NOT the official Aider harness (which uses diff format + Docker across 6 languages). This is
a faithful *whole-file* approximation on the Python exercises only — pure inference + local
pytest, low system load.

Like the official benchmark it runs **two attempts**: if the tests fail, the pytest output is
fed back into the same conversation and the model gets one chance to fix it. Both rates are
reported, because they mean different things:
  pass@1 — solved cold, no feedback  (comparable to this repo's older single-attempt numbers)
  pass@2 — solved within two tries   (**the number the aider leaderboard headlines**)

Usage:
  python aider_lite.py --url http://localhost:8080 --model qwen3.6-35b-a3b \
      --key "$(cat /tmp/llama_key.txt)" --n 12 --out /tmp/aider_35b.json
  python aider_lite.py ... --tries 1        # old single-attempt behaviour
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

def call(url, model, key, messages, max_tokens=3000, timeout=600, think=False):
    if isinstance(messages, str):                      # back-compat: bare prompt
        messages = [{"role": "user", "content": messages}]
    body = {"model": model, "messages": messages,
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
            # Keep enough output to be actionable on the retry attempt; the
            # console line truncates this further to a single line.
            return r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT: the test run exceeded 120s (likely an infinite loop)."
        except Exception as e:
            return False, repr(e)


RETRY_TEMPLATE = (
    "The tests failed. Below is the pytest output.\n\n"
    "```\n{output}\n```\n\n"
    "The tests themselves are correct and must not be changed. Fix `{slug}.py` so every test "
    "passes. Return the COMPLETE corrected contents of `{slug}.py` in a SINGLE ```python code "
    "block, nothing else."
)


def attempt(url, model, key, messages, ex_dir, name, py, max_tokens, think):
    """One model call + test run. Returns (ok, test_output, reply_text, seconds)."""
    started = time.time()
    reply = call(url, model, key, messages, max_tokens=max_tokens, think=think)
    elapsed = time.time() - started
    ok, output = run_tests(ex_dir, name, extract_code(reply), py)
    return ok, output, reply, elapsed

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
    ap.add_argument("--tries", type=int, default=2,
                    help="attempts per exercise; 2 matches the official aider benchmark "
                         "(test output fed back on failure). 1 = legacy single-attempt.")
    args = ap.parse_args()
    max_tokens = args.max_tokens or (20000 if args.think else 3000)

    all_ex = sorted(d for d in os.listdir(PRACTICE) if os.path.isdir(os.path.join(PRACTICE, d)))
    names = args.exercises.split(",") if args.exercises else all_ex[:args.n]

    results = []
    t0 = time.time()
    for i, name in enumerate(names, 1):
        ex_dir = os.path.join(PRACTICE, name)
        slug, _, prompt = build_prompt(ex_dir, name)
        messages = [{"role": "user", "content": prompt}]
        pass1 = pass2 = False
        gen_s = 0.0
        tail = ""
        try:
            ok, tail, reply, dt = attempt(args.url, args.model, args.key, messages,
                                          ex_dir, name, args.py, max_tokens, args.think)
            gen_s += dt
            pass1 = pass2 = ok
            # Second attempt: feed the pytest output back into the same
            # conversation, exactly as the official aider benchmark does.
            if not ok and args.tries > 1:
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user",
                                 "content": RETRY_TEMPLATE.format(output=tail, slug=slug)})
                ok, tail, _, dt = attempt(args.url, args.model, args.key, messages,
                                          ex_dir, name, args.py, max_tokens, args.think)
                gen_s += dt
                pass2 = ok
        except Exception as e:
            tail = f"CALL_ERR {e!r}"

        rec = {"name": name, "pass": pass2, "pass1": pass1, "pass2": pass2,
               "gen_s": round(gen_s, 1)}
        results.append(rec)
        mark = "PASS" if pass1 else ("PASS@2" if pass2 else "FAIL")
        print(f"[{i}/{len(names)}] {name:24} {mark:7} ({gen_s:.0f}s)"
              + ("" if pass2 else f"  …{tail.splitlines()[-1][:70] if tail.strip() else ''}"), flush=True)
        # crash-resilient incremental write (this box dies often)
        if args.out:
            with open(args.out + ".jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")

    n = len(results)
    p1 = sum(r["pass1"] for r in results)
    p2 = sum(r["pass2"] for r in results)
    print(f"\n=== {args.model} (whole-file, python subset, tries={args.tries})")
    print(f"    pass@1 = {p1}/{n} = {100*p1/n:.1f}%")
    print(f"    pass@2 = {p2}/{n} = {100*p2/n:.1f}%   <- aider leaderboard headline metric")
    print(f"    wall={time.time()-t0:.0f}s")
    if args.out:
        json.dump({"model": args.model, "tries": args.tries, "n": n,
                   "pass": p2, "pass1": p1, "pass2": p2, "results": results},
                  open(args.out, "w"), indent=2)
        print(f"    wrote {args.out}")

if __name__ == "__main__":
    main()
