#!/usr/bin/env python3
"""Multi-language aider-style polyglot bench (python + javascript + java).

Extends bench/aider_lite.py beyond Python. Same 2-attempt protocol (pass@1 / pass@2,
the retry sees the failing test output), same whole-file edit format.

  python: pytest                 (fast, ~1s/test run)
  javascript: npx jest           (needs node; shared node_modules cache)
  java: ./gradlew test           (needs JDK; first run downloads the gradle dist)

Usage:
  python aider_multi.py --url http://127.0.0.1:18020 --model qwen3.8-27b \
      --key "$(cat /tmp/vllm_key.txt)" --langs python,javascript --n 40 \
      --effort medium --out results.json
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request, glob

ROOT = os.path.expanduser("~/polyglot-benchmark")

LANGS = {
    "python":     {"ext": "py", "sep": "_", "fence": "python"},
    "javascript": {"ext": "js", "sep": "-", "fence": "javascript"},
    "java":       {"ext": "java", "sep": "", "fence": "java"},
}

def read(p):
    try:
        with open(p) as f: return f.read()
    except Exception: return ""

def practice(lang):
    return os.path.join(ROOT, lang, "exercises", "practice")

def solution_file(lang, ex_dir, name):
    """The file the model must write, and its repo-relative path."""
    if lang == "python":
        return os.path.join(ex_dir, name.replace("-", "_") + ".py")
    if lang == "javascript":
        return os.path.join(ex_dir, name + ".js")
    if lang == "java":
        cand = glob.glob(os.path.join(ex_dir, "src", "main", "java", "*.java"))
        return cand[0] if cand else ""
    return ""

def build_prompt(lang, ex_dir, name):
    sol = solution_file(lang, ex_dir, name)
    base = os.path.basename(sol)
    instr = read(os.path.join(ex_dir, ".docs", "instructions.md"))
    instr += "\n" + read(os.path.join(ex_dir, ".docs", "instructions.append.md"))
    stub = read(sol)
    fence = LANGS[lang]["fence"]
    return sol, (
        f"{instr}\n\n"
        f"Below is the stub file `{base}`. Implement it so all unit tests pass. "
        f"Keep the public function/class names and signatures. Return the COMPLETE "
        f"contents of `{base}` in a SINGLE ```{fence} code block, nothing else.\n\n"
        f"```{fence}\n{stub}\n```"
    )

def extract_code(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    blocks = re.findall(r"```[a-zA-Z]*\s*\n(.*?)```", text, re.DOTALL)
    return max(blocks, key=len).strip() if blocks else text.strip()

def call(url, model, key, messages, max_tokens, timeout=900, effort=None, think=False):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    elif think:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d["choices"][0]["message"]["content"]

JS_CACHE = os.path.expanduser("~/.cache/aider_multi_node_modules")

def run_tests(lang, ex_dir, name, code, py, timeout=300):
    with tempfile.TemporaryDirectory() as td:
        shutil.copytree(ex_dir, td, dirs_exist_ok=True)
        sol = solution_file(lang, td, name)
        if not sol:
            return False, "no solution file"
        with open(sol, "w") as f:
            f.write(code)
        try:
            if lang == "python":
                tf = glob.glob(os.path.join(td, "*_test.py"))
                cmd = [py, "-m", "pytest", "-q", "-x", tf[0]] if tf else None
            elif lang == "javascript":
                # reuse one node_modules across exercises
                if os.path.isdir(JS_CACHE):
                    dst = os.path.join(td, "node_modules")
                    if not os.path.exists(dst):
                        os.symlink(JS_CACHE, dst)
                # babel cannot resolve bare preset names through the symlinked
                # node_modules; pin them with require.resolve instead.
                with open(os.path.join(td, "babel.config.js"), "w") as bf:
                    bf.write("module.exports = { presets: [[require.resolve("
                             "'@babel/preset-env'), {targets: {node: 'current'}}]] };\n")
                cmd = ["npx", "--no-install", "jest", "--silent"]
            elif lang == "java":
                os.chmod(os.path.join(td, "gradlew"), 0o755)
                cmd = ["./gradlew", "test", "--offline", "-q", "--console=plain"]
            else:
                cmd = None
            if not cmd:
                return False, "no runner"
            env = dict(os.environ, NODE_PATH=JS_CACHE)
            r = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=timeout, env=env)
            return r.returncode == 0, (r.stdout + r.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            return False, f"TIMEOUT (> {timeout}s)"
        except Exception as e:
            return False, repr(e)

RETRY = ("The tests failed. Below is the test output.\n\n```\n{output}\n```\n\n"
         "The tests themselves are correct and must not be changed. Fix `{base}` so every "
         "test passes. Return the COMPLETE corrected contents of `{base}` in a SINGLE "
         "```{fence} code block, nothing else.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--key", default="x")
    ap.add_argument("--langs", default="python,javascript")
    ap.add_argument("--n", type=int, default=30, help="exercises PER language")
    ap.add_argument("--effort", default=None); ap.add_argument("--think", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--tries", type=int, default=2)
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    t0 = time.time()
    for lang in args.langs.split(","):
        lang = lang.strip()
        if lang not in LANGS: continue
        pdir = practice(lang)
        if not os.path.isdir(pdir):
            print(f"[{lang}] no exercises at {pdir}", flush=True); continue
        names = sorted(d for d in os.listdir(pdir) if os.path.isdir(os.path.join(pdir, d)))[:args.n]
        for i, name in enumerate(names, 1):
            ex_dir = os.path.join(pdir, name)
            sol, prompt = build_prompt(lang, ex_dir, name)
            if not sol or not os.path.exists(sol):
                continue
            base = os.path.basename(sol)
            msgs = [{"role": "user", "content": prompt}]
            p1 = p2 = False; secs = 0.0; tail = ""
            try:
                ts = time.time()
                reply = call(args.url, args.model, args.key, msgs, args.max_tokens,
                             effort=args.effort, think=args.think)
                secs += time.time() - ts
                ok, tail = run_tests(lang, ex_dir, name, extract_code(reply), args.py)
                p1 = p2 = ok
                if not ok and args.tries > 1:
                    msgs += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": RETRY.format(
                                 output=tail, base=base, fence=LANGS[lang]["fence"])}]
                    ts = time.time()
                    reply2 = call(args.url, args.model, args.key, msgs, args.max_tokens,
                                  effort=args.effort, think=args.think)
                    secs += time.time() - ts
                    p2, tail = run_tests(lang, ex_dir, name, extract_code(reply2), args.py)
            except Exception as e:
                tail = f"ERR {e!r}"
            rows.append({"lang": lang, "name": name, "pass1": p1, "pass2": p2,
                         "secs": round(secs, 1)})
            mark = "PASS" if p1 else ("PASS@2" if p2 else "FAIL")
            print(f"[{lang} {i}/{len(names)}] {name:26} {mark:7} ({secs:.0f}s)", flush=True)
            if args.out:
                with open(args.out + ".jsonl", "a") as f:
                    f.write(json.dumps(rows[-1]) + "\n")

    print("\n=== RESULTS ===")
    for lang in sorted(set(r["lang"] for r in rows)):
        sub = [r for r in rows if r["lang"] == lang]
        n = len(sub)
        print(f"  {lang:11} n={n:3}  pass@1={sum(r['pass1'] for r in sub)/n:6.1%}  "
              f"pass@2={sum(r['pass2'] for r in sub)/n:6.1%}")
    n = len(rows)
    if n:
        print(f"  {'OVERALL':11} n={n:3}  pass@1={sum(r['pass1'] for r in rows)/n:6.1%}  "
              f"pass@2={sum(r['pass2'] for r in rows)/n:6.1%}")
    print(f"  wall={time.time()-t0:.0f}s")
    if args.out:
        json.dump({"model": args.model, "rows": rows}, open(args.out, "w"), indent=2)

if __name__ == "__main__":
    main()
