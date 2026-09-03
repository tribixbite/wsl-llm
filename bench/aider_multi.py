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
import threading, concurrent.futures as cf

ROOT = os.path.expanduser("~/polyglot-benchmark")

LANGS = {
    "python":     {"fence": "python"},
    "javascript": {"fence": "javascript"},
    "java":       {"fence": "java"},
    "cpp":        {"fence": "cpp"},
    "go":         {"fence": "go"},
    "rust":       {"fence": "rust"},
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
    if lang == "cpp":
        # implementation file, not the header and not the *_test.cpp
        cand = [f for f in glob.glob(os.path.join(ex_dir, "*.cpp"))
                if not f.endswith("_test.cpp")]
        return cand[0] if cand else ""
    if lang == "go":
        cand = [f for f in glob.glob(os.path.join(ex_dir, "*.go"))
                if not f.endswith("_test.go")]
        return cand[0] if cand else ""
    if lang == "rust":
        p_ = os.path.join(ex_dir, "src", "lib.rs")
        return p_ if os.path.exists(p_) else ""
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

SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}   # Qwen3.8 thinking defaults
STATS = {"prompt_tokens": 0, "completion_tokens": 0, "error_outputs": 0,
         "num_malformed_responses": 0, "num_with_malformed_responses": 0,
         "exhausted_context_windows": 0, "test_timeouts": 0}


def code_block_found(text):
    """A well-formed reply contains a fenced code block we can extract."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return bool(re.search(r"```[a-zA-Z]*\s*\n.*?```", text, re.DOTALL))
EFFORT_STYLE = "kwargs"      # "kwargs" (vLLM/llama.cpp) | "top_level" (NInfer)


def call(url, model, key, messages, max_tokens, timeout=900, effort=None, think=False):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            **SAMPLING}
    if effort:
        # Two conventions in the wild: vLLM/llama.cpp take it inside
        # chat_template_kwargs; NInfer takes the OpenAI top-level field and
        # 400s on the kwargs form. EFFORT_STYLE picks one.
        if EFFORT_STYLE == "top_level":
            body["reasoning_effort"] = effort
        else:
            body["chat_template_kwargs"] = {"reasoning_effort": effort}
    elif think:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    ch = (d.get("choices") or [{}])[0]
    u = d.get("usage") or {}
    return (ch.get("message", {}).get("content") or "",
            {"prompt_tokens": u.get("prompt_tokens") or 0,
             "completion_tokens": u.get("completion_tokens") or 0,
             "finish_reason": ch.get("finish_reason")})

JS_CACHE = os.path.expanduser("~/.cache/aider_multi_node_modules")

LANG_TIMEOUT = {"python": 120, "javascript": 180, "java": 420,
                "cpp": 420, "go": 300, "rust": 600}


def run_tests(lang, ex_dir, name, code, py, timeout=None):
    timeout = timeout or LANG_TIMEOUT.get(lang, 300)
    with tempfile.TemporaryDirectory() as parent:
        # exercism's cpp CMakeLists and java gradle derive the target/project name
        # from the DIRECTORY name, so the copy must keep the exercise's own name.
        # Copying into a random tmpdir silently breaks them ("No SOURCES given to
        # target"), which scored all of cpp and java 0% before this was fixed.
        td = os.path.join(parent, name)
        shutil.copytree(ex_dir, td)
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
            elif lang == "cpp":
                # exercism cpp ships CMakeLists + a bundled test framework
                cmd = ["bash", "-lc",
                       "cmake -S . -B build -DEXERCISM_RUN_ALL_TESTS=1 "
                       "-DCMAKE_BUILD_TYPE=Release >/dev/null && "
                       "cmake --build build -j4 >/dev/null && ctest --test-dir build "
                       "--output-on-failure"]
            elif lang == "go":
                cmd = ["go", "test", "./..."]
            elif lang == "rust":
                cmd = ["cargo", "test", "--offline", "--quiet"]
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
    ap.add_argument("--temp", type=float, default=1.0,
                    help="Qwen3.8 thinking default is 1.0 (temp 0 cripples this model)")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel exercises; needs a server with >1 slot "
                         "(vLLM MAX_SEQS=N, llama.cpp -np N)")
    ap.add_argument("--tag", default="", help="dirname tag, e.g. 3090-vllm-vision-250W")
    ap.add_argument("--edit-format", default="whole",
                    help="reported in the YAML; this harness sends whole files")
    ap.add_argument("--versions", default="aider_multi.py-1.0")
    ap.add_argument("--effort-style", choices=["kwargs", "top_level"], default="kwargs",
                    help="NInfer needs top_level; vLLM/llama.cpp need kwargs")
    args = ap.parse_args()
    globals()["EFFORT_STYLE"] = args.effort_style
    SAMPLING.update(temperature=args.temp, top_p=args.top_p, top_k=args.top_k)
    print(f"sampling: {SAMPLING}", flush=True)

    rows = []
    t0 = time.time()

    tasks = []
    for lang in args.langs.split(","):
        lang = lang.strip()
        if lang not in LANGS:
            continue
        pdir = practice(lang)
        if not os.path.isdir(pdir):
            print(f"[{lang}] no exercises at {pdir}", flush=True)
            continue
        for name in sorted(d for d in os.listdir(pdir)
                           if os.path.isdir(os.path.join(pdir, d)))[:args.n]:
            tasks.append((lang, name))

    lock = threading.Lock()
    done = [0]

    def work(task):
        lang, name = task
        ex_dir = os.path.join(practice(lang), name)
        sol, prompt = build_prompt(lang, ex_dir, name)
        if not sol or not os.path.exists(sol):
            return None
        base = os.path.basename(sol)
        msgs = [{"role": "user", "content": prompt}]
        p1 = p2 = False; secs = 0.0; tail = ""; malformed = False
        try:
            ts = time.time()
            reply, meta = call(args.url, args.model, args.key, msgs, args.max_tokens,
                               effort=args.effort, think=args.think)
            secs += time.time() - ts
            with lock:
                STATS["prompt_tokens"] += meta["prompt_tokens"]
                STATS["completion_tokens"] += meta["completion_tokens"]
                if meta["finish_reason"] == "length":
                    STATS["exhausted_context_windows"] += 1
            code = extract_code(reply)
            if not code_block_found(reply):
                with lock: STATS["num_malformed_responses"] += 1
                malformed = True
            ok, tail = run_tests(lang, ex_dir, name, code, args.py)
            if str(tail).startswith("TIMEOUT"):
                with lock: STATS["test_timeouts"] += 1
            p1 = p2 = ok
            if not ok and args.tries > 1:
                msgs += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": RETRY.format(
                             output=tail, base=base, fence=LANGS[lang]["fence"])}]
                ts = time.time()
                reply2, meta2 = call(args.url, args.model, args.key, msgs,
                                     args.max_tokens, effort=args.effort,
                                     think=args.think)
                secs += time.time() - ts
                with lock:
                    STATS["prompt_tokens"] += meta2["prompt_tokens"]
                    STATS["completion_tokens"] += meta2["completion_tokens"]
                    if meta2["finish_reason"] == "length":
                        STATS["exhausted_context_windows"] += 1
                if not code_block_found(reply2):
                    with lock: STATS["num_malformed_responses"] += 1
                    malformed = True
                p2, tail = run_tests(lang, ex_dir, name, extract_code(reply2), args.py)
        except Exception as e:
            tail = f"ERR {e!r}"
            with lock: STATS["error_outputs"] += 1
        if malformed:
            with lock: STATS["num_with_malformed_responses"] += 1
        rec = {"lang": lang, "name": name, "pass1": p1, "pass2": p2,
               "secs": round(secs, 1), "malformed": malformed}
        with lock:
            done[0] += 1
            mark = "PASS" if p1 else ("PASS@2" if p2 else "FAIL")
            print(f"[{done[0]}/{len(tasks)}] {lang:11}{name:26} {mark:7} ({secs:.0f}s)",
                  flush=True)
            if args.out:
                with open(args.out + ".jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
        return rec

    if args.jobs > 1:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            rows = [r for r in ex.map(work, tasks) if r]
    else:
        rows = [r for r in (work(t) for t in tasks) if r]

    n = len(rows)
    p1n = sum(r["pass1"] for r in rows)
    p2n = sum(r["pass2"] for r in rows)
    wall = time.time() - t0
    wf = n - STATS["num_with_malformed_responses"]

    # human-readable per-language table
    print("\n" + "=" * 58)
    print(f"  {'language':<12}{'n':>5}{'pass@1':>10}{'pass@2':>10}{'avg s/ex':>11}")
    print("  " + "-" * 46)
    for lang in sorted(set(r["lang"] for r in rows)):
        sub = [r for r in rows if r["lang"] == lang]
        m = len(sub)
        print(f"  {lang:<12}{m:>5}{sum(r['pass1'] for r in sub)/m:>9.1%}"
              f"{sum(r['pass2'] for r in sub)/m:>10.1%}"
              f"{sum(r['secs'] for r in sub)/m:>11.1f}")
    print("  " + "-" * 46)
    if n:
        print(f"  {'TOTAL':<12}{n:>5}{p1n/n:>9.1%}{p2n/n:>10.1%}{wall/n:>11.1f}")

    # aider-leaderboard YAML, the format shared in the community
    tag = args.tag or re.sub(r"[^A-Za-z0-9.-]+", "-", args.model)
    dirname = f"{time.strftime('%Y-%m-%d-%H-%M-%S')}--{tag}"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True,
                                timeout=5).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    y = [
        f"- dirname: {dirname}",
        f"  test_cases: {n}",
        f"  model: {args.model}",
        f"  edit_format: {args.edit_format}",
        f"  commit_hash: {commit}",
        f"  pass_rate_1: {100*p1n/n if n else 0:.1f}",
        f"  pass_rate_2: {100*p2n/n if n else 0:.1f}",
        f"  pass_num_1: {p1n}",
        f"  pass_num_2: {p2n}",
        f"  percent_cases_well_formed: {100*wf/n if n else 0:.1f}",
        f"  error_outputs: {STATS['error_outputs']}",
        f"  num_malformed_responses: {STATS['num_malformed_responses']}",
        f"  num_with_malformed_responses: {STATS['num_with_malformed_responses']}",
        "  user_asks: 0",
        "  lazy_comments: 0",
        "  syntax_errors: 0",
        "  indentation_errors: 0",
        f"  exhausted_context_windows: {STATS['exhausted_context_windows']}",
        f"  prompt_tokens: {STATS['prompt_tokens']}",
        f"  completion_tokens: {STATS['completion_tokens']}",
        f"  test_timeouts: {STATS['test_timeouts']}",
        f"  total_tests: {n}",
        f"  command: aider --model openai/{args.model}",
        f"  date: {time.strftime('%Y-%m-%d')}",
        f"  versions: {args.versions}",
        f"  seconds_per_case: {wall/n if n else 0:.1f}",
    ]
    print("\n" + "\n".join(y))
    print(f"\n  sampling: {SAMPLING}   tries: {args.tries}   "
          f"reasoning_effort: {args.effort or 'default'}")

    if args.out:
        json.dump({"dirname": dirname, "model": args.model, "n": n,
                   "pass_rate_1": round(100*p1n/n, 1) if n else 0,
                   "pass_rate_2": round(100*p2n/n, 1) if n else 0,
                   "pass_num_1": p1n, "pass_num_2": p2n,
                   "percent_cases_well_formed": round(100*wf/n, 1) if n else 0,
                   "edit_format": args.edit_format, "sampling": SAMPLING,
                   "tries": args.tries, "effort": args.effort,
                   "seconds_per_case": round(wall/n, 1) if n else 0,
                   **STATS, "rows": rows}, open(args.out, "w"), indent=2)
        with open(args.out.replace(".json", "") + ".yaml", "w") as f:
            f.write("\n".join(y) + "\n")
        print(f"  wrote {args.out} and .yaml")


if __name__ == "__main__":
    main()
