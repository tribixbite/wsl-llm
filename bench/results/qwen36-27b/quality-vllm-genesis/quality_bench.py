#!/usr/bin/env python3
"""Run quality bench prompts (Conway GoL HTML, regex, sudoku, etc.)
on the new vLLM+Genesis stack. Save outputs."""
import json, sys, os, time, urllib.request

URL = "http://192.168.1.32:8081"
MODEL = "qwen3.6-27b"
OUT_DIR = "/tmp/quality_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Hard prompts from prior bench
PROMPTS = {
    "conway_gol_rle": (
        "Implement Conway's Game of Life in a single HTML file, but the initial state is parsed from the URL hash as a Run-Length Encoded string in standard `xRLE`-style: `b` = dead cell, `o` = live cell, `$` = end of row, `!` = end of pattern, with optional run-counts before each token (e.g. `3o2b1o!2b3o!` is a 6-cell-wide pattern with 2 rows). The hash also drives the initial position; if no hash is provided, use a glider in the top-left. The page should: tick at 10 fps, allow click-toggling cells when paused, have play/pause/step/reset buttons, encode the current state back into the URL hash on change so it can be shared. Keep it under ~12 KB. No external dependencies. Single self-contained HTML."
    ),
    "regex_thompson_nfa": (
        "Implement a complete regular-expression engine in pure Python using the Thompson NFA construction. Support: literal chars, `.` (any), `*`, `+`, `?`, `|`, parentheses, `[...]` character classes, `\\d`, `\\w`, `\\s`, anchors `^` and `$`, escaped metas. Build the NFA, simulate it (epsilon-closure + Thompson set-of-states matching). No backtracking. Provide `compile(pattern)` and `match(re_obj, text)`. Include 5+ test cases at the bottom that pass."
    ),
    "sudoku_csp_ac3": (
        "Implement a Sudoku solver in TypeScript using CSP + AC-3 + backtracking. The board is a 9×9 grid of integers 1-9, with 0 = empty. Apply AC-3 to prune the domains of all empty cells, then backtrack with MRV (minimum remaining values) heuristic. Show the algorithm step count. Solve this puzzle: [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]. Output the solved board and step count."
    ),
    "svelte_todo": (
        "Write a complete Svelte 5 + TypeScript todo app component. Use `$state` runes for state. Features: add task (Enter key), toggle complete (checkbox), delete (button), filter by all/active/completed, count remaining, persist to localStorage. Include Tailwind classes for styling. Keep it in one .svelte file."
    ),
    "kotlin_lru": (
        "Implement an LRU cache in Kotlin using only stdlib. Generic `LRUCache<K, V>` with `get(k)`, `put(k, v)`, and `size()`. Thread-safe. Use a doubly-linked list + HashMap for O(1) operations. Include a runnable main() with 6+ test cases proving LRU eviction order is correct."
    ),
}

def run(name, prompt, max_tokens=8000):
    print(f"[{name}] launching...")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t_start = time.time()
    ttft = None
    completion = []
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=900) as r:
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
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.time() - t_start
                    completion.append(content)
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", 0)
    wall = time.time() - t_start
    text = "".join(completion)
    decode_tps = completion_tokens / (wall - ttft) if (wall - ttft) > 0 else 0

    out_path = f"{OUT_DIR}/{name}.txt"
    with open(out_path, "w") as f:
        f.write(text)
    print(f"[{name}] tokens={completion_tokens} wall={wall:.1f}s decode_tps={decode_tps:.2f}")
    return {"name": name, "tokens": completion_tokens, "wall": wall, "ttft": ttft, "decode_tps": decode_tps}

results = []
for name, prompt in PROMPTS.items():
    try:
        r = run(name, prompt)
        results.append(r)
    except Exception as e:
        print(f"[{name}] ERROR: {e}")
        results.append({"name": name, "error": str(e)})

# Summary
print("\n=== Quality Bench Summary ===")
print(f"{'Prompt':<30} {'Tokens':>8} {'Wall(s)':>10} {'Decode TPS':>12}")
for r in results:
    if "error" in r:
        print(f"{r['name']:<30} ERROR: {r['error']}")
    else:
        print(f"{r['name']:<30} {r['tokens']:>8} {r['wall']:>10.2f} {r['decode_tps']:>12.2f}")

with open(f"{OUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nOutputs saved to {OUT_DIR}/")
