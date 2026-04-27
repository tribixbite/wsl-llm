#!/usr/bin/env python3
"""Run LiveCodeBench v6 against our local vLLM+Genesis Qwen3.6-27B server.
Drop-in for SI's lcb_pass_at_1() — uses HTTP instead of in-process vLLM."""
import json, sys, os, time
import urllib.request
import concurrent.futures as cf

# Make SI imports work
sys.path.insert(0, os.path.expanduser("~/git/SI/src"))
sys.path.insert(0, os.path.expanduser("~/git/SI/.venv/lib/python3.10/site-packages"))

from si.livecodebench import (
    load_lcb, lcb_pass_at_1, _check_problem, _extract_code, _build_user_prompt, _SYSTEM_LCB,
    LCBResult,
)
from si.llm import GenParams
from sandbox_fusion import set_endpoint

# Use existing sandbox-fusion container (started by SI)
SANDBOX_ENDPOINT = os.environ.get("SANDBOX_FUSION_ENDPOINT", "http://localhost:46387")
set_endpoint(SANDBOX_ENDPOINT)
print(f"Sandbox endpoint: {SANDBOX_ENDPOINT}", flush=True)

URL = "http://192.168.1.32:8081"
MODEL = "qwen3.6-27b"

def http_chat_one(messages, params):
    """Call our local vLLM server. Returns text completion."""
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "n": params.n,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    # Returns list of n completions
    return [c["message"]["content"] for c in d.get("choices", [])]


class HTTPLLM:
    """Drop-in replacement for SI's GemmaLLM that hits our HTTP server.
    Concurrent batching via ThreadPoolExecutor for throughput."""
    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def chat_batch(self, user_prompts, params, system=None):
        def make_msgs(p):
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": p})
            return msgs
        msgs_list = [make_msgs(p) for p in user_prompts]
        results = [None] * len(msgs_list)
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(http_chat_one, m, params): i for i, m in enumerate(msgs_list)}
            done = 0
            for fut in cf.as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    print(f"  [warn] prompt {i} error: {e}", flush=True)
                    results[i] = [""]
                done += 1
                if done % 25 == 0:
                    print(f"  ... {done}/{len(msgs_list)} done", flush=True)
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--bon", type=int, default=1)
    parser.add_argument("--out", type=str, default="/tmp/lcb_qwen36_27b.json")
    parser.add_argument("--workers", type=int, default=1)  # serial by default — vLLM has max-num-seqs=1
    args = parser.parse_args()

    print(f"Loading LCB v6 problems...", flush=True)
    problems = load_lcb()
    if args.max_problems:
        problems = problems[:args.max_problems]
    print(f"  -> {len(problems)} problems", flush=True)

    llm = HTTPLLM(max_workers=args.workers)
    t0 = time.time()
    print(f"Generating {len(problems)} × {args.bon} completions @ temp={args.temperature}...", flush=True)
    user_prompts = [_build_user_prompt(p) for p in problems]
    sampling_temp = args.temperature if args.bon == 1 else max(args.temperature, 0.8)
    params = GenParams(temperature=sampling_temp, top_p=0.95, max_tokens=args.max_tokens, n=args.bon)
    nested = llm.chat_batch(user_prompts, params, system=_SYSTEM_LCB)
    gen_time = time.time() - t0
    print(f"Generation done in {gen_time:.1f}s. Now verifying via sandbox...", flush=True)

    # Verify (sandbox calls — sequential to avoid sandbox-fusion overload)
    per_problem = {}
    per_diff = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0], "unknown": [0, 0]}
    passed = 0
    for i, (prob, comp_list) in enumerate(zip(problems, nested)):
        ok = False
        for cand in comp_list:
            code = _extract_code(cand)
            if _check_problem(prob, code, timeout_s=10.0):
                ok = True
                break
        per_problem[prob.problem_id] = ok
        d = prob.difficulty if prob.difficulty in per_diff else "unknown"
        per_diff[d][1] += 1
        if ok:
            passed += 1
            per_diff[d][0] += 1
        if (i + 1) % 100 == 0:
            print(f"  verified {i+1}/{len(problems)}: {passed} passed so far", flush=True)

    wall_s = time.time() - t0
    pass_at_1 = passed / max(1, len(problems))

    out = {
        "model": MODEL,
        "endpoint": URL,
        "benchmark": "lcb-v6",
        "passed": passed,
        "total": len(problems),
        "pass_at_1": pass_at_1,
        "wall_s": wall_s,
        "gen_s": gen_time,
        "bon": args.bon,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "per_difficulty": {k: {"passed": v[0], "total": v[1], "pass_rate": v[0]/max(1,v[1])} for k, v in per_diff.items()},
        "per_problem": per_problem,
    }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== LCB v6 Result ===", flush=True)
    print(f"Pass@1: {passed}/{len(problems)} = {pass_at_1*100:.2f}%")
    print(f"By difficulty:")
    for d in ["easy", "medium", "hard"]:
        v = per_diff[d]
        if v[1] > 0:
            print(f"  {d:8}: {v[0]:4}/{v[1]:4} = {100*v[0]/v[1]:.2f}%")
    print(f"Wall: {wall_s:.1f}s ({gen_time:.1f}s gen + {wall_s-gen_time:.1f}s verify)")
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
