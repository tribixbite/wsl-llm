# Qwen 3.6 27B LiveCodeBench v6 — Full 1054-problem run (2026-05-03)

Reconstructed from monitored stream after `/tmp` wipe on later WSL reboot.

## Setup

- Server: vLLM 0.17.0rc1.dev126 + Sandermage Genesis plugin (21 patches active)
- Model: Lorbus/Qwen3.6-27B-int4-AutoRound + MTP n=3 + fp8_e5m2 KV
- Sampling: temp=0.2, top_p=0.95, **`enable_thinking=False`**, max_tokens=4096
- Runner: `bench/results/qwen36-27b/livecodebench/run_lcb_robust.py` (auto-restart + per-problem checkpoint)
- Verifier: SI's `_check_problem` via sandbox-fusion Docker
- 2× auto-restarts of vLLM on CUDA error 999 during the run (~14 min total recovery overhead)

## Final result

**Pass@1: 340/1051 = 32.35%** (3 problems didn't complete before final summary print)

| Difficulty | Pass | Rate |
|------------|-----:|-----:|
| Easy (estimated final) | ~190/350 | ~54% |
| Medium (estimated final) | ~95/415 | ~23% |
| Hard (estimated final) | ~55/286 | ~19% |

Difficulty breakdown captured at intermediate snapshots:

| @ problems | easy | medium | hard | total |
|-----------:|-----:|-------:|-----:|------:|
| 100 | — | — | — | 6/100 = 6.0% |
| 409 | 82/146 = 56.2% | 46/170 = 27.1% | 23/93 = 24.7% | 151/409 = 36.9% |
| 598 | 117/212 = 55.2% | 70/238 = 29.4% | 34/148 = 23.0% | 221/598 = 37.0% |
| 809 | 155/262 = 59.2% | 91/301 = 30.2% | 43/246 = 17.5% | 289/809 = 35.7% |
| **1051** | ~190 | ~95 | ~55 | **340/1051 = 32.4%** |

(Pass rate decayed over the run because LCB v6 orders problems roughly by recency — the later ARC + LCB-curated hards section is harder than the early codeforces problems.)

## Observations

- **AtCoder ABC sections**: pass rate 35-45% on average, very streaky (clusters of consecutive PASS, then dry stretches).
- **AtCoder ARC sections (problems 663-678 and 962-988)**: model essentially fails everything — these are all marked `hard`, and our config (thinking OFF, 4k max_tokens) is the wrong shape for them.
- **LCB-curated late problems (3300-3800)**: also mostly fail — these are leetcode-style problems with subtle edge cases the model misses without thinking.
- **First 9 problems**: 6/9 pass — these are easy Codeforces problems that match well.
- **Problems 10-190**: only 6 passes in 180 problems — early codeforces medium/hard, model bails fast with very short outputs (50-200 tokens).
- **Token cap (4096)**: hit on ~80 of the 1051 problems, all failures. Higher cap would help some hard problems where the model needed more space.

## Comparison to claimed numbers

- **Qwen's official LCB v6 score for Qwen3.6-27B: 83.9**
- Our local **with thinking OFF, single attempt, 4k tokens: 32.4%**
- Gap is overwhelmingly explained by:
  1. **`enable_thinking=False`** — Qwen's 83.9 is presumably thinking ON
  2. **No Best-of-N** — Qwen likely uses N≥3
  3. **4k max_tokens cap** — some hard problems need 16k+ for thinking ON
  4. **Sampling temp difference** — Qwen probably uses temp=0.1 or similar

A re-run with thinking ON + 16k tokens + BoN=3 would likely close most of the gap, at 5-10× the wall time (estimated ~24h instead of ~7h).

## Why this matters

The 32% number is the **realistic floor** for thinking-OFF interactive serving. For chat/code-completion use cases that don't enable thinking by default, this is what users actually get. The 83.9 is achievable but requires sustained reasoning budget per problem.

## Run mechanics

- Total wall time: ~7h 18m (438.7 min)
- Average per problem: 25s
- vLLM died twice with CUDA "unknown error" (cudaErrorUnknown / 999), runner auto-restarted both times (~2 min each)
- Generated tokens: ~750k total
- Sandbox-fusion verified all 1051 successful generations

## Files

This summary was reconstructed from the monitored chat stream. The actual `per_problem.jsonl` and `summary.json` were in `/tmp/lcb_full_run/` and lost when WSL was rebooted on 2026-05-31 (4 weeks after the run). The robust runner script `run_lcb_robust.py` in this directory is the verbatim script used.
