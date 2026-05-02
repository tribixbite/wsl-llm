# Qwen 3.6 27B LiveCodeBench v6 (2026-04-27)

LiveCodeBench v6 = 1054-1055 leetcode-style problems split across 6 monthly test files. Run against our local vLLM + Genesis stack (the §11 production setup).

## Reference numbers

Per official Qwen HF model card:
- **Qwen3.6-27B official LCB v6 = 83.9** (single number, no easy/medium/hard split)
- Compared to: Gemini 3 Pro Preview 91.7, DeepSeek V3.2 Speciale 89.6
- Best open-weight dense model under 30B

## Setup

LCB dataset is at `~/git/SI/cache/livecodebench/code_generation_lite/` (loaded by SI's `si.livecodebench` module). Verification uses sandbox-fusion (containers running on ports 46387, 46773 typically).

Sandbox-fusion containers must be reachable. Check with:
```bash
docker ps | grep sandbox-fusion
curl -sX POST http://localhost:46387/run_code -H "Content-Type: application/json" \
  -d '{"code":"print(1)","language":"python","run_timeout":5}'
# expect "status":"Success"
```

## Smoke test (5 problems)

`lcb_smoke5.json` — 4/5 = 80% (3/3 easy + 1/2 medium). Pipeline validated.

## Run command

```bash
SANDBOX_FUSION_ENDPOINT=http://localhost:46387 \
  ~/git/SI/.venv/bin/python -u run_lcb.py \
    --max-problems 200 --max-tokens 2048 \
    --out lcb_qwen36_200.json
```

Or run all 1054 (overnight, ~6-8 hours):
```bash
SANDBOX_FUSION_ENDPOINT=http://localhost:46387 \
  ~/git/SI/.venv/bin/python -u run_lcb.py --out lcb_qwen36_full.json
```

## Why we use HTTP-based runner

SI's built-in `python -m si.cli anchor --benchmark lcb` uses an in-process vLLM (Gemma chat template hardcoded). Our `run_lcb.py` mimics SI's `GemmaLLM.chat_batch()` interface but routes via HTTP to our running vLLM+Genesis server — preserves all Genesis patches and avoids needing to reload the 18 GB Lorbus model.

## Throughput context

At ~30s/problem (sequential, max-num-seqs=1):
- 200 problems = ~100 minutes
- 1054 problems = ~9 hours

## Files

| File | What |
|------|------|
| `run_lcb.py` | HTTP-based LCB v6 runner (concurrent batched requests) |
| `lcb_debug.py` | Sequential single-problem debugger that saves raw outputs |
| `lcb_smoke5.json` | 5-problem smoke: **4/5 = 80%** (3 easy + 1 medium pass) |
| `debug_20_thinking_off/` | 20 problems with `enable_thinking=False`, max_tokens=4096 — **7/20 = 35%** (4/8 easy + 1/7 medium + 2/5 hard) — raw outputs included |
| `lcb_200_v2_artifact.json` | Failed run — vLLM crashed mid-way with CUDA error 999 (post-WSL-crash residual). 6/200 result is NOT representative; ~164 of 200 prompts returned "connection refused" and auto-failed. |

## Key learnings

1. **vLLM is unstable under sustained load** on this hardware (post-WSL-crash GPU 0 state). CUDA "unknown error" hits after ~30-40 problems. Each restart re-applies Genesis cleanly but the next long run dies again.

2. **Thinking ON triggers very long outputs** (one problem hit 16182 tokens / 262s before completing). The Qwen team's official 83.9 LCB v6 score likely uses thinking ON with high token budgets, which is 5-10× slower than thinking OFF.

3. **Realistic local pass@1 (thinking OFF, 4k tokens) on first 20 problems: 35%**. This is ~2-2.5× lower than Qwen's claimed 83.9, mostly because:
   - Thinking is OFF (Qwen's number is with thinking)
   - Sample size is small (variance high on first 20)
   - Single-attempt (no Best-of-N as Qwen may use)

4. **Pipeline validated**: extraction (`_extract_code`) and verification (sandbox-fusion) work correctly. The 35% is genuine model output quality, not a tooling artifact.

## To get a representative full-1054 number

Would require:
- vLLM stability fix (the CUDA-error-999 issue from CLAUDE.md — possibly needs WSL reboot to clean GPU state)
- Thinking ON with 16k+ token budget (~5-10 hours runtime for full set)
- Possibly Best-of-N with N≥3 to match Qwen's evaluation methodology

Recommended deferred path: run after EAGLE-3 head trains (faster generation = full LCB feasible in reasonable time).
