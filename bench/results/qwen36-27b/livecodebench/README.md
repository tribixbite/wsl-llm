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
| `run_lcb.py` | HTTP-based LCB v6 runner |
| `lcb_smoke5.json` | 5-problem smoke test results |
| `lcb_qwen36_200.json` (when complete) | 200-problem subset, easy/medium/hard split |
| `lcb_qwen36_full.json` (if run) | All 1054 problems, comparable to Qwen's 83.9 number |
