# Aider-style polyglot bench (local) — 2026-06-02

Answers "can we run aider bench locally?" — **yes**. Tooling: `bench/aider_lite.py`
(+ exercises at `~/polyglot-benchmark`, aider 0.86.2 in `~/bench_env`).

## Important: this is NOT the official Aider leaderboard protocol

| | Official Aider polyglot | `aider_lite.py` (this) |
|---|---|---|
| languages | 6 (225 exercises) | Python only (34) |
| edit format | unified diff | whole-file |
| attempts | **2** (2nd sees test errors) | **1** |
| reasoning | **ON** (leaderboard configs) | **OFF** (`enable_thinking:false`) |
| sandbox | Docker per-exercise | local pytest |

So our numbers are a **lower bound** and are not directly comparable to the public
leaderboard. They ARE a fair, fixed, apples-to-apples set for comparing our own
models/configs to each other.

## Result — Qwen3.6-35B-A3B (production daily driver, temp 0.6, thinking OFF)

**pass@1 = 8/34 = 23.5%** (single-attempt, whole-file, Python subset). Wall 318 s
(~9 s/exercise) on a single 3090, llama-server :8080.

Passed: affine-cipher, food-chain, hangman, list-ops, pig-latin,
simple-linked-list, two-bucket, variable-length-quantity.
Representative failures are **genuine** (harness verified): e.g. `react` produced a
clean 10 KB implementation that failed one callback-edge-case test (1 passed /
1 failed) — a real model bug, not an extraction artifact.

### Why this is low vs the model's 80.4 LCB / strong SWE-bench

- **Thinking OFF** — exercism polyglot problems are medium/hard; the 35B-A3B (3B
  active) leans on reasoning for multi-step logic. Official Aider configs run
  reasoning ON.
- **Single attempt** — the official 2nd attempt (with failing-test feedback)
  recovers many of these.
- **3B active params** — consistent with the research finding that the dense 27B
  beats the 35B-A3B on hard/agentic tasks (active-param count dominates).

The 23.5% is the realistic floor for **interactive thinking-OFF** use, mirroring
the LCB v6 thinking-OFF result (32% on the easier LCB set).

## 27B Dense vs 35B-A3B — head-to-head (same 34 exercises, identical config)

Both temp 0.6, thinking OFF, whole-file, single-attempt. 27B run via Madreag
turboquant llama-server (UD-Q4_K_XL) on GPU 0, **one model at a time** (35B stopped
during the 27B run, then restored) per the power-reset finding.

| Model | pass@1 | wall | t/s class |
|-------|-------:|-----:|-----------|
| Qwen3.6-35B-A3B (MoE, 3B active) | 8/34 = **23.5%** | 318 s (~9 s/ex) | ~100 t/s |
| **Qwen3.6-27B (Dense)** | 12/34 = **35.3%** | 1683 s (~49 s/ex) | ~30 t/s |

**The 27B Dense wins — +50% relative (12 vs 8 passes)** — empirically confirming the
research prediction that the all-active 27B beats the 3B-active MoE on hard
multi-step problems. Cost: ~5× slower wall (dense 27B ~30 t/s and longer outputs).

Caveats: single-attempt @ temp 0.6 has run-to-run variance (e.g. the 35B passed
food-chain/hangman/list-ops that the 27B missed, and vice versa); the 4-exercise
gap on n=34 is directional, not high-precision. The *direction* matches every other
signal (LCB v6, SWE-bench Pro, Terminal-Bench).

Per-exercise both-pass: affine-cipher, pig-latin, simple-linked-list, two-bucket.
27B-only passes: book-store, beer-song, bottle-song, dominoes, poker, proverb,
robot-name, zipper. 35B-only passes: food-chain, hangman, list-ops,
variable-length-quantity.

**Takeaway:** for hard one-shot coding, switch to the 27B (worth the slowdown);
for interactive speed, the 35B-A3B stays the daily driver — exactly the split the
research recommended.

### Still TODO for leaderboard-comparable numbers
A reasoning-ON + 2-attempt (diff-format, test-feedback) run on both models. Expect
both to jump substantially — thinking ON is where these models earn their LCB/SWE
scores.

## Reproduce

```bash
KEY=$(ps -eo args | grep -oP '\-\-api-key \K[A-Za-z0-9]+' | head -1)
~/bench_env/bin/python bench/aider_lite.py \
  --url http://localhost:8080 --model qwen3.6-35b-a3b --key "$KEY" --n 34 --out /tmp/aider_35b.json
```

## ⚠️ Operational: probable cause of the repeated hard resets this session

The machine hard-reset several times. The first was a confirmed CPU-thermal trip
during a `-j24` CUDA build. The **later** resets correlate with **GPU model-load
events**, and crucially **both 3090s were at the 350 W power limit, not the 200 W**
that `windows/gpu-init.bat` is supposed to set on boot. The fatal pattern was
loading a **second** model on GPU 1 while GPU 0 was serving — **two 3090s drawing
~350 W each simultaneously (~700 W GPU + ~140 W CPU)** likely tripped PSU/OCP →
instant reset. Mitigations:

1. **Run `windows/gpu-init.bat` on the Windows host** to cap both GPUs at 200 W
   (cannot be set from WSL — `nvidia-smi -pl` is host-only here).
2. **One GPU / one model at a time.** Never load a second model while another
   serves. For the 27B comparison: stop production, load 27B alone, bench, restart.
3. Keep heavy CPU builds at `-j6` (separate CPU-thermal issue).
