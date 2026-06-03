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

## Pending: 27B Dense comparison

Not yet run — requires a model swap (stop 35B, load 27B alone on GPU 0; **one
model at a time** per the power-crash finding below). Research predicts the 27B
wins on hard problems; an empirical `aider_lite` 27B number would confirm. A
reasoning-ON + 2-attempt run on both models is the way to get leaderboard-comparable
numbers.

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
