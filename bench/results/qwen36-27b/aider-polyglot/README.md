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

## Three-way: 27B Dense vs 35B-A3B vs Coder-30B-A3B (same 34 exercises, identical config)

All temp 0.6, thinking OFF, whole-file, single-attempt. Each run via llama-server
on GPU 0, **one model at a time** (production 35B stopped during each non-35B run,
then restored) per the power-reset finding.

| Model | active params | pass@1 | wall | t/s class |
|-------|:---:|-------:|-----:|-----------|
| **Qwen3.6-27B (Dense)** | **27B** | 12/34 = **35.3%** | 1683 s (~49 s/ex) | ~30 t/s |
| Qwen3.6-35B-A3B (MoE) | 3B | 8/34 = 23.5% | 318 s (~9 s/ex) | ~100 t/s |
| Qwen3-Coder-30B-A3B (MoE, code-tuned) | 3B | 8/34 = 23.5% | 182 s (~5 s/ex) | ~100 t/s |

**The 27B Dense wins outright (+50% relative). The two 3B-active MoEs tie at
exactly 23.5% — and crucially, the *coding-specialized* 30B coder does NOT beat the
general 35B-A3B.** This is the cleanest possible confirmation of the thesis:
**on hard multi-step problems, active-param count dominates — coding-specialization
does not overcome a 3B-active bottleneck.** The coder's only edge is speed (fastest
of the three, 182 s; short outputs, no thinking).

(Coder added 2026-06-03 via Qwen3-Coder-30B-A3B-Instruct UD-Q4_K_XL GGUF, turboquant
engine, qwen3moe arch. Its official sampling is temp 0.7/top_p 0.8; harness default
0.6/0.95 used for comparability — minor. Failures verified genuine, e.g. `grep`:
real SyntaxError, not a harness artifact. A cosmetic `~/.inputrc` readline warning
appears in stderr but does not affect the exit-code-based pass/fail.)

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

## Reasoning-ON experiment (35B-A3B) — thinking HURT here (2026-06-09)

Re-ran the 35B with `enable_thinking:true` (server `REASONING_BUDGET=24000`,
request `max_tokens=20000`), same 34 exercises, single attempt:

**pass@1 = 7/34 = 20.6% — *worse* than thinking-OFF (8/34 = 23.5%).** Wall 3397 s
(~100 s/ex, 10× slower).

Root cause (verified, not assumed): the 3B-active MoE **overthinks and never
converges** on several problems. On `affine-cipher` (a *trivial* exercise that
passes thinking-OFF), thinking-ON produced **57,021 chars of `reasoning_content`,
hit the 20k-token cap (`finish_reason: length`), and emitted an EMPTY answer**
(`content` len 0). Five exercises ran the full ~210 s (= 20k tokens @ ~95 t/s) and
failed the same way. So thinking didn't improve quality — it spiraled and truncated.

**The config bug:** `max_tokens` (20k) was hit *before* the reasoning budget (24k)
could force a `</think>`. The fix: set the **reasoning budget BELOW max_tokens** so
the model is forced to stop reasoning and answer.

### The fix works — forced-budget thinking (budget 6k < max 14k): 11/34 = 32.4%

Re-ran with `REASONING_BUDGET=6000`, `max_tokens=14000` (budget < max → server
injects `</think>` at 6k thinking tokens, leaving 8k for the answer). Verified
convergence first (affine-cipher: `finish_reason: stop`, real code emitted).

| 35B-A3B config | pass@1 | wall |
|---|---:|---:|
| thinking OFF | 8/34 = 23.5% | 318 s |
| naive thinking (budget 24k ≥ max 20k → spiral) | 7/34 = 20.6% | 3397 s |
| **forced-budget thinking (budget 6k < max 14k)** | **11/34 = 32.4%** | 2274 s |

**Thinking lifts the 35B-A3B +38% relative (8→11) — but ONLY with budget
discipline.** Naive thinking-ON *hurt* (spiral/truncation); forced-budget thinking
*helped* and nearly catches the thinking-OFF 27B Dense (12/34 = 35.3%).

### Five-way summary (single-attempt, whole-file, 34 python exercises)

| Model / mode | pass@1 |
|---|---:|
| **Qwen3.6-27B Dense, forced-budget thinking (6k<14k)** | **13/34 = 38.2%** ← best |
| Qwen3.6-27B Dense, thinking OFF | 12/34 = 35.3% |
| Qwen3.6-35B-A3B, forced-budget thinking | 11/34 = 32.4% |
| Qwen3.6-35B-A3B, thinking OFF | 8/34 = 23.5% |
| Qwen3-Coder-30B-A3B, thinking OFF | 8/34 = 23.5% |
| Qwen3.6-35B-A3B, naive thinking | 7/34 = 20.6% |

**The 27B Dense + forced-budget thinking is the champion (13/34 = 38.2%).** Thinking
helped the 27B too (+1 over its thinking-OFF 12/34), a smaller lift than the 35B's +3
— the all-active dense model was already strong and far less prone to the MoE
overthinking spiral. Confirms both theses at once: (1) active-param count dominates
hard coding, (2) forced-budget thinking is a real, safe quality lever on top. The 27B
thinking run survived a mid-run WSL restart at 27/34 thanks to incremental JSONL
persistence (7 remaining exercises re-run and merged; tail scored 5/7). Raw:
`aider_27b_think6k_combined.json`.

**Operational recommendation:** if you want thinking on the 35B-A3B, set
`REASONING_BUDGET` to ~6000 (NOT 0 and NOT ≥ your max_tokens) and keep max_tokens
≥ ~14000. Production left at `REASONING_BUDGET=0` (non-thinking default) per the
daily-driver preference; flip per-request with a sane budget when you want the
quality bump. Raws: `aider_35b_think.json` (naive), `aider_35b_think6k.json` (fixed).

### Still TODO
The official **2-attempt** diff-format protocol, and a forced-budget thinking run on
the **27B Dense** (~5 h at ~30 t/s; all-active, so likely the strongest config of all).

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
