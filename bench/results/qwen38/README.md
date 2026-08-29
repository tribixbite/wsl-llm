# Qwen 3.8 27B on a single/dual RTX 3090 — backend + TPS (2026-08-27)

Qwen3.8-27B (released ~Aug 14 2026) is a **hybrid GatedDeltaNet + gated-attention
VLM** (27.78B dense, 64 layers, 262k native ctx) with a **native MTP head**.
Official LCB v6 claim **90.3** (vs Qwen3.6-27B's 83.9).

Quants used: `unsloth/Qwen3.8-27B-GGUF` (Dynamic V3.0).

## ⚠️ Two things that will bite you

**1. You MUST rebuild llama.cpp (≥ commit `ece963f`).** Older builds load Qwen 3.8
fine — correct VRAM, normal speed — and emit **garbage tokens** from the broken
GatedDeltaNet CUDA kernel. Our April-vintage build predated the fix. After rebuilding
at latest master, output is clean (verified: correct iterative Fibonacci + coherent
explanation). If you `git pull`, also replace the installed `libggml*.so`/`libllama.so`
— a stale `.so` silently keeps the bug.

**2. UD-Q6_K_XL (25.30 GB) does NOT fit one 24 GB 3090.** It needs both GPUs
(`CUDA_VISIBLE_DEVICES=0,1 --split-mode layer` → 12.0 GB + 14.7 GB). For a single
card use **UD-Q5_K_XL (20.88 GB)**.

| Unsloth quant | size | single 3090? |
|---|---:|---|
| UD-Q6_K_XL | 25.30 GB | ❌ dual-GPU only |
| UD-Q5_K_XL | 20.88 GB | ✅ best single-card |
| UD-Q4_K_XL | 17.56 GB | ✅ long-context headroom |

## TPS results (streaming decode_TPS, 800-token gens, thinking OFF)

| Config | prose | code | json | **avg** |
|---|---:|---:|---:|---:|
| **Q6_K_XL, dual-GPU, +MTP** ⭐ | 41.1 | **59.3** | 60.0 | **53.5** |
| Q5_K_XL, single-GPU, +MTP | 29.2 | 39.6 | 39.4 | 36.1 |
| Q5_K_XL, single-GPU, no spec | 19.1 | 18.8 | 18.5 | 18.8 |

### The big lever: MTP speculative decoding (+92%)

`--spec-type draft-mtp` nearly **doubles** throughput (18.8 → 36.1 t/s). The MTP
draft head **ships inside the Unsloth GGUF** as `blk.64.nextn.*` — without the flag
llama.cpp logs `unused tensor blk.64.nextn.* -- ignoring` and you leave ~half your
speed on the table. No separate drafter download needed (the standalone
`mtp-Qwen3.8-27B-Q4_0.gguf` fetch 404s; it's redundant).

Dual-GPU Q6 adds another +48% over single-GPU Q5 (more layers on faster paths, and
no VRAM pressure), landing at **~60 t/s on code**.

## Recommended serve commands

```bash
# Best TPS + best quality (both GPUs, 25.3 GB split 12.0/14.7)
CUDA_VISIBLE_DEVICES=0,1 ~/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen38/Qwen3.8-27B-UD-Q6_K_XL.gguf --alias qwen3.8-27b \
  -c 16384 -np 1 -ngl 999 -fa on --split-mode layer \
  --spec-type draft-mtp \
  --temp 0.7 --top-p 0.8 --top-k 20 --host 127.0.0.1 --port 8084 --jinja

# Single-GPU (leaves GPU 1 free)
CUDA_VISIBLE_DEVICES=0 ~/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen38/Qwen3.8-27B-UD-Q5_K_XL.gguf ... --spec-type draft-mtp ...
```

Build (thermal-safe `-j6`, see `cpu-thermal-build-trips` memory):
```bash
cd ~/llama.cpp && git pull origin master   # must include ece963f
~/.local/bin/cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DLLAMA_CURL=OFF -DGGML_NATIVE=ON
~/.local/bin/cmake --build build --target llama-server -j6
```

## Sampling (official)

| Mode | temp | top_p | top_k | presence |
|---|---:|---:|---:|---:|
| Thinking (default) | 1.0 | 0.95 | 20 | 0.0 |
| Instruct / non-thinking | 0.7 | 0.80 | 20 | 1.5 |

Thinking is ON by default; control with `reasoning_effort` = `xhigh`/`medium`/`low`/
`none`. **Never greedy** (same guidance as prior Qwen gens).

## Quality: aider polyglot (python subset, whole-file)

Same harness as `../qwen36-27b/aider-polyglot/README.md`. Served on the
**Q6_K_XL dual-GPU + MTP** config above.

### ⭐ Headline: pass@2 = 22/34 = 64.7%

The aider leaderboard headlines **pass@2** (two tries, the second seeing the failing
pytest output) — not pass@1. On the champion config
(**Q6_K_XL dual-GPU + MTP + `reasoning_effort=medium`, `--tries 2`**):

| metric | score |
|---|---:|
| pass@1 (first attempt) | 11/34 = 32.4% |
| **pass@2 (leaderboard metric)** | **22/34 = 64.7%** |

**The retry loop doubles the score (32.4 → 64.7).** Handing the model its own test
failures is worth more than any sampling/thinking knob measured in this repo. Every
single-attempt number below is therefore a *floor*, not a comparable leaderboard
figure. Wall 3601 s for 34 exercises with 2 attempts.

### Single-attempt (pass@1) comparison across models

These runs used `--tries 1`; useful for model-vs-model comparison on equal terms,
but **not** comparable to published aider leaderboard numbers.

| Model / mode | pass@1 |
|---|---:|
| **Qwen3.8-27B, MTP + `reasoning_effort=medium`** | **14/34 = 41.2%** ← best overall |
| Qwen3.6-27B Dense, forced-budget thinking | 13/34 = 38.2% |
| Qwen3.6-27B Dense, thinking OFF | 12/34 = 35.3% |
| Qwen3.6-35B-A3B, forced-budget thinking | 11/34 = 32.4% |
| Qwen3.6-35B-A3B / Qwen3-Coder-30B-A3B, thinking OFF | 8/34 = 23.5% |
| **Qwen3.8-27B, thinking OFF** | **7/34 = 20.6%** ← worst |

### The headline finding: Qwen 3.8 is thinking-first, and it matters enormously

**Thinking mode DOUBLES Qwen 3.8's score (20.6% → 41.2%).** That is by far the
largest thinking delta we've measured — Qwen 3.6-27B only gained +2.9 pts from
thinking (35.3 → 38.2), and the 35B-A3B gained +8.9 (23.5 → 32.4).

Critically, **with thinking OFF Qwen 3.8 is the WORST model we've benchmarked
(20.6%) — worse than Qwen 3.6-35B-A3B and the Coder-30B.** If you evaluate Qwen 3.8
the way you'd evaluate a 3.6-generation model (thinking off for speed), you will
conclude it's a regression. In its intended mode it's the best model on this
hardware.

The thinking-OFF result was verified genuine (not a harness artifact):
`finish_reason: stop`, valid code block emitted, no truncation, no runaway reasoning
— the model simply produces weaker code when its reasoning is suppressed.

Unlike the Qwen 3.6-35B-A3B "overthinking spiral" (which hit the token cap and
emitted empty answers), Qwen 3.8 at `reasoning_effort=medium` **converges fast and
cleanly** (~30–120 s/exercise, well under the 12k-token cap). `reasoning_effort` is
the right control surface for this model — not a numeric `REASONING_BUDGET`.

**Operational recommendation: always run Qwen 3.8 with `reasoning_effort` ≥ medium
and `--spec-type draft-mtp`.** MTP recovers most of the speed cost of thinking
(59 t/s code), so you get both the quality and acceptable throughput.

### `xhigh` is NOT better than `medium` (preliminary)

A sweep at `reasoning_effort=xhigh` (Qwen 3.8's own default) with max_tokens 16000
ran far slower and scored no better on the overlapping exercises:

| | medium | xhigh |
|---|---|---|
| affine-cipher | PASS | PASS (193.8 s) |
| beer-song | PASS | **FAIL** (108.0 s) |
| book-store | — | ran >390 s, past the ~16k-token cap |
| per-exercise cost | ~30–120 s | ~110–220 s+ |

`xhigh` reproduces the **cap-truncation failure mode** documented for the
Qwen3.6-35B-A3B: on hard problems it reasons past the token budget and never emits
an answer. Combined with ~2–4× the wall time, **`medium` is the right operating
point** — matching the independent note in `CLAUDE.md` from the Legion machine
("`xhigh` costs ~220 s/exercise, use `medium`"). Raw (partial):
`aider_q38_q6_effort_xhigh.json`.

## ⭐ vLLM stack (`syv-ai/qwen38-27b-rtx3090`) — WORKS, and is the TPS champion

**105 t/s average, 116 t/s code, 124 t/s json — ~2× the best llama.cpp config**,
confirming the repo's ~114–121 tok/s claim. TTFT 132–171 ms.

| Config | prose | code | json | avg |
|---|---:|---:|---:|---:|
| **vLLM W4A16 + MTP n=4, `MAX_SEQS=1`** ⭐ | 74.5 | **116.4** | **124.1** | **105.0** |
| llama.cpp Q6_K_XL dual-GPU + MTP | 41.1 | 59.3 | 60.0 | 53.5 |
| llama.cpp Q5_K_XL single + MTP | 29.2 | 39.6 | 39.4 | 36.1 |

### The gotcha that made it look broken: `MAX_SEQS`

The first launch used the default **`max_num_seqs: 8`** at
`gpu_memory_utilization 0.93` (≈22.3 GB of 24 GB). It loaded, LISTENed, logged
`Application startup complete` — then **0% GPU util and every request timed out**.

That is **not** an engine bug. It is the exact **WSL2 VRAM-eviction trap** the
Legion notes document for llama.cpp's `--parallel 4`: WSL2 has no OOM guardrail, so
**WDDM silently evicts the weights to system RAM while `/health` keeps answering**.
Setting **`MAX_SEQS=1`** fixed it outright — same VRAM figure (21.9 GB), but the
model stays resident and serves at 105 t/s.

**Lesson (straight from `docs/QWEN38_27B_LEGION_BENCHMARKS.md`): never trust
"it loaded" — validate every config with a *timed generation*.** A health probe
cannot distinguish a healthy server from an evicted one. This cost us an incorrect
"vLLM is unreliable on this box" conclusion, since the earlier 0.17/0.22 failures
may well have been the same eviction rather than engine instability.

### Install obstacles cleared (none was the real blocker)



| Obstacle | Fix |
|---|---|
| Docker could not select the `nvidia` device driver | `nvidia-container-toolkit` is not installed (root). Switched to the venv path. |
| Docker `credsStore: desktop.exe` breaks pulls in WSL | removed `credsStore` from `~/.docker/config.json` (backup at `.bak`) |
| README assumes Python 3.12; venv built 3.10 | patch against `venv/lib/python3.10/site-packages/vllm` |
| 4/19 patches rejected (written for vLLM 0.27.1, installed 0.28.0) | all 4 are DFlash2/vision; ran MTP n=4 instead |
| `NameError: name 'envs' is not defined` in `qwen3_vl.py` | the rejected `vision-tower-cpu-offload.patch` applied its *usage* hunks but not the import — added `import vllm.envs as envs` |
| Docker's prepare ran as root → model tree mode-600 root-owned, unreadable | re-downloaded + requantized into a user-owned `models2/` (no sudo needed) |

Reproduce the working-but-wedging install: venv + `pip install vllm`, apply the 15
compatible patches, `MODEL=.../models2/Qwen3.8-27B-W4A16-AutoRound bash
single-user/start_qwen.sh`. If vLLM is ever fixed on this box, that is the command.

## Not tested here

The `syv-ai/qwen38-27b-rtx3090` vLLM W4A16 stack claims ~114 t/s single-user /
~130 with DFlash2 spec-decode at ~23 GiB — roughly 2× our best llama.cpp number and
the likely true throughput champion. Untested so far; the GGUF path was chosen first
because it reuses our existing stack and both quants were already downloaded.

## Files

| File | What |
|---|---|
| `tps.tsv` | Raw streaming decode-TPS rows for all three configs |
| `aider_q38_q6.json` | Aider polyglot (python subset) score — see `../qwen36-27b/aider-polyglot/README.md` for protocol + Qwen 3.6 comparison |

## ⭐ Multi-language pass@2 — the real number: 73.0%

Single-language (Python-only) numbers badly understated this model. Running the
same 2-attempt protocol across two languages on the **vLLM W4A16 + MTP,
MAX_SEQS=1** stack (105 t/s):

| Language | n | pass@1 | pass@2 |
|---|---:|---:|---:|
| JavaScript | 40 | 57.5% | **90.0%** |
| Python | 34 | 32.4% | 52.9% |
| **Overall** | **74** | 45.9% | **73.0%** |

**This clears ">100 t/s at >64%": 105 t/s and 73.0% pass@2 simultaneously.**

Python is by far the harder half of this set for Qwen 3.8 — JS is +37 pts. Any
Python-only figure (our earlier 50.0% / 52.9%) is not representative. The n=74
sample also tightens the estimate a lot versus the n=34 runs, whose 95% CI spanned
~24 points.

Harness: `bench/aider_multi.py` (python/javascript/java). JS needed a babel fix —
presets cannot resolve through a symlinked `node_modules`, so the runner rewrites
`babel.config.js` with `require.resolve`. **Before that fix JS scored a false 0%**,
which would have inverted this conclusion.
