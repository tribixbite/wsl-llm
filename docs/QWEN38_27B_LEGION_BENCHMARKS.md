# Qwen3.8-27B on a Legion RTX 5080 Laptop (16 GB, Blackwell sm_120)

**Date:** 2026-08-27 · **Model:** `unsloth/Qwen3.8-27B-GGUF` → `Qwen3.8-27B-UD-Q3_K_XL.gguf` (12.24 GiB)
**Engines compared:** llama.cpp (WSL2 + native Windows) · ExLlamaV3 (native Windows) · vLLM (ruled out, see §7)

> This is a **different machine** from the one `docs/QWEN36_BENCHMARKS.md` describes.
> That report is the 2×RTX 3090 desktop (sm_86). This one is the Legion laptop.

---

## 0. Headline results

| Finding | Impact |
|---|---|
| **Fn+Q → Performance mode** raised the GPU from a ~90 W cap to 175 W | **+32% decode, +42% prefill** |
| **`--parallel 1` is mandatory** — the default of 4 slots overshoots VRAM and WDDM silently evicts the model to system RAM | avoids a **~700× collapse** (39.8 → 0.04 t/s) |
| **MTP speculative decoding** (the 1.28 GiB draft head Unsloth ships) | **1.89× overall, 2.14× on code** — 39.8 → 75.3 t/s |
| MTP costs **no measurable accuracy** (n=102/arm, p=0.59) | the speedup is effectively free |
| **WSL2 ≈ native Windows** for fully-GPU-resident inference | within ±2% — no reason to migrate |
| **Thinking mode** (`reasoning_effort=medium`) | **38.2% vs 20.6% pass@1**, p=0.040 |
| Context is nearly free — only 16 of 64 layers hold a KV cache | 64k ctx costs ~2.5% decode |

**Recommended daily driver:**

```bash
~/llama.cpp/build/bin/llama-server \
  -m ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf \
  -ngl 99 -fa on -c 32768 -ctk q4_0 -ctv q4_0 --parallel 1 \
  --spec-type draft-mtp -md ~/models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
  --jinja --reasoning-effort medium \
  --host 127.0.0.1 --port 8080
```
→ **~74 t/s, 14,704 MiB peak, 1.6 GiB headroom, 32k context, 38.2% pass@1.**
Leave thinking ON: disabling it drops pass@1 from 38.2% to 20.6%.

---

## 1. Hardware

| | |
|---|---|
| GPU | RTX 5080 **Laptop**, 16,303 MiB, GDDR7 256-bit, **sm_120 (compute_cap 12.0)** |
| Bandwidth | 14001 MHz × 2 × 256 bit / 8 = **896 GB/s** |
| Power | default 80 W · **Quiet/Balanced ≈ 65–95 W** · **Performance = 175 W** |
| CPU / RAM | Intel Core Ultra 9 275HX (24c) · 62 GiB DDR5 |
| Driver | 610.57.01, CUDA UMD 13.3; toolkits 12.6 + 12.8 |
| OS | WSL2 Ubuntu 22.04 (kernel 5.15) + Windows 11 26220 |
| Display | driven by the **Intel iGPU** — the 5080 reports `Display Active: Disabled`, so all 16 GB is usable |

**Theoretical decode ceiling** for the 13.146 GB file: `896 / 13.146 = 68.2 t/s`.
At 175 W we measure 40.6 t/s without speculation = **60% efficiency** (normal for llama.cpp).
At the ~90 W cap it was 30.7 t/s = 45%, because the **memory clock throttled 14001 → 9001 MHz**.

---

## 2. Model architecture (read from the GGUF and upstream `config.json`)

`Qwen/Qwen3.8-27B` → `model_type: qwen3_5`; the GGUF declares **`general.architecture = qwen35`**,
which mainline llama.cpp already supports as `LLM_ARCH_QWEN35`. **No fork is required** — unlike
Qwen3.6 on the desktop box, which needed the Madreag turboquant build.

| Property | Value |
|---|---|
| Layers | 64 (+1 MTP) — **16 full-attention, 48 linear (gated DeltaNet)**, `full_attention_interval=4` |
| Attention | 24 Q / 4 KV heads, **head_dim 256**, `attn_output_gate=true` |
| Hidden / FFN | 5120 / 17408 · vocab **248,320** (untied lm_head) |
| Context | 262,144 native; interleaved M-RoPE, `partial_rotary_factor=0.25` |
| MTP | `mtp_num_hidden_layers=1`, shipped separately as `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` |
| Vision | 27-layer tower (out of scope here) |

**KV cache is cheap because only 16 layers have one:**
`16 × 4 kv heads × 256 × 2 (K+V) × 2 B = 64 KiB/token` at f16 — about 4× cheaper than a
conventional dense 27B. The 48 DeltaNet layers hold a *constant* ~150 MiB recurrent state
per sequence, independent of context length.

### ⚠️ Quant quality caveat — verified directly in the file
Unsloth's Dynamic-V3 rebuild of `UD-Q3_K_XL` contains **24 tensors in 2-bit classes totalling
2,002,780,160 params = 7.33% of the model** (IQ2_S×15, IQ2_XS×4, Q2_K×3, IQ2_XXS×2). The earlier
revision `408fcc18` (12.52 GiB) reportedly had none. If accuracy matters more than 0.28 GiB,
pin that revision. This was confirmed by enumerating tensor dtypes, not taken from a forum post.

---

## 3. The two traps that cost ~700× and ~32%

### 3a. `--parallel` default → silent VRAM eviction

`llama-server` defaults to **`--parallel 4`**. Each slot carries its own DeltaNet recurrent
state and compute buffers:

| slots | peak VRAM | decode |
|---|---:|---:|
| 4 (default) | **15,941 MiB** / 16,303 | **0.04 t/s** |
| 1 | 13,384 MiB | **27.0 t/s** |

At 15.9 GiB we crossed the ceiling and **WDDM evicted the weights to system RAM rather than
raising OOM**. Observed VRAM trace: `13376 → 15941 → 2910 MiB` while `/health` still returned
`{"status":"ok"}`; prompt eval fell to 0.28 t/s and the server died ~100 s later.

WSL2 **ignores** NVIDIA's "Prefer No Sysmem Fallback" control ([WSL#11050](https://github.com/microsoft/WSL/issues/11050)),
so there is no OOM guardrail. **Never trust "it loaded"** — validate every config with a timed
generation. `bench/legion/find_max_context.sh` does this automatically.

ExLlamaV3 has the identical trap: its `Cache` defaults to `max_batch_size=16` and failed with
*"Insufficient VRAM in split for model and cache"* until set to 1.

### 3b. Laptop power mode

`nvidia-smi -pl` is rejected on laptop GPUs ("not supported in current scope") and the Lenovo
`LENOVO_GAMEZONE_DATA` WMI class returns *Access denied* unelevated — so this must be changed
by hand (Fn+Q, or Legion Space → Performance).

| | 90 W cap | 175 W | gain |
|---|---:|---:|---:|
| tg128 | 30.74 t/s | **40.64 t/s** | +32% |
| pp512 | 949.9 t/s | **1353.6 t/s** | +42% |
| memory clock | throttles to 9001 MHz | holds **14001 MHz** | — |
| run-to-run σ | ±1.35 | **±0.05** | far more stable |

Peak draw reached 159.9 W at only 58 °C — this machine is **power-limited, never thermally
limited**, so there is headroom left. Log `enforced.power.limit` with every measurement: it
was observed drifting between 65 W and 95 W before the mode change.

---

## 4. llama.cpp configuration matrix (175 W, `--parallel 1`, `-fa on`, CUDA graphs on)

Decode t/s is the mean of the three house prompts (prose/code/json), streaming, TTFT excluded.

| config | decode | peak VRAM | notes |
|---|---:|---:|---|
| baseline 32k q8_0 | 39.81 | 13,388 | reference |
| **MTP 16k q8_0** | **75.77** | 14,512 | fastest; 1.8 GiB headroom |
| **MTP 64k q4_0** | **74.65** | 15,570 | fastest at long context, 0.7 GiB headroom |
| **MTP 32k q4_0** | **74.09** | 14,704 | ⭐ best balance |
| MTP 32k q8_0 | 75.25 | 15,216 | 1.1 GiB headroom |
| MTP 64k q8_0/q4_0 | 56.55 | **15,954** | ⚠️ VRAM cliff — 25% slower at 0.3 GiB slack |
| ngram-mod 32k | 39.56 | 13,388 | no gain |
| ngram-simple 32k | 38.95 | 13,390 | no gain |
| ngram-map-k 32k | 38.76 | 13,388 | no gain |
| baseline 32k f16 KV | 39.88 | 14,262 | KV dtype barely affects speed |
| baseline 64k q8_0/q4_0 | 38.83 | 14,124 | |
| baseline 64k q4_0 | 38.70 | 13,612 | |

**Conclusions**

1. **MTP is the only speculative method that works here.** All three n-gram variants gave
   nothing on these prompts. MTP: **1.89× overall, 2.14× on code** (39.8 → 85.0 t/s on the
   code prompt). 85 t/s legitimately exceeds the 68.2 t/s bandwidth ceiling — which is the
   point of speculative decoding: multiple tokens per weight read.
2. **KV cache dtype is a quality/context knob, not a speed knob** — f16 39.88 vs q4_0 38.70.
   Pick it for headroom, not throughput.
3. **The VRAM cliff is real and gradual.** `mtp_64k_q8k_q4v` at 15,954 MiB ran 25% slower than
   the same config at 15,570 MiB. Keep ≥1 GiB slack.

### Context scaling (pre-Performance-mode, so absolute numbers are lower)

Decode is essentially flat from 8k to 64k — 27.18 → 27.33 t/s — because only 16 of 64 layers
have a growing cache. 64k ctx costs 13,536 MiB. **Context is not the constraint on this card.**

---

## 5. Engine comparison

All at 175 W, single stream, same three prompts, TTFT excluded.

| engine | quant | size | decode | VRAM | notes |
|---|---|---:|---:|---:|---|
| llama.cpp (WSL2) | GGUF UD-Q3_K_XL | 12.24 GiB | 39.81 | 13,388 | baseline |
| llama.cpp (Windows) | same | same | ~41.0 | — | `llama-bench` tg128 41.03 vs WSL 40.64 |
| **ExLlamaV3 (Windows)** | EXL3 3.0bpw | 12.53 GiB | **44.07** | 13,414 | +11% over llama.cpp at equal VRAM |
| **llama.cpp + MTP (WSL2)** | GGUF UD-Q3_K_XL | +1.28 GiB | **75.25** | 15,216 | ⭐ winner |

**WSL2 vs native Windows** (llama-bench, identical model and flags):

| | WSL2 (cb30059) | Windows (b10659) |
|---|---:|---:|
| pp512 | 1353.58 ± 30.72 | 1325.28 ± 39.17 |
| tg128 | 40.64 ± 0.05 | 41.03 ± 0.15 |

Within ±2% — a wash. (Binaries are same-day but not the identical commit.)

**ExLlamaV3 notes.** It is genuinely faster per GB than llama.cpp, but there is no EXL3 MTP
draft head, so it cannot match llama.cpp+MTP. Getting it running on Windows required two
undocumented fixes: `setuptools` (missing → `ModuleNotFoundError`) and **`triton-windows`**
(without Triton, `exllamav3.modules.attention_fn.dsa_triton` fails to export its kernels and
the whole package fails to import). Only `SC_3.00bpw_H4` (12.53 GiB) fits 16 GB —
3.50bpw is 14.29 GiB and 4.00bpw is 15.70 GiB.

---

## 6. Accuracy

Harness: `bench/aider_lite.py` — 34 Exercism Python exercises, whole-file, single attempt,
local pytest. Directly comparable to the existing 27B/35B-A3B datapoints in this repo.

### ⚠️ Single runs of this benchmark are not trustworthy
Sampling is temp 0.6 / top_p 0.95 with no seed. **The same baseline config scored 29.4%,
17.6%, and 14.7% on three runs.** A first-pass comparison suggested MTP halved quality
(29.4% vs 14.7%); pooling three runs per arm shows that was noise.

| arm | runs | pooled | rate |
|---|---|---:|---:|
| baseline (no MTP) | 10/34, 6/34, 5/34 | 21/102 | **20.6%** |
| MTP | 5/34, 8/34, 5/34 | 18/102 | **17.6%** |

Two-proportion z-test: **z = 0.53, p = 0.59 — no significant difference.**
**MTP's 1.89× speedup costs no measurable accuracy.**

### Is MTP bit-exact? No — but it doesn't matter downstream
A greedy probe (temp 0, top_k 1, 5 prompts) with a proper control:
- baseline vs baseline rerun → **byte-identical** (the engine *is* deterministic)
- baseline vs MTP → identical on 4/5 prompts; diverged on one at an **end-of-sequence
  decision** (baseline stopped at 71 chars, MTP continued to 697)

So llama.cpp's MTP is not distribution-preserving — it changes numerics enough to flip
near-tie tokens (batch-shape-dependent FP reduction order). The pooled quality test above is
what settles whether that matters: it does not.

### Thinking mode is the single biggest accuracy lever

Qwen3.8 is **thinking-first** (`reasoning_effort` defaults to `xhigh`). At `xhigh` an exercise
cost ~220 s, so `medium` is the practical setting.

| config | pass@1 | |
|---|---:|---|
| non-thinking (pooled, n=102) | 20.6% | |
| **thinking, `reasoning_effort=medium`, MTP on** | **38.2%** (13/34) | **z=2.06, p=0.040 — significant** |

**Thinking nearly doubles pass@1.** Given Qwen ships thinking on by default, do not disable it
for coding work — `--reasoning-budget 0` is a throughput optimisation that costs a lot of accuracy.

### Cross-machine comparison (same harness, same 34 Python exercises)

| model / config | pass@1 | machine |
|---|---:|---|
| **Qwen3.8-27B UD-Q3_K_XL, think(med)+MTP** | **38.2%** | **RTX 5080 Laptop 16 GB** |
| Qwen3.6-27B | 35.3% | 2×RTX 3090 (24 GB, Q4) |
| Qwen3.6-35B-A3B, forced-budget thinking | 32.4% | 2×RTX 3090 |
| Qwen3.8-27B UD-Q3_K_XL, non-thinking | 20.6% | RTX 5080 Laptop |

The 16 GB laptop at a 3-bit quant edges out the 24 GB desktop's Qwen3.6-27B at 4-bit — though
with n=34 the 38.2% vs 35.3% gap is well inside the noise band this benchmark exhibits (see the
warning above), so read it as "comparable", not "better".

---

## 7. Why vLLM was ruled out

Not a preference — arithmetic. **Every 4-bit safetensors quant of this model exceeds 16 GB:**

| repo | format | size |
|---|---|---:|
| `Frozenlock/Qwen3.8-27B-int4-AutoRound` | AutoRound INT4 | 17.69 GiB |
| `RedHatAI/Qwen3.8-27B-INT4` | W4A16 | 18.12 GiB |
| `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` | AWQ INT4 | 18.17 GiB |
| `unsloth/Qwen3.8-27B-NVFP4` | NVFP4 | 21.81 GiB |
| `Qwen/Qwen3.8-27B-FP8` | FP8 | 28.75 GiB |

They land at 17.7–21.8 GiB because, unlike GGUF k-quants, they keep embeddings, lm_head and
norms at higher precision — and this model's untied 248,320 × 5120 embed+output is ~2.5 GiB at
BF16 by itself. There is also an open vLLM issue for silent hangs on exactly the RTX 5080 16 GB.
The same VRAM arithmetic rules out SGLang. **GGUF k-quants are what make a 27B fit this card at all.**

---

## 8. Reproducing

```bash
scripts/fetch-qwen38.sh                      # resilient download (Xet disabled; WSL2 DNS kills it)
bench/legion/find_max_context.sh             # max context that stays VRAM-resident
bench/legion/run_llamacpp_matrix.sh          # KV / speculative / context matrix
CONFIGS_FILE=bench/legion/configs_mtp_sweep.txt bench/legion/run_llamacpp_matrix.sh
bench/legion/verify_mtp_equivalence.sh       # greedy equivalence + determinism control
bench/legion/run_quality.sh                  # aider_lite pass@1
bench/legion/run_windows_bench.sh            # native Windows via WSL interop
```

Build (CUDA 12.8 is the hard minimum for sm_120; cmake auto-promotes `120` → `120a`):

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-cli llama-bench llama-perplexity -j 20
```

`GGML_CUDA_FA_ALL_QUANTS=ON` matters: without it, flash attention is disabled entirely when
`K type != V type`, silently killing the asymmetric `-ctk q8_0 -ctv q4_0` config.

**Contrary to research, CUDA graphs should stay ON.** llama.cpp#27330 reports Xid 8 hangs on
laptop Blackwell; none occurred here across hours of runs, and graphs are worth **+20%**
(30.74 vs 25.55 t/s).

---

## 9. Open items / not verified

- Thinking-mode pass@1 at `xhigh` (too slow: ~220 s/exercise). `medium` is measured at 38.2%.
- The thinking run was interrupted by a machine reboot at 24/34 and resumed for the
  remaining 10 exercises; `aider_lite` appends per-exercise JSONL, so the merge is exact.
- No KL-divergence vs a Q8_0 baseline yet. Note the logits file would be ~23.6 GiB at
  `--chunks 200` given the 248k vocab, so `--chunks` must be bounded.
- `UD-Q3_K_XL` revision `408fcc18` (no 2-bit tensors) not benchmarked against current.
- Official aider-polyglot (diff format, multi-language) not run — only the whole-file Python subset.
- ExLlamaV3 accuracy not measured (would need TabbyAPI to expose an OpenAI endpoint).
- Ollama / LM Studio not benchmarked; both vendor llama.cpp and should track it minus overhead.
- `wsl --update` (2.4.13 → 2.7.11, Blackwell CUDA-graph fixes) not yet applied.
