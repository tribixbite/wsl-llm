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
| **Thinking mode** (`reasoning_effort=medium`) | **58.8% pass@2**, 26.5% pass@1 |
| **pass@2 is ~2.2× pass@1 in both arms** | thinking 26.5%→**58.8%**, non-thinking 17.6%→**38.2%** |
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
→ **~74 t/s, 14,704 MiB peak, 1.6 GiB headroom, 32k context, 58.8% pass@2.**
Leave thinking ON: with thinking off, pass@2 falls 58.8% → 38.2%.

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

### Quant composition — verified directly in the file
Unsloth's Dynamic-V3 rebuild of `UD-Q3_K_XL` contains **24 tensors in 2-bit classes totalling
2,002,780,160 params = 7.33% of the model** (IQ2_S×15, IQ2_XS×4, Q2_K×3, IQ2_XXS×2); the earlier
`408fcc18` (12.52 GiB) has **zero**. Both confirmed by enumerating tensor dtypes, not taken from
a forum post.

Users concluded from this that V3 must be worse. **Measured, it is not** — see §11: against a
Q8_0 reference V3 has lower mean KLD (0.0248 vs 0.0271), higher top-1 agreement (93.1% vs 92.8%)
and lower RMS Δp, while also being smaller and faster. **Use the current revision; do not pin
`408fcc18`.**

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

### pass@2 — the aider leaderboard metric

The official aider polyglot benchmark gives the model **two attempts**: on failure the test
output is fed back into the same conversation and the model gets one chance to fix it. The
leaderboard headlines `pass_rate_2`. Our harness was single-attempt until now; `--tries 2`
implements the real protocol.

| arm | pass@1 | **pass@2** | recovered by the retry |
|---|---:|---:|---:|
| **thinking, `reasoning_effort=medium`** | 9/34 = 26.5% | **20/34 = 58.8%** | +11 |
| non-thinking (`--reasoning-budget 0`) | 6/34 = 17.6% | **13/34 = 38.2%** | +7 |

**The single most important methodological point in this report: pass@2 is ~2.2× pass@1 in
both arms.** A single-attempt score understates this model's practical coding ability by more
than half, because most failures are near-misses that one round of test feedback repairs — and
that is exactly how the model gets used in practice, since you paste the error back.

Thinking recovered 11 of 25 first-attempt failures: beer-song, book-store, bottle-song,
food-chain, grade-school, list-ops, poker, proverb, sgf-parsing, simple-linked-list, zipper.

**Statistical honesty.** Thinking leads on both metrics, but at n=34 per arm the pass@2 gap
(58.8% vs 38.2%) is **z=1.70, p=0.089 — not significant at the 0.05 level.** The direction is
consistent with the pass@1 result (26.5% vs a pooled non-thinking 20.6% over n=102, p=0.040),
so thinking is very likely better, but this particular comparison is underpowered. Given the
run-to-run spread this benchmark exhibits (a single config scored 14.7–29.4% across three
runs), treat the pass@2 arms as "thinking is probably meaningfully better" rather than proven.

Config for both arms: 32k ctx, q8_0 KV, `--parallel 1`, **no MTP**, CUDA graphs off (§10).
The thinking arm ran across three legs due to machine reboots; `aider_lite` appends
per-exercise JSONL so the merge is exact.

### Cross-machine comparison (same harness, same 34 Python exercises)

| model / config | pass@1 | machine |
|---|---:|---|
| **Qwen3.8-27B UD-Q3_K_XL, think(med) — pass@2** | **58.8%** | **RTX 5080 Laptop 16 GB** |
| Qwen3.8-27B UD-Q3_K_XL, think(med)+MTP — pass@1 | 38.2% | RTX 5080 Laptop 16 GB |
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

## 9. Open items — status

| item | status |
|---|---|
| Official aider-polyglot (diff format, multi-language) | **done** — §14: 63.3% pass@2, 100% well-formed |
| KL-divergence vs a Q8_0 reference | **done** — §11 |
| `UD-Q3_K_XL` revision `408fcc18` comparison | **done** — §11/§12: current V3 wins |
| ExLlamaV3 accuracy | **done** — §13: 38.2% pass@2, identical to llama.cpp |

Still not done:
- Thinking-mode pass@1 at `xhigh` (~220 s/exercise — impractical here; `medium` is measured).
- Ollama / LM Studio not benchmarked; both vendor llama.cpp and should track it minus overhead.
- `wsl --update` (2.4.13 → 2.7.x) not applied — deferred, and not the cause of §10.
- Tool-calling correctness (BFCL / `test-chat-auto-parser`) not exercised; the polyglot run's
  100% well-formed diff rate is adjacent evidence but not the same thing.
- KLD used `--chunks 100` (11.8 GiB of logits); more chunks would tighten the tail percentiles.

## 10. Machine stability — memory corruption, not a driver or the workload

This box bugchecked **five times** (twice before this work began). Symbol resolution via
`kd.exe -z <dump> -c "!analyze -v"` on all five dumps:

| dump | bugcheck | module | process | failure bucket |
|---|---|---|---|---|
| 8/12 5:59 AM | 0x1E | `nt` | chrome.exe | `AV_R_nt!PpmEventAddAffinityMaskAsSubset` |
| 8/27 4:58 PM | 0x1E | `nt` | svchost.exe | `AV_nt!ExpPoolTrackerChargeEntry` |
| 8/27 7:40 PM | 0x1E | `nt` | Registry | `AV_R_nt!RtlRaiseStatus` |
| 8/27 8:47 PM | 0x3B | `clipsp` | svchost.exe | `AV_clipsp!unknown_function` |
| 8/27 9:50 PM | 0x3B | `clipsp` | svchost.exe | `AV_clipsp!unknown_function` |

**Diagnosis: memory corruption.** Access violations scattered across unrelated kernel
subsystems (power management, pool tracking, exception dispatch, licensing) hitting unrelated
processes (chrome, svchost, Registry) is the signature of bad memory, not one buggy driver.

**The hardware is the prime suspect.** This laptop carries an aftermarket **128 GB kit — two
64 GB Crucial `CT64G56C46S5.M16B1` DDR5 SODIMMs at 5600 MT/s**, one per channel. Two
*dual-rank* 64 GB modules at full 5600 is a marginal load for an Arrow Lake HX memory
controller, which typically wants a derate to 5200 or 4800 at that population.

Corroborating: **no WHEA errors are logged** — consumer SODIMMs have no ECC, so corruption is
silent and surfaces as random access violations rather than recorded hardware faults. Windows
Memory Diagnostic has never been run on this machine.

**Ruled out by experiment.** Crashes continued after every one of these:
- peak VRAM cut from 15.2 GiB to 13.4 GiB (WDDM eviction pressure)
- CUDA graphs disabled (`GGML_CUDA_DISABLE_GRAPHS=1`)
- thermal backoff plus a 20 s inter-exercise cooldown; GPU never exceeded 81 C
- no WHEA/thermal-trip events at any point

> **Correction.** An earlier revision of this document claimed three crashes faulted at the
> same instruction because their addresses shared the low digits `96710`. Symbol resolution
> disproves that: they are in different modules and different functions. The shared digits
> were coincidence. The conclusion "one repeating driver bug" was wrong.

**Recommended, in order:** MemTest86 from USB (Windows' own diagnostic is too weak for
marginal timing faults) → if it fails, derate memory to 5200/4800 in BIOS. `wsl --update`
(this host is on **2.4.13**; 2.7.0 carries Blackwell CUDA-graph fixes) and an NVIDIA driver
update are worthwhile hygiene but do **not** address this.

**Why it did not cost us the results:** `aider_lite.py` appends one JSONL record per exercise,
so a killed run resumes exactly where it died. The 34-exercise thinking run completed across
three legs and merged exactly. Any long run on this machine should be checkpointed the same way.


---

## 11. Quantization damage: KL-divergence vs a Q8_0 reference

Reference: `Qwen3.8-27B-Q8_0.gguf` (27.05 GiB, partially CPU-offloaded since it exceeds 16 GB).
wikitext-2 test, `-c 512 --chunks 100`, `-fa on`, seed 1337. Q8_0's own KLD to FP16 is ~0.0014,
a few percent of what is measured here, so it is a defensible stand-in for BF16 (50.9 GiB, won't fit).

**Q8_0 reference PPL = 6.7480 ± 0.1031.**

| metric (vs Q8_0) | **V3** (24× 2-bit tensors, 12.24 GiB) | **408fcc18** (0× 2-bit, 12.52 GiB) | better |
|---|---:|---:|---|
| Mean PPL(Q) | 6.8421 | 6.8810 | **V3** |
| PPL(Q)/PPL(base) | **1.0139** | 1.0197 | **V3** |
| **Mean KLD** | **0.02484 ± 0.00049** | 0.02710 ± 0.00044 | **V3** |
| Median KLD | **0.01066** | 0.01222 | **V3** |
| 90% KLD | **0.05094** | 0.05550 | **V3** |
| 95% KLD | **0.08239** | 0.09201 | **V3** |
| 99% KLD | **0.24983** | 0.27338 | **V3** |
| 99.9% KLD | 0.90142 | **0.86048** | 408 |
| Maximum KLD | 5.9301 | **3.6762** | 408 |
| RMS Δp | **4.452 ± 0.099 %** | 4.739 ± 0.095 % | **V3** |
| **Same top-1 token** | **93.145 ± 0.158 %** | 92.757 ± 0.162 % | **V3** |

### The community complaint about Dynamic V3 is not supported

Users reported the V3 rebuild as "sloppier, more error-prone while coding" because it introduced
**24 tensors in 2-bit classes covering 7.33% of parameters** (which we verified directly in the
file). Measured against a Q8_0 reference, the opposite holds:

- **V3 tracks the reference better at every percentile from 0.1% through 99%**, on the mean, on
  RMS Δp, and on same-top-1-token agreement. The mean-KLD gap (0.0248 vs 0.0271, ~9%) is well
  outside the error bars.
- 408fcc18 wins **only in the extreme tail** — max KLD 3.68 vs 5.93, and 99.9% KLD. So V3 does
  suffer occasional larger single-token divergences, which is presumably what users noticed, but
  its typical-case fidelity is better.
- V3 is also **0.28 GiB smaller and ~17–30% faster to decode** (§12).

For context, llama.cpp's own Llama-3-8B scoreboard puts imatrix `q4_K_M` at mean KLD 0.0282 and
`q3_K_M` at 0.0844. **This 3-bit-class quant lands at 0.0248 — better than the q4_K_M reference
band**, which is what Unsloth's dynamic per-tensor mixing plus imatrix calibration buys.

**Recommendation: use the current `UD-Q3_K_XL`.** Do not pin `408fcc18` — it is larger, slower,
and measurably further from the reference in the typical case.

---

## 12. The two Q3_K_XL revisions differ in speed, not just size

| | V3 | 408fcc18 |
|---|---:|---:|
| size | 12.24 GiB | 12.52 GiB |
| decode t/s (median, non-thinking) | **34.6** | 28.8 |
| generation length (median tokens) | **620** | 1131 |
| responses hitting the 3000-token cap | 2/62 (3%) | 4/23 (17%) |

Cause is the tensor-type mix, not size:

| dtype share | V3 | 408fcc18 |
|---|---:|---:|
| IQ4_XS | 34.3% | 48.0% |
| IQ3_S | 30.1% | 42.4% |
| K-quants (Q3_K/Q4_K/Q5_K/Q6_K/Q8_0) | 19.5% | 9.6% |
| 2-bit i-quants | 7.3% | 0% |

408fcc18 is **90.4% i-quant**. I-quants dequantize through codebooks and are slower on CUDA than
K-quants, so the revision that avoids 2-bit tensors pays for it in throughput. Chat templates are
byte-identical between the two (sha1 `a7e79f8fe37f381c`), ruling that out as a confound.

The 408 revision was also markedly more verbose — on `rust/forth` it failed to converge within
35 minutes where V3 finished the same exercise in 232 s (both failing the tests).


---

## 13. ExLlamaV3 vs llama.cpp on quality — a dead heat

Speed alone does not decide an engine. Both were scored with the **same harness**
(`bench/aider_lite.py --tries 2`, non-thinking, 34 Python exercises) by putting a minimal
OpenAI-compatible shim over ExLlamaV3 (`bench/legion/exl3_server.py`) so the comparison is
apples-to-apples rather than "speed from one tool, quality from another".

| engine | quant | size | decode | pass@1 | **pass@2** |
|---|---|---:|---:|---:|---:|
| llama.cpp | GGUF UD-Q3_K_XL | 12.24 GiB | 39.8 t/s | 17.6% | **38.2%** (13/34) |
| ExLlamaV3 | EXL3 3.0bpw | 12.53 GiB | 44.1 t/s | 14.7% | **38.2%** (13/34) |
| **llama.cpp + MTP** | GGUF UD-Q3_K_XL | +1.28 GiB | **75.3 t/s** | — | — |

**Identical pass@2 (13/34 both).** At matched size and settings the two quant formats and
engines deliver equivalent quality on this benchmark, so the decision is purely throughput and
operational fit:

- ExLlamaV3 is ~11% faster than llama.cpp's *baseline*…
- …but **llama.cpp + MTP is ~70% faster than ExLlamaV3**, because there is no EXL3 MTP draft head.
- ExLlamaV3 is Windows-only here (open sm_120 WSL2 bug), needs `setuptools` + `triton-windows`
  to import at all, and has the same `max_batch_size` VRAM trap.

**Verdict: llama.cpp + MTP.** ExLlamaV3 is a credible engine on this hardware and worth
revisiting if an EXL3 MTP/EAGLE draft appears, but today MTP is decisive.


---

## 14. Official Aider polyglot benchmark (diff edit format)

30 exercises across Python/Go/Rust, `--edit-format diff`, `--tries 2`, thinking at
`reasoning_effort=medium`. This is the real leaderboard harness, not our whole-file approximation.

| metric | result |
|---|---:|
| pass_rate_1 | 46.7% (14/30) |
| **pass_rate_2** | **63.3%** (19/30) |
| **percent_cases_well_formed** | **100.0%** |
| malformed responses | 0 |
| syntax errors / lazy comments | 0 / 0 |
| exhausted context windows / test timeouts | 0 / 0 |
| seconds per case | 124 |

**The 100% well-formed rate is the notable result.** In diff format the model must emit
SEARCH/REPLACE blocks that apply cleanly against the existing file; a model that cannot do this
reliably is unusable in a real coding agent no matter how good its raw code is. Zero malformed
responses across 30 exercises and 3 languages says the edit-format plumbing (llama.cpp `--jinja`
plus this GGUF's template) is sound. Our whole-file harness cannot see this axis at all.

Note it scores **higher** than our `aider_lite` pass@2 (63.3% vs 58.8%) despite the harder edit
format — the polyglot set and language mix differ, so the two are not directly comparable.

### ⚠️ `benchmark.py` shuffles the exercise set unseeded
`random.shuffle(test_dnames)` runs with no seed before `--num-tests N` truncates, so **every
invocation tests a different random N**. Our first quant A/B overlapped on only 8 of 30
exercises and was discarded. Pin the set with `--keywords` (see `KEYWORDS` in
`bench/legion/run_aider_polyglot.sh` and `bench/legion/poly30_exercise_set.txt`) before comparing
any two runs.


---

## 15. Polyglot in diff format — Python / JavaScript / Java

30 exercises, `--edit-format diff`, `--tries 2`, thinking at `reasoning_effort=medium`.

> **Kotlin and TypeScript are not available.** The aider polyglot set ships exactly six
> languages — cpp, go, java, javascript, python, rust — with no Kotlin and no TypeScript, and
> `kotlinc` is not installed on this box. **Java is substituted as the closest JVM/Gradle
> analogue**; JavaScript covers the JS-family request.

| language | n | pass@1 | **pass@2** | malformed |
|---|---:|---:|---:|---:|
| python | 8 | 12.5% | **75.0%** | 0 |
| java | 9 | 22.2% | **66.7%** | 0 |
| javascript | 13 | 38.5% | **61.5%** | 1 |
| **total** | **30** | **26.7%** (8/30) | **66.7%** (20/30) | 1 |

Aggregate: `percent_cases_well_formed` **96.7%**, 0 syntax errors, 0 lazy comments,
0 exhausted context windows, 0 test timeouts, 148.5 s/case.

### Compared with the Python/Go/Rust run (§14)

| set | pass@1 | pass@2 | well-formed |
|---|---:|---:|---:|
| python/go/rust | 46.7% | 63.3% | 100.0% |
| **python/javascript/java** | 26.7% | **66.7%** | 96.7% |

Similar pass@2, but a very different route to it: the py/js/java mix starts far weaker
(26.7% vs 46.7% first-attempt) and the retry recovers **+40 points** versus +16.6 for
py/go/rust. In other words the model's *first* draft in JS/Java is often wrong, but it
repairs it reliably once it sees the test output — further evidence that pass@1 badly
misrepresents this model.

Recovered on the second attempt: python +5 (grep, hangman, scale-generator, wordy, zipper),
java +4 (protein-translation, react, twelve-days, zipper), javascript +3
(parallel-letter-frequency, promises, queen-attack).

**Caveats.** Per-language n is 8–13, so the per-language rates carry wide error bars and the
ordering between them is not meaningful — only the aggregate is. The language split is uneven
because `benchmark.py` shuffles unseeded (§14). The single malformed response was in JavaScript.

Running JS and Java outside Docker required de-containerising two hardcoded paths — see
`bench/legion/setup_aider_multilang.sh`.
