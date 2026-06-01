# Qwen3.6-27B DFlash via the Luce fork (lucebox-hub) — 2026-06-01

The April "push-to-80" investigation tested DFlash via **`spiritbuun/buun-llama-cpp`**
and got 18/45 t/s — concluding DFlash underperformed. That was the **wrong fork**:
the spiritbuun build doesn't have working tree kernels for the Qwen3.6 hybrid
GatedDeltaNet, so the recurrent-state handling collapses acceptance.

This run uses the purpose-built **`Luce-Org/lucebox-hub`** fork (`luce-dflash`),
which ships custom CUDA kernels (`ggml_gated_delta_net_tree`,
`ggml_ssm_conv_tree`) specifically for the hybrid target on Ampere. Its own
RESULTS.md documents **3.43× / 129 t/s on a single RTX 3090**.

**Bottom line: DFlash (Luce) is real and fast — but only under greedy decoding
(temp=0). At the project's standard temp=0.6 sampling it gives ZERO speedup, and
even its best greedy number (58 t/s code) is below the existing vLLM+Genesis+MTP
production stack (67 t/s code at temp=0.6). Not a production upgrade today.**

## Setup

| Component | Value |
|---|---|
| Engine | `Luce-Org/lucebox-hub` → `dflash_server`, built sm_86 + BSA, CUDA 12.6 |
| Target | `unsloth/Qwen3.6-27B-GGUF` **Q4_K_M** (16.8 GB), arch `qwen35` |
| Draft | `Lucebox/Qwen3.6-27B-DFlash-GGUF` `dflash-draft-3.6-q4_k_m.gguf` (1.06 GB, arch `qwen35-dflash-draft`) |
| Spec | `--ddtree --ddtree-budget 22` (best-first tree verify) |
| KV | TQ3_0 (`DFLASH27B_KV_TQ3=1`), `--fa-window 2048` |
| Ctx | 8192 (see VRAM note) · GPU 0, 24 GB · **18.1 GB used** |
| Launcher | `scripts/serve-27b-dflash-luce.sh` |

## Results

### temp=0.6 (project standard sampling) — DDTree is INERT

| Task | accept_rate | decode t/s |
|------|---:|---:|
| prose | 0.000 | 18.8 |
| code | 0.000 | 18.8 |
| math | 0.000 | 18.7 |

Acceptance is **exactly zero** → the ~19 t/s is pure autoregressive decode with
the draft/verify overhead wasted.

### greedy (temp=0) — DDTree works

| Task | accept_rate | decode t/s |
|------|---:|---:|
| **code** | 0.554 | **58.2** |
| math | 0.554 | 55.4 |
| json | 0.427 | 44.2 |
| prose | 0.166 | 21.7 |
| **avg** | 0.425 | **44.9** |

Speedup is real (2.8–3.1× on code/math over the ~19 t/s base) but off a low base
and **greedy-only**. Prose barely accelerates (natural language is less
predictable → low acceptance).

## Root cause of the greedy-only behavior (from the fork's own README, L459/469)

> *"The DDTree verify skeleton stays argmax (preserves accept rate); only the
> committed token at each verify step is drawn from a small CPU sampler chain…
> `temperature=0` keeps the path bit-exact greedy. **Full Leviathan-style
> rejection sampling on the tree is still a future addition.**"*

The draft tree is built and verified against **argmax**. With temp>0 the committed
token is drawn from a sampler that diverges from the argmax-verified tree, so
every draft is rejected → accept ≈ 0. Proper speculative decoding stays fast at
temp>0 via Leviathan/EAGLE rejection sampling; **the Luce fork hasn't implemented
that on the DDTree yet** (explicitly on their roadmap). This is the single
dispositive limitation.

## Why our absolute numbers trail their 129 t/s headline

- **Base decode 19 t/s vs their 35 t/s AR.** Our greedy code (58) ÷ our base
  (19) = 3.1× ≈ their 3.43×, so the *speedup multiple* reproduces — the gap is
  the **base decode being ~half theirs**. Suspects (not chased — see below):
  TQ3_0 KV FWHT-rotation overhead, `dflash_server` path vs their `test_generate`
  harness, GPU 0's degraded PCIe 4x, and `chain_seed` (README L423 says it lifts
  AL from ~4 to ~9; our accept 0.55 ≈ AL ~4–5 suggests it may be off in this build).
- **accept 0.55 vs their 0.65** → lower AL, compounding the base-speed loss.

These weren't pursued because the **greedy-only limitation already disqualifies
DFlash for temp=0.6 production**, and the box had two CPU-thermal reboots this
session (heavy builds) — not worth a KV/chain_seed sweep for a path that can't
beat the baseline at the required sampling temp.

## Comparison to the production stack

| Stack | code t/s | prose t/s | sampling | notes |
|---|---:|---:|---|---|
| **vLLM 0.17 + Genesis + MTP n=3** (production) | **67** | 46 | temp=0.6 ✅ | current daily 27B |
| SGLang NEXTN n5_k1_d6 (code-tuned) | 64 | 42 | temp=0.6 ✅ | from push-to-80 |
| DFlash Luce, greedy | 58 | 22 | temp=0 ⚠️ | this run |
| DFlash Luce, temp=0.6 | 19 | 19 | temp=0.6 | accept=0, inert |
| DFlash spiritbuun (Apr) | 45 | 18 | temp=0.6 | wrong fork |

## Verdict & when to revisit

- **Not a production upgrade.** Slower than the existing MTP stack *and*
  greedy-only.
- **Revisit when** the Luce fork lands Leviathan-style tree rejection sampling
  (then temp=0.6 acceptance becomes possible) — watch their roadmap / releases.
- **Possible niche now:** deterministic temp=0 code generation, where 58 t/s
  beats the AR base — but the project standard is temp=0.6, and Qwen advises
  against greedy, so this is marginal. If a base-speed fix (f16 KV / chain_seed)
  lifts greedy code toward ~90 t/s, deterministic codegen could become attractive.

## Reproduce

```bash
# build (use -j6 to avoid CPU thermal trips — see CLAUDE.md)
cd ~/lucebox-hub/server
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DDFLASH27B_ENABLE_BSA=ON
cmake --build build --target dflash_server -j6

# serve (GPU 0; 18 GB at ctx 8192)
CUDA_VISIBLE_DEVICES=0 TARGET=~/models/q4km-target/Qwen3.6-27B-Q4_K_M.gguf \
  ~/git/wsl-llm/scripts/serve-27b-dflash-luce.sh
```

## Operational notes (this session)

- **ctx matters a lot for VRAM**: ctx=32768 maxed the 3090 to 24080/24575 MiB and
  decode collapsed to **1.8 t/s** (DDTree scratch spilling). ctx=8192 → 18.1 GB,
  stable. The 1.8 t/s number is a VRAM-thrash artifact, not representative.
- **Always SIGTERM `dflash_server`, never `kill -9`** — a SIGKILL left **21 GB
  leaked on the GPU** (WSL2 doesn't reclaim a hard-killed CUDA context). Graceful
  SIGTERM releases VRAM cleanly.
