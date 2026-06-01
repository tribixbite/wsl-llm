# EAGLE-3 + DFlash Survey (post-WSL-reboot resume, 2026-05-31)

Re-investigation after ~4-week gap. The world moved a lot.

## What appeared since May 3

### EAGLE-3 drafters for Qwen3.6-27B (none existed May 3)

1. **`Ex0bit/Qwen3.6-27B-PRISM-EAGLE3`** (May 19) — primary find
   - Two variants:
     - `compressed/` (1.1 GB) — for SGLang, recommended
     - `full/` (3.1 GB) — for vLLM, more compatible
   - Trained on PRISM-PRO base, validated on stock Qwen3.6-27B BF16
   - Claimed: SGLang 93→183 t/s (1.97×), vLLM 90→130 t/s (1.44×)
   - Accept-length τ=2.2-2.4 on stock Qwen3.6-27B

2. **`Dogacel/specdrift-qwen3.6-27b-eagle3`** (May 10) — alt, 625M params, no card

### DFlash drafters

3. **`turboderp/Qwen3.6-27B-DFlash-exl3`** (May 6) — EXL3 quants of z-lab/Qwen3.6-27B-DFlash
   - Branches: 2.50, 3.00, 3.50, 4.00, 5.00, 6.00 bpw
   - **Mean accepted tokens 4.46 at 4.00 bpw** (vs MTP ~2.4)
   - Community bench: 140-177 t/s on agentic code
   - **Caveat**: same arch as Qwen3.6-27B → ~14 GB drafter at 4 bpw. Plus 16 GB target = 30 GB → exceeds single 3090. Multi-GPU needed?

### AWQ + MTP bundled

4. **`shawnw3i/Qwen3.6-27B-AWQ-MTP`** (May 28) — 19.6 GB AWQ + MTP head in one repo
   - Uncensored variant — quality risk for non-RP work
   - Claims 110 t/s on A800
   - Drop-in for vLLM 0.21.0 `--quantization awq_marlin`

### Engine releases

- **vLLM 0.20.0** (Apr 27), 0.21.0 (May 15), **0.22.0** (May 29) — major
  - 0.22.0 has **EAGLE 3.1**: claims 2× acceptance length on long contexts
- **SGLang 0.5.12** (May) — day-0 Qwen3.6 cookbook + EAGLE-3 SWA
- **ExLlamaV3 0.0.37** (May 24) — minor fixes, same arch

### Sandermage Genesis v7.72.x (May 5+)

- Added **PN59 Streaming-GDN** — 95% memory drift reduction, 256K context on 24 GB
- PN60-65 Blackwell patches (not relevant)
- No EAGLE-3 specific patches yet

## What we attempted today (2026-05-31)

### ✅ ExLlamaV3 install + baseline
- Solved the torch/flash-attn/xformers/CUDA matrix (see `bench/results/qwen36-27b/exllamav3/README.md`)
- **Baseline: 26.4 prose / 26.7 code t/s** — confirms bandwidth ceiling
- DFlash next (blocked on potential VRAM constraint, needs investigation)

### ❌ SGLang 0.5.12 + EAGLE-3 + Lorbus AutoRound
- Hit a sequence of issues:
  - huggingface_hub 0.36 dropped `is_offline_mode` → patched with shim
  - sgl_kernel needed `libnvrtc.so.13` → added to LD_LIBRARY_PATH
  - `--mamba-scheduler-strategy extra_buffer` + `SGLANG_ENABLE_SPEC_V2=1` required
  - **Hard blocker**: `gptq_marlin_repack` fails with `size_n=96 is not divisible by tile_n_size=64`
- This is the **same Marlin bug** that affects all AWQ/GPTQ Qwen3.6-27B paths in SGLang
- Path closed unless we get a non-Marlin Qwen3.6-27B target

### ❌ vLLM 0.17 + EAGLE-3 + Lorbus AutoRound
- Genesis plugin still loads (19/39 applied, same as before)
- Model loads
- **Crashes at first batch**: `hidden_states, _ = outputs` → `ValueError: too many values to unpack (expected 2)`
- Looks like vLLM 0.17's EAGLE3 path was written before Qwen3.6 hidden-state shape changes
- vLLM 0.22.0 likely fixes this (EAGLE 3.1) — untested due to time

### Pending paths (commit-worthy work for next session)

| Path | Risk | Expected outcome |
|------|------|-----------------|
| **vLLM 0.22.0 + Genesis (or no-Genesis) + EAGLE-3 + Lorbus** | Genesis plugin may not work on 0.22 — separate venv test | 100-130 t/s if works |
| **vLLM 0.17 + AWQ-MTP (shawnw3i)** | Uncensored variant — quality drop possible | ~70-90 t/s; small win |
| **ExL3 + DFlash dual-GPU (TP=2)** | User reserves GPU 1 — can't do today | If permitted: 140-177 t/s |
| **ExL3 + DFlash single-GPU with tighter drafter quant** | 2.50 bpw drafter = ~10 GB? + 16 GB target = fits | Lower acceptance but might still hit 80+ |
| **Sandermage Genesis v7.72.x** + Streaming-GDN | Lower priority — solves ctx not speed | n/a for our 32k workload |

## UPDATE 2026-06-01 — DFlash (Luce fork) tested; plan re-prioritized

A research pass corrected the April assumptions:

- **The April DFlash "dead end" was a wrong-fork artifact** — it used
  `spiritbuun/buun-llama-cpp` (broken hybrid tree kernels). The purpose-built
  **`Luce-Org/lucebox-hub`** fork has custom GatedDeltaNet tree CUDA kernels and
  documents 3.43× / 129 t/s on a single 3090.
- **Built + benched it.** See `bench/results/qwen36-27b/dflash-luce/README.md`.
  DFlash accelerates **only under greedy (temp=0)**: code 58 t/s greedy, but
  **accept=0 / 19 t/s at temp=0.6** (DDTree verify is argmax-only; no Leviathan
  tree rejection sampling yet — their README L459/469). Greedy 58 t/s code is
  still **below** vLLM+Genesis+MTP production (67 t/s code at temp=0.6).
  **Not a production upgrade until tree rejection sampling lands.**
- **PRISM-EAGLE3 / vLLM 0.22 EAGLE3 are likely lateral, not breakthroughs:**
  PRISM's accept length τ≈2.2–2.4 ≈ the MTP n=3 already in production; its
  headline numbers are BF16-target (won't fit 24 GB) on Blackwell. vLLM 0.22
  EAGLE3-on-hybrid is unverified (all benchmarks GB200), and SGLang INT4 still
  hits the Marlin `size_n` + SSM-dtype double-bug.

### Standing conclusion

The realistic single-3090 ceiling **at temp=0.6** remains the **vLLM+Genesis+MTP
~67 t/s code** production stack. The genuine unlock is still a higher
accept-length drafter that works under *sampling*. Watch (in priority order):
1. **Luce DFlash + Leviathan tree rejection sampling** (their roadmap) → makes the
   greedy speed available at temp=0.6.
2. **vLLM EAGLE3 confirmed on Qwen3.6 hybrid + quantized target** (needs a
   non-Blackwell repro) — only then is a PRISM-full fresh-venv test worth it.

### Lower-priority next steps

1. Greedy deterministic codegen only: sweep Luce KV type (f16 vs TQ3) +
   `chain_seed` to lift base decode 19→~35 and greedy code 58→~90.
2. vLLM 0.22 + EAGLE3 + PRISM-full fresh venv — only after #2 above is confirmed.

(Superseded April recommendations: ExL3 + DFlash 2.50bpw — no ExL3 DFlash exists;
ExL3 + DFlash TP=2 — needs GPU 1, and DFlash is greedy-only anyway.)

## Files

- `bench/results/qwen36-27b/exllamav3/README.md` — install recipe + 26.4 baseline
- `bench/results/qwen36-27b/exllamav3/bench.py` — reproducer
- `bench/results/qwen36-27b/exllamav3/requirements.txt` — pinned deps
- `bench/results/qwen36-27b/livecodebench/lcb_full_1051_summary.md` — May 3 LCB result reconstructed
