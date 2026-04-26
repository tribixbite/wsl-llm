# Qwen 3.6 27B Speed Ceiling Investigation (2026-04-25)

Re-bench to determine if KV format, smaller weight quants, or alternative MTP step counts can break the existing 54 t/s ceiling. **Conclusion: no — they cannot.**

See [docs/QWEN36_27B_BENCHMARKS.md §8](../../../../docs/QWEN36_27B_BENCHMARKS.md#8-speed-ceiling-investigation--what-actually-moves-27b-dense-ts) for the full analysis.

## Files

| File | Description |
|------|-------------|
| `kv_sweep.sh` | Madreag fork sweep across all 11 supported KV types (turbo3, turbo2, turbo4, turbo3_tcq, turbo2_tcq, turbo1.5, q8_0, bf16, q4_0, iq4_nl, q5_0) on IQ4_XS @ 64k |
| `kv_sweep.tsv` | Results: gen t/s, prompt t/s, VRAM, tokens per KV type |
| `quant_sweep.sh` | Madreag fork on three weight quants (IQ4_XS, UD-Q4_K_XL, UD-Q3_K_XL) with turbo3 KV @ 64k |
| `quant_sweep.tsv` | Results: gen t/s per quant |

## Headline numbers

**KV format sweep** (Madreag IQ4_XS @ 64k, prose prompt): 11 types, range **24.0 → 27.3 t/s** = 13% spread.
- Winner: turbo3 (27.28 t/s, 15.4 GB VRAM)
- Floor: turbo2_tcq (24.0 t/s)
- Conclusion: **KV format does not unlock 27B Dense throughput.**

**Weight quant sweep** (Madreag turbo3 KV @ 64k):
- IQ4_XS (15.4 GB file) → 25.24 t/s ⭐ (Madreag tuned IQ kernel)
- UD-Q4_K_XL (17.6 GB) → 20.40 t/s
- UD-Q3_K_XL (14.5 GB) → 20.05 t/s — **smaller, but no faster** (UD mixed-tier blocks ≠ raw bandwidth)

**vLLM MTP step sweep** (Lorbus AutoRound + fp8 KV @ 64k, same prose prompt):
- MTP n=2: 29.37 t/s
- MTP n=3: 31.06 t/s ⭐
- MTP n=4: CUDA illegal memory access (head trained for n=3)

## Why 27B Dense is bandwidth-bound

RTX 3090 ≈ 936 GB/s memory bandwidth. 27B at Q4 ≈ 14-17 GB/token to read.
Theoretical max ≈ 60 t/s. Observed without MTP ≈ 25 t/s = 41% utilization.
MTP n=3 doubles throughput on coding (54 t/s) by amortizing weight reads over multiple verified tokens.

## What's NOT in this sweep

Untried because either expensive (full engine port) or already known-poor:
- **ExLlamaV3** (turboderp) — could give +30-60% on baseline; would lose MTP multiplier
- **TensorRT-LLM** — days of setup, unproven gain
- **z-lab/Qwen3.6-27B-DFlash** — block-diffusion drafting, claims 4× over MTP for Qwen 3.5
- **Sandermage Genesis** vLLM patches (TurboQuant 3-bit KV in vLLM) — KV format already proved irrelevant
