# Qwen 3.6 27B "Push to 80 t/s" Investigation (2026-04-25/26)

User goal: get Qwen3.6-27B Dense to **80 t/s** on a single RTX 3090.

**Result**: **64.15 t/s code (champion), 51 prose**. We did not reach 80 t/s.
The realistic ceiling on this hardware with **publicly-available artifacts** is ~64 t/s code, ~51 t/s prose.

See [docs/QWEN36_27B_BENCHMARKS.md §10](../../../../docs/QWEN36_27B_BENCHMARKS.md#10-push-to-80-ts) for the analysis.

## Why we couldn't hit 80

Three-way bottleneck:
1. **27B Dense weight bandwidth**: 14-17 GB/token at INT4, RTX 3090 ≈ 936 GB/s → ~60 t/s theoretical no-spec ceiling. Observed: 25-32 t/s = 41-54% utilization.
2. **MTP acceptance rate decay**: pos1 ~97%, pos2 ~95%, pos3 ~91%, pos4 ~21% (per HPC-AI bench). n=3 is the optimum; deeper chains waste forward passes.
3. **No EAGLE-3 head exists for Qwen3.6-27B yet** — only Qwen3-{1.7B/4B/8B/14B/32B}, Qwen3.5-{9B/35B-A3B}, Qwen3-Coder. EAGLE-3 typically gives 3-4× over MTP's 2×, but you'd need to train one (~6h via [SpecForge](https://github.com/sgl-project/SpecForge) on a 3090, LoRA mode).

## What we tested

### Phase A — SGLang NEXTN tree-spec sweep (`run_tree_sweep.sh`)

Result file: `sglang_tree_sweep.tsv`. 7 configs varying `--speculative-num-steps` (chain depth), `--speculative-eagle-topk` (tree breadth), `--speculative-num-draft-tokens` (verified count).

| Config | n_steps | topk | draft | Prose t/s | Code t/s |
|--------|--------:|-----:|------:|----------:|---------:|
| n3_k1_d4 (baseline) | 3 | 1 | 4 | 45.51 | 56.42 |
| **n3_k2_d8** ⭐ | 3 | 2 | 8 | **50.57** | 55.56 |
| n3_k4_d12 | 3 | 4 | 12 | OOM | - |
| n4_k1_d5 | 4 | 1 | 5 | 46.89 | 52.6 |
| **n5_k1_d6** ⭐ | 5 | 1 | 6 | 42.51 | **64.15** |
| n4_k2_d10 | 4 | 2 | 10 | 47.31 | 52.32 |
| n2_k1_d3 | 2 | 1 | 3 | 43.79 | 46.62 |

Findings:
- **Tree breadth (topk=2) helps prose** (+11% over chain).
- **Deep chain (n=5) helps code** (+14% over n=3) — counter to the n>=4 wasteful claim, because high-acceptance code patterns extend usable depth.
- Bigger trees (topk≥4) OOM at 0.86 mem-utilization.

### Phase B — Threshold loosening + deeper chain (`run_threshold_sweep.sh`)

Result file: `sglang_threshold_sweep.tsv`. Loosened `--speculative-accept-threshold-{single,acc}` and tested `n5/n6` with thresholds.

| Config | thresholds | Prose t/s | Code t/s |
|--------|------------|----------:|---------:|
| n3_k2_d8 | 0.7 / 0.9 | 51.05 | 56.30 |
| n5_k1_d6 | 0.7 / 0.9 | 46.06 | 51.09 |
| n5_k1_d6 | 0.5 / 0.9 | 43.74 | 58.65 |
| n6_k1_d7 | 1.0 / 1.0 | 37.16 | 57.51 |
| n6_k1_d7 | 0.7 / 0.9 | 41.82 | 48.08 |

**Threshold loosening did NOT help** — counter-intuitive. SGLang's accept thresholds appear to behave differently than EAGLE/MTP literature suggests; deeper investigation needed to know what the right values are. Default 1.0/1.0 stays best.

### Phase C — DFlash via spiritbuun fork (`run_dflash.sh`)

Result file: `dflash.tsv`. Built [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) from source (CUDA, sm_86, ~10 min build), used [spiritbuun/Qwen3.6-27B-DFlash-GGUF](https://huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF) (1.85 GB Q8 drafter) with target = `Qwen3.6-27B-IQ4_XS.gguf`.

| Config | Prose t/s | Code t/s |
|--------|----------:|---------:|
| dflash_default (--spec-type dflash) | 18.18 | 44.82 |
| dflash_topk4 / dflash_topk8 | CRASH | CRASH |
| dflash_auto | 18.87 | 44.97 |

DFlash on this combination underperformed both SGLang NEXTN AND vLLM MTP. The 97 t/s claim from the agent's research is likely target-quant-specific or workload-specific. Did not reproduce on Qwen3.6-27B-IQ4_XS.

### Phase D — vLLM compilation tweaks

Tested `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8]}'` per agent recommendation. Result: **vLLM rejected FULL_DECODE_ONLY** — `CUDAGraphMode.FULL_DECODE_ONLY is not supported with spec-decode for attention backend FlashInferBackend; setting cudagraph_mode=PIECEWISE`. Throughput dropped to 23/33 vs 30/55 baseline (likely due to capture_sizes change interfering).

### Phase E — vLLM MTP n=1 test

Per agent claim that n>1 hurts acceptance. Result: 26.84 prose / 45.22 code — **worse than n=3** (30/55). n=3 is genuinely the right MTP setting for the Lorbus checkpoint.

## Production winner unchanged

**SGLang 0.5.9 + NEXTN n=3 + topk=2 + draft=8 + Lorbus AutoRound INT4** = 50.57 prose / 55.56 code, 23.1 GB VRAM.

For workloads dominated by **code generation only**, switch to `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` for **64.15 t/s code** (at the cost of dropping prose to 42.5 t/s).

## What would actually unlock 80 t/s

1. **Train a custom Qwen3.6-27B EAGLE-3 head** via [SpecForge](https://github.com/sgl-project/SpecForge) (LoRA mode, ~6 hours overnight on RTX 3090). EAGLE-3 typically delivers 3-4× over base; vs current ~2× MTP, that's the missing factor. Would need a small calibration corpus.
2. **Wait for someone to publish a Qwen3.6-27B EAGLE-3 head**. Activity around `zenith1232/qwen36-eagle3-drafter-v5` (currently only the 35B-A3B target) suggests a 27B head may be coming.
3. **vLLM + Sandermage Genesis patches** with `turboquant_3bit_nc` KV cache quant. Medium post claims 85 t/s @ 125k ctx. Untested by us — would need to apply unmerged patches to vLLM 0.17.
4. **Wait for ExLlamaV3 + Qwen3.6 + MTP support** — turboderp's hand-tuned Ampere kernels could match or beat vLLM, but no MTP integration yet.

## Files

| File | Description |
|------|-------------|
| `sglang_tree_sweep.tsv` | 7 NEXTN configs, prose+code |
| `sglang_threshold_sweep.tsv` | Threshold + deeper chain configs |
| `dflash.tsv` | DFlash via spiritbuun fork results |
| `run_tree_sweep.sh` | Reproducer for Phase A |
| `run_threshold_sweep.sh` | Reproducer for Phase B |
| `run_dflash.sh` | Reproducer for Phase C |
