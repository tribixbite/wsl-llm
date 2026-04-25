# Qwen 3.6 27B Game of Life HTMLs (10 backend × config variants)

All 10 outputs pass `bun run /tmp/test_rle.ts` validation (3/3 — handles the prompt example `3o2b1o!2b3o`, multi-row `o!o!o`, and high-count `24bo`).

| Variant | Backend / Config | t/s | VRAM | File |
|---------|------------------|----:|-----:|------|
| madreag_q4kxl_turbo3_32k  | Madreag turboquant + UD-Q4_K_XL + turbo3 KV @ 32k    | 19.97 | 17.5 GB | [madreag_q4kxl_turbo3_32k.html](madreag_q4kxl_turbo3_32k.html) |
| madreag_q4kxl_turbo3_64k  | Madreag turboquant + UD-Q4_K_XL + turbo3 KV @ 64k    | 20.21 | 17.9 GB | [madreag_q4kxl_turbo3_64k.html](madreag_q4kxl_turbo3_64k.html) |
| madreag_q4kxl_q8_32k      | Madreag turboquant + UD-Q4_K_XL + q8_0 KV @ 32k      | 20.08 | 18.1 GB | [madreag_q4kxl_q8_32k.html](madreag_q4kxl_q8_32k.html) |
| madreag_q4kxl_bf16_32k    | Madreag turboquant + UD-Q4_K_XL + bf16 KV @ 32k      | 19.80 | 19.1 GB | [madreag_q4kxl_bf16_32k.html](madreag_q4kxl_bf16_32k.html) |
| madreag_iq4xs_turbo3_32k  | Madreag turboquant + IQ4_XS + turbo3 KV @ 32k        | 24.18 | 15.4 GB | [madreag_iq4xs_turbo3_32k.html](madreag_iq4xs_turbo3_32k.html) |
| madreag_iq4xs_turbo3_262k | Madreag turboquant + IQ4_XS + turbo3 KV @ 262k       | 23.96 | 18.5 GB | [madreag_iq4xs_turbo3_262k.html](madreag_iq4xs_turbo3_262k.html) |
| vllm_lorbus_no_mtp_fp8_64k | vLLM + Lorbus AutoRound INT4 + fp8 KV (no MTP) @ 64k | 24.90 | 21.7 GB | [vllm_lorbus_no_mtp_fp8_64k.html](vllm_lorbus_no_mtp_fp8_64k.html) |
| vllm_lorbus_mtp3_fp8_64k  | vLLM + Lorbus AutoRound INT4 + fp8 KV + MTP n=3 @ 64k | **53.74** | 21.3 GB | [vllm_lorbus_mtp3_fp8_64k.html](vllm_lorbus_mtp3_fp8_64k.html) |
| vllm_lorbus_mtp3_fp8_125k | vLLM + Lorbus + fp8 + MTP n=3 @ 125k                 | **53.70** | 21.3 GB | [vllm_lorbus_mtp3_fp8_125k.html](vllm_lorbus_mtp3_fp8_125k.html) |
| **vllm_lorbus_mtp3_fp8_262k** ⭐ | vLLM + Lorbus + fp8 + MTP n=3 @ **262k**       | **54.55** | 21.3 GB | [vllm_lorbus_mtp3_fp8_262k.html](vllm_lorbus_mtp3_fp8_262k.html) |

## Key findings

- **Speed champion**: vLLM + Lorbus AutoRound INT4 + MTP n=3 + fp8 KV at 262k context = **~54 t/s** (2.7× faster than Madreag UD-Q4_K_XL)
- **Without MTP**: vLLM ≈ Madreag IQ4_XS at ~24 t/s — speedup is entirely from speculative decoding
- **Context size doesn't affect throughput** for either backend — 32k vs 262k is ±5%
- **KV format barely matters** for speed on 27B Dense (turbo3 vs q8 vs bf16 within 2%) — model-weight bandwidth dominates
- **Quant size matters most** in llama.cpp world: IQ4_XS (15 GB) is +20% faster than UD-Q4_K_XL (17 GB)
- **MTP n=4 fails** with CUDA illegal memory access — the MTP head is trained for n=3
- **27B is fundamentally slower than 35B-A3B MoE**: 27B all-active params (~13.5 GB to read/token at Q4) vs 3B-active for 35B-A3B → bandwidth-bound at lower throughput

See [docs/QWEN36_27B_BENCHMARKS.md](../../../../docs/QWEN36_27B_BENCHMARKS.md) for the full benchmark report.
