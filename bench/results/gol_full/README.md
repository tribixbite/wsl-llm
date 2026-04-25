# Game of Life — Full HTML Outputs (9 quant/engine variants)

Conway's Game of Life prompt with RLE URL-hash sync, run at `max_tokens=16384` (all natural-stop, no truncation). Each output passes a parser-validation harness (`bun run /tmp/test_rle.ts <html>`) confirming `parseRLE("3o2b1o!2b3o")`, `parseRLE("o!o!o")`, and `parseRLE("24bo")` produce correct grid dimensions.

| Variant | Engine + cache type | Model | t/s on this run | File |
|---------|---------------------|-------|----------------:|------|
| ik_llama_k_xl_64k_q8 | Madreag binary, q8_0 KV @ 64k / 2 slots | UD-Q4_K_XL | 101.7 | [`ik_llama_k_xl_64k_q8.html`](ik_llama_k_xl_64k_q8.html) |
| ik_llama_imatrix_q4_0 | Madreag binary, q8_0 KV @ 64k / 2 slots | bartowski imatrix-Q4_0 | 106.3 | [`ik_llama_imatrix_q4_0.html`](ik_llama_imatrix_q4_0.html) |
| **madreag_turbo3_262k** ⭐ (production) | Madreag binary, turbo3 KV @ 262k / 1 slot | UD-Q4_K_XL | 98.8 | [`madreag_turbo3_262k.html`](madreag_turbo3_262k.html) |
| thetom_turbo3 | TheTom fork, turbo3 KV @ 32k / 1 slot | UD-Q4_K_XL | 97.1 | [`thetom_turbo3.html`](thetom_turbo3.html) |
| spiritbuun_turbo3 | spiritbuun fork, turbo3 KV @ 32k / 1 slot | UD-Q4_K_XL | 80.7 | [`spiritbuun_turbo3.html`](spiritbuun_turbo3.html) |
| spiritbuun_turbo3_tcq | spiritbuun fork, turbo3_tcq (Viterbi) @ 32k / 1 slot | UD-Q4_K_XL | 87.8 | [`spiritbuun_turbo3_tcq.html`](spiritbuun_turbo3_tcq.html) |
| amesianx_tbq3 | AmesianX fork, tbq3 @ 16k / 1 slot | UD-Q4_K_XL | 94.1 | [`amesianx_tbq3.html`](amesianx_tbq3.html) |
| animehacker_tq3_0 | animehacker fork, tq3_0 @ 32k / 1 slot | UD-Q4_K_XL | 17.0 (slow) | [`animehacker_tq3_0.html`](animehacker_tq3_0.html) |
| heretic_turbo3 | Madreag binary, turbo3 KV @ 64k / 1 slot | Youssofal Abliterated-Heretic Q4_K_M | 107.1 | [`heretic_turbo3.html`](heretic_turbo3.html) |

Open each `.html` directly in a browser to compare side-by-side. Each file is self-contained (no external dependencies). Try `?#3o2b1o!2b3o` in the URL hash to load a custom initial pattern.

Notes:
- Several variants required 1-3 retries before producing a working `parseRLE`. Earlier outputs had assorted bugs (regex rejecting valid input, multi-digit count miscounted, infinite loops on leading digits, `charCodeAt(48)` typo, etc). The samples here are the first passing attempt with a strict 3-test validator.
- amesianx was tested at 16k context because its `tbq3` initialization on this hardware was slow at 32k+.
- animehacker is dramatically slower at long generation (~17 t/s vs ~100 t/s for others), reflecting its unoptimized CUDA path. Self-described as "PolarQuant 3-bit, NOT full TurboQuant with QJL".

See [docs/QWEN36_BENCHMARKS.md](../../../docs/QWEN36_BENCHMARKS.md) for the full benchmark report.
