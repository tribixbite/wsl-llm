# Game of Life — Full HTML Outputs (8 quant/engine variants)

Conway's Game of Life prompt with RLE URL-hash sync, run at `max_tokens=16384` (all natural-stop, no truncation) on the same Qwen3.6-35B-A3B-UD-Q4_K_XL model unless noted.

| Variant | t/s | Tokens | File |
|---------|----:|-------:|------|
| ik_llama.cpp + UD-Q4_K_XL @ 64k / q8_0 KV (mid-experiment baseline) | 103.6 | 5449 | [`ik_llama_k_xl_64k_q8.html`](ik_llama_k_xl_64k_q8.html) |
| ik_llama.cpp + bartowski imatrix-Q4_0 @ 64k / q8_0 KV | 115.4 | 5464 | [`ik_llama_imatrix_q4_0.html`](ik_llama_imatrix_q4_0.html) |
| **Madreag turbo3 + UD-Q4_K_XL @ 262k** ⭐ (current production) | **101.6** | 5988 | [`madreag_turbo3_262k.html`](madreag_turbo3_262k.html) |
| TheTom turbo3 + UD-Q4_K_XL @ 32k | 97.1 | 6082 | [`thetom_turbo3.html`](thetom_turbo3.html) |
| spiritbuun turbo3 + UD-Q4_K_XL @ 32k | 96.7 | 5468 | [`spiritbuun_turbo3.html`](spiritbuun_turbo3.html) |
| spiritbuun turbo3_tcq (Viterbi) + UD-Q4_K_XL @ 32k | 78.5 | 14735 | [`spiritbuun_turbo3_tcq.html`](spiritbuun_turbo3_tcq.html) |
| AmesianX tbq3 + UD-Q4_K_XL @ 32k | 96.5 | 4350 | [`amesianx_tbq3.html`](amesianx_tbq3.html) |
| animehacker tq3_0 + UD-Q4_K_XL @ 32k | 17.0 | 6183 | [`animehacker_tq3_0.html`](animehacker_tq3_0.html) |

Open each `.html` directly in a browser to compare side-by-side. Each file is self-contained (no external dependencies).

See [docs/QWEN36_BENCHMARKS.md](../../../docs/QWEN36_BENCHMARKS.md) for the full benchmark report.
