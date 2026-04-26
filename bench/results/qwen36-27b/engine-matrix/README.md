# Qwen 3.6 27B Engine Matrix Bench (2026-04-25)

Engine comparison across **3 llama.cpp forks**, **vLLM nightly** (with MTP), and **SGLang** (with NEXTN/MTP). Same model (Lorbus AutoRound INT4 for vLLM/SGLang, IQ4_XS GGUF for llama.cpp engines), same prompts (800-token LSM-tree prose + 800-token TS BST code), same context (64k), single RTX 3090.

See [docs/QWEN36_27B_BENCHMARKS.md §9](../../../../docs/QWEN36_27B_BENCHMARKS.md#9-engine-matrix-multi-backend-deep-dive) for the full analysis.

## Headline results

| Engine | Quant | Speculative | Prose t/s | Code t/s |
|--------|-------|-------------|----------:|---------:|
| Madreag turboquant | IQ4_XS, turbo3 KV | none | 26.47 | ~25 |
| Madreag turboquant | IQ4_XS, q8_0 KV | none | 26.01 | ~25 |
| ik_llama.cpp | IQ4_XS, q8_0 KV | none | 25.55 | ~25 |
| ik_llama.cpp | IQ4_XS, q4_0 KV | none | 25.15 | ~25 |
| Upstream llama.cpp | IQ4_XS, q8_0 KV | none | 25.11 | ~25 |
| Upstream llama.cpp | IQ4_XS, f16 KV | none | 24.82 | ~25 |
| vLLM 0.17 nightly | Lorbus AutoRound INT4, fp8 KV | none | 22.04 | 31.55 |
| vLLM 0.17 nightly | Lorbus AutoRound INT4, fp8 KV | --enforce-eager (no MTP) | 7.09 | 12.12 |
| **vLLM 0.17 nightly** | **Lorbus AutoRound INT4, fp8 KV** | **MTP n=3** | **30.09** | **54.87** |
| **SGLang 0.5.9** | **Lorbus AutoRound INT4, fp8_e4m3 KV** | **none** | **31.78** | 32.39 |
| **SGLang 0.5.9** ⭐ | **Lorbus AutoRound INT4, fp8_e4m3 KV** | **NEXTN n=3** | **43.22** ⭐ | **54.16** |

## Key conclusions

1. **SGLang wins** — both as a faster baseline (32 t/s prose vs vLLM 22) and with NEXTN+MTP speculative (43 t/s prose vs vLLM 30, +44%).
2. **Code throughput is engine-agnostic with MTP** — vLLM 54.87 vs SGLang 54.16, statistical tie.
3. **llama.cpp family is bandwidth-locked at ~25 t/s** — Madreag, ik_llama.cpp, upstream all converge within 7% range, regardless of KV format.
4. **`--enforce-eager` is a 70% throughput loss in vLLM** — never use it for inference; CUDA graphs are essential.

## What broke

- **TRITON_ATTN backend in vLLM** — OOM at 0.92 mem-utilization (uses more cache memory than FLASHINFER); skipped.
- **SGLang NGRAM speculative** — `AttributeError: 'NgramVerifyInput' object has no attribute 'topk'` in hybrid_linear_attn_backend.py:511. Bug, not config error. Filed (or to be filed) upstream.
- **SGLang with --dtype float16** — runtime crash in causal_conv1d (DeltaNet kernel) due to dtype mismatch. **Use `--dtype bfloat16`**.
- **ExLlamaV3** — install fights with PyTorch 2.11/CUDA 13.0 wheel matrix (flash-attn build needs custom flags); deferred. exl3 quants exist on HF (UnstableLlama/Qwen3.6-27B-exl3-4.15bpw).

## SGLang setup notes

```bash
# venv (uv works fine):
uv venv ~/sglang_env --python 3.10
uv pip install --python ~/sglang_env/bin/python "sglang[all]"

# launch (NEXTN n=3 = best):
SGLANG_DISABLE_CUDNN_CHECK=1 \
CUDA_VISIBLE_DEVICES=0 \
~/sglang_env/bin/python -m sglang.launch_server \
    --model-path ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --quantization auto-round --dtype bfloat16 \
    --tp 1 --context-length 65536 \
    --mem-fraction-static 0.88 --max-running-requests 1 \
    --kv-cache-dtype fp8_e4m3 \
    --port 8082 --host 0.0.0.0 --trust-remote-code \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

The `SGLANG_DISABLE_CUDNN_CHECK=1` env var is required because SGLang ships with PyTorch 2.9.1 + CuDNN 9.10 (a known buggy combo per pytorch#168167) and refuses to start without explicit override. Effects on this workload are negligible since we don't use Conv3d.

## Files

| File | Description |
|------|-------------|
| `llamacpp_engines.tsv` | Phase 1 results: 3 llama.cpp engines × 6 KV configs |
| `vllm.tsv` | Phase 2 vLLM partial results (default+eager configs; the rest hit JSON quoting/OOM and were re-bench'd manually) |
| `vllm_mtp_clean.tsv` | Clean MTP n=3 result on prose+code prompts |
| `sglang.tsv` | Phase 3 SGLang: default-bf16, NEXTN n=3 (NGRAM crashed) |
| `run_phase1_llamacpp.sh` | Reproducer — 3 llama.cpp engines on IQ4_XS, prose prompt |
| `run_phase2_vllm.sh` | Reproducer — 6 vLLM configs (some require fixed JSON quoting) |
| `run_phase3_sglang.sh` | Reproducer — SGLang launch+bench harness |
