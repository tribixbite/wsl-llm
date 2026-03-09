# Benchmark Results (March 5, 2026)

Hardware: 2x RTX 3090 (24GB each), 64GB DDR4, 12c/24t CPU
llama.cpp: b8204 (CUDA), b8181 (Vulkan Windows)
All values in tokens/second. PP = prompt processing, TG = text generation.

## Qwen3.5-35B-A3B -- Q4_K_XL (20.7 GiB, 34.66B params, 3B active)

Fits fully on 1x RTX 3090 (24GB).

### GPU Benchmarks

| Backend | GPUs | ngl | FA | KV Cache | pp64 | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4096 | tg128 |
|---------|------|----:|---:|----------|-----:|------:|------:|------:|-------:|-------:|-------:|------:|
| **Vulkan (Win)** | **1** | **99** | **on** | **F16** | **1142** | **1586** | **2227** | **2841** | **2830** | **2837** | **2833** | **114.3** |
| Vulkan (Win) | 1(GPU1) | 99 | on | F16 | 1115 | 1584 | 2230 | 2822 | 3358 | 3774 | 3809 | 110.1 |
| CUDA (WSL) | 2 | 99 | on | F16 | 929 | 1270 | 1901 | 2618 | 3171 | 3659 | 3650 | 97.6 |
| CUDA (WSL) | 1 (GPU0) | 99 | on | F16 | 845 | 1205 | 1797 | 2412 | 2639 | 2623 | 2589 | 102.3 |
| CUDA (WSL) | 1 (GPU1) | 99 | on | F16 | 917 | 1263 | 1944 | 2625 | 3222 | 3471 | 3834 | 98.6 |

### KV Cache and Flash Attention Variations (Vulkan single GPU)

| FA | KV Cache | pp256 | pp512 | pp2048 | pp4096 | tg128 |
|----|----------|------:|------:|-------:|-------:|------:|
| on | F16 | 2227 | 2841 | 2837 | 2833 | 114.3 |
| on | Q8_0 | 2223 | 2815 | 2750 | 2737 | 113.3 |
| on | Q4_0 | 2279 | 2839 | 2785 | 2743 | 113.1 |
| off | F16 | 2215 | 2783 | 2710 | 2663 | 105.1 |

KV quant has no meaningful impact. Flash attention gives ~8% TG boost.

### CPU Benchmarks

| Platform | Threads | pp64 | pp128 | pp256 | pp512 | tg128 |
|----------|--------:|-----:|------:|------:|------:|------:|
| Windows (Vulkan backend, ngl=0) | 12 | 81 | 138 | 229 | 381 | 6.5 |
| WSL (pure CPU) | 12 | 65 | 69 | 72 | 73 | 8.3 |
| WSL (pure CPU) | 24 | -- | -- | 89 | -- | 6.8 |

### Winner: Vulkan single GPU -- 114.3 t/s TG, ~2840 t/s PP

---

## Qwen3-Coder-Next -- Q4_K_XL (41.5 GiB, 79.67B params, 3B active)

Does NOT fit on 1x RTX 3090. Requires dual-GPU for full offload.

### GPU Benchmarks

| Backend | GPUs | ngl | FA | pp64 | pp128 | pp256 | pp512 | pp1024 | pp2048 | tg128 |
|---------|------|----:|---:|-----:|------:|------:|------:|-------:|-------:|------:|
| **CUDA (WSL)** | **2** | **99** | **on** | **654** | **859** | **1267** | **1702** | **2076** | **2344** | **90.0** |
| CUDA (WSL) | 2 | 99 | on (Q8 KV) | -- | -- | 1264 | -- | 2102 | 2400 | 90.0 |
| Vulkan (Win) | 1 | 26 | on | 96 | 152 | 240 | 381 | -- | -- | 9.1 |
| Vulkan (Win) | 1 | 25 | on | -- | -- | 235 | -- | -- | -- | 9.0 |
| Vulkan (Win) | 1 | 20 | on | -- | -- | 210 | -- | -- | -- | 7.7 |
| Vulkan (Win) | 1 | 15 | on | 74 | 116 | 186 | 296 | -- | -- | 6.8 |
| Vulkan (Win) | 1 | 27 | on | -- | -- | 99* | -- | -- | -- | 9.5 |
| Vulkan (Win) | 1 | 28+ | -- | OOM | OOM | OOM | OOM | OOM | OOM | OOM |

*ngl27: PP drops due to VRAM pressure. ngl26 is the sweet spot for single GPU Vulkan.

### CPU Benchmarks

| Platform | Threads | pp64 | pp128 | pp256 | tg128 |
|----------|--------:|-----:|------:|------:|------:|
| Windows (Vulkan backend, ngl=0) | 12 | 44 | 74 | 121 | 5.8 |
| WSL (pure CPU) | 12 | 51 | 55 | 57 | 7.8 |

### Winner: CUDA dual-GPU -- 90.0 t/s TG (10x vs single GPU Vulkan)

---

## Server Benchmark (Qwen3.5, CUDA dual-GPU, llama-server API)

| Test | Tokens | Time | TPS |
|------|-------:|-----:|----:|
| Single short (64 tok) | 64 | 879ms | 72.8 |
| Medium coding (256 tok) | 256 | 2.8s | 92.7 |
| Long generation (512 tok) | 512 | 5.5s | 93.6 |
| 5 concurrent requests | 640 total | 4.9s | 131.7 aggregate |
| 10 concurrent requests | 1280 total | 10.3s | 124.2 aggregate |

---

## vLLM Benchmark (Qwen3.5-35B-A3B-GPTQ-Int4, March 6, 2026)

vLLM 0.17.0rc1.dev126 (nightly), `--quantization moe_wna16`, single GPU (TP=1)

### Single Request Performance

| Config | 128 tok | 256 tok | 512 tok | 1024 tok | GPU Mem |
|--------|--------:|--------:|--------:|---------:|---------|
| **TP=1, ctx=8k** | 98.6 | 103.3 | **106.0** | **106.7** | 22.7 GB |
| TP=1, ctx=16k | 103.9 | - | 106.1 | 106.6 | 22.7 GB |
| TP=1, ctx=16k, fp8 KV | 101.5 | - | 104.1 | 105.0 | 22.7 GB |
| TP=1, ctx=16k, prefix cache | 103.9 | - | 105.2 | 105.7 | 22.7 GB |
| TP=2, ctx=32k | 5.8 | 15.1 | 15.5 | 15.7 | 24.3 GB x2 |
| TP=1, enforce-eager | 6.1 | 14.8 | 14.9 | 15.0 | 21.0 GB |

### Concurrent Request Performance

| Config | 5 concurrent (agg) | 10 concurrent (agg) |
|--------|--------------------:|---------------------:|
| **TP=1, ctx=16k, prefix cache** | **156.5** | **154.5** |
| TP=2, ctx=32k | 9.1 | 12.1 |
| TP=1, enforce-eager | 48.0 | 70.5 |

### Winner: TP=1, ctx=16k, prefix caching -- 105 t/s single, 155 t/s concurrent

---

## Key Findings

1. **Vulkan > CUDA for single-GPU models** (Qwen3.5): 114 vs 98 t/s. Multi-GPU overhead hurts TG.
2. **CUDA essential for multi-GPU models** (Coder-Next): 90 vs 9 t/s. 10x speedup with dual GPU.
3. **vLLM GPTQ-Int4 beats CUDA llama.cpp** for Qwen3.5: 105 vs 98 t/s. Best concurrent: 155 vs 132 agg.
4. **vLLM TP=2 is 7x slower than TP=1** on PCIe (no NVLink). Never use TP=2 for lightweight MoE.
5. **vLLM enforce-eager is 7x slower** than CUDA graphs. Always use CUDA graphs.
6. **KV cache quantization has zero speed impact** -- use Q4_0/fp8 to save VRAM for longer contexts.
7. **Flash attention gives ~8% TG boost** on Vulkan. Always enable.
8. **Vulkan dual-GPU is broken** in llama.cpp b8181 -- splits model terribly (30 t/s vs 114).
9. **Server overhead is minimal** -- 93.6 t/s sustained vs 97.6 raw llama-bench (~4% overhead).
10. **vLLM concurrent scales better** -- 155 agg TPS (vLLM) vs 132 agg TPS (llama.cpp).

## Engines Tested

| Engine | Version | Status | Notes |
|--------|---------|--------|-------|
| **vLLM (GPTQ-Int4)** | **0.17.0rc1 nightly** | **Working** | **Best concurrent (155 t/s agg), Qwen3.5 only** |
| llama.cpp (Vulkan) | b8181 | Working | Best single-req Qwen3.5 (114 t/s) |
| llama.cpp (CUDA) | b8204 | Working | Best for Coder-Next dual-GPU (90 t/s) |
| llama.cpp (CPU) | b8181/b8204 | Working | Slow (6-8 t/s TG) |
| vLLM | 0.16.0 | Failed | qwen35moe not supported (need nightly) |
| llama-cpp-python | 0.3.16 | Failed | qwen35moe not supported (bundled llama.cpp too old) |
| LM Studio | v1.104.2 | Failed | qwen35moe not supported (needs v0.4.5+) |
| Ollama | v0.11.4 | Failed | SIGSEGV on dual 3090 GPU detection |
