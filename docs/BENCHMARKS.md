# Benchmark Results & Findings

> **Hardware**: 2x RTX 3090 24GB (PCIe, no NVLink) | 12c/24t | 64GB DDR4 | Windows 11 + WSL2 Ubuntu 22.04
>
> **Last updated**: March 9, 2026

---

## Current Production Config

**Engine**: ik_llama.cpp (fork) | **Model**: Qwen3.5-35B-A3B Q4_K_XL (20.7 GiB) | **Context**: 131k | **Slots**: 4

```bash
~/ik_llama.cpp/build/bin/llama-server -m ~/models/qwen35-q4.gguf \
  -ngl 99 -fa 1 -c 131072 -np 4 \
  --cache-type-k bf16 --cache-type-v bf16 --no-context-shift \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --reasoning-budget 0 \
  --host 0.0.0.0 --port 8080 --jinja \
  --api-key "<key>" --alias "qwen3.5-35b-a3b"
```

**Result**: 112 t/s server TG, thinking OFF by default, exposed via llm.pet

---

## Qwen3.5-35B-A3B (20.7 GiB, 34.66B params, 3B active MoE)

### Engine Comparison — Server Mode (Real Requests)

| Engine | Build | Context | Single TPS | Concurrent 5 (agg) | Concurrent 10 (agg) | Notes |
|--------|-------|---------|----------:|--------------------:|---------------------:|-------|
| **ik_llama.cpp CUDA** | Mar 9 | **8k** | **120-122** | — | — | **Best single-req** |
| **ik_llama.cpp CUDA** | Mar 9 | **131k** | **112** | — | — | **Production config** |
| ik_llama.cpp CUDA | Mar 9 | 262k | 107-109 | — | — | Unstable, crashes |
| Upstream llama.cpp CUDA | latest master | 8k | 56-58 | 131.7 | 124.2 | 23 graph splits |
| **vLLM GPTQ-Int4** | 0.17.0rc1 | **16k** | **105-107** | **156.5** | **154.5** | **Best concurrent** |
| Vulkan (Windows native) | b8181 | 8k | — | — | — | No server bench done |

### Raw Benchmarks (llama-bench TG128)

| Backend | GPUs | FA | KV | Build | tg128 t/s |
|---------|------|----|----|-------|----------:|
| **ik_llama.cpp CUDA** | 2 | on | F16 | Mar 9 | **128** |
| Upstream CUDA (FA_ALL_QUANTS) | 2 | on | F16 | latest master | 121 |
| Upstream CUDA (old b8204) | 2 | on | F16 | b8204 | 98 |
| **Vulkan (Windows)** | **1 (GPU0)** | **on** | **F16** | **b8181** | **114.3** |
| Vulkan (Windows) | 1 (GPU1) | on | F16 | b8181 | 110.1 |
| CUDA (WSL, old) | 2 | on | F16 | b8204 | 97.6 |
| CUDA (WSL, old) | 1 (GPU0) | on | F16 | b8204 | 102.3 |

### Why ik_llama.cpp is 2x Faster in Server Mode

| Metric | Upstream | ik_llama.cpp |
|--------|----------|-------------|
| Raw bench (llama-bench) | 121 t/s | 128 t/s |
| **Server mode** | **56-58 t/s** | **120-122 t/s** |
| Graph splits | 23 | 3 |
| Ratio | 1x | **2.14x** |

The raw bench difference is only ~6%, but **server mode is 2.14x faster** because ik_llama.cpp reduces cross-GPU synchronization points (graph splits) from 23 to 3. Each graph split is a PCIe round-trip — devastating on PCIe bandwidth.

### Prompt Processing (tokens/second)

| Backend | pp64 | pp128 | pp256 | pp512 | pp1024 | pp2048 | pp4096 |
|---------|-----:|------:|------:|------:|-------:|-------:|-------:|
| Vulkan (Win, GPU0) | 1142 | 1586 | 2227 | 2841 | 2830 | 2837 | 2833 |
| CUDA (WSL, 2 GPU) | 929 | 1270 | 1901 | 2618 | 3171 | 3659 | 3650 |

### Flash Attention & KV Cache Variations (Vulkan, Single GPU)

| FA | KV Cache | pp256 | pp512 | pp2048 | pp4096 | tg128 |
|----|----------|------:|------:|-------:|-------:|------:|
| on | F16 | 2227 | 2841 | 2837 | 2833 | **114.3** |
| on | Q8_0 | 2223 | 2815 | 2750 | 2737 | 113.3 |
| on | Q4_0 | 2279 | 2839 | 2785 | 2743 | 113.1 |
| off | F16 | 2215 | 2783 | 2710 | 2663 | 105.1 |

**Conclusion**: KV quant has zero speed impact. Flash attention gives ~8% TG boost.

### vLLM Detailed Results (GPTQ-Int4, moe_wna16)

| Config | 128 tok | 512 tok | 1024 tok | GPU Mem | 5 conc (agg) | 10 conc (agg) |
|--------|--------:|--------:|---------:|---------|-----:|------:|
| **TP=1, ctx=16k, prefix cache** | 103.9 | **106.1** | **106.7** | 22.7 GB | **156.5** | **154.5** |
| TP=1, ctx=8k | 98.6 | 106.0 | 106.7 | 22.7 GB | — | — |
| TP=1, ctx=16k, fp8 KV | 101.5 | 104.1 | 105.0 | 22.7 GB | — | — |
| TP=2, ctx=32k | 5.8 | 15.5 | 15.7 | 24.3x2 | 9.1 | 12.1 |
| TP=1, enforce-eager | 6.1 | 14.9 | 15.0 | 21.0 GB | 48.0 | 70.5 |

### CPU Benchmarks

| Platform | Threads | pp256 | pp512 | tg128 |
|----------|--------:|------:|------:|------:|
| Windows Vulkan ngl=0 | 12 | 229 | 381 | 6.5 |
| WSL CPU only | 12 | 72 | 73 | 8.3 |
| WSL CPU only | 24 | 89 | — | 6.8 |

---

## Qwen3-Coder-Next (41.5 GiB, 79.67B params, 3B active MoE)

Does NOT fit on 1x RTX 3090. Requires dual-GPU.

### GPU Benchmarks

| Backend | GPUs | FA | tg128 t/s | Notes |
|---------|------|----|---------:|-------|
| **CUDA (WSL)** | **2** | **on** | **90.0** | **Only viable option** |
| Vulkan (Win) | 1, ngl=26 | on | 9.1 | Only ~62% of layers on GPU |
| Vulkan (Win) | 1, ngl=15 | on | 6.8 | |

### Coder-Next vLLM Feasibility

| Quant | Size | Fits 48GB? | Notes |
|-------|------|-----------|-------|
| AWQ-4bit | 45.9 GB | Barely | No room for KV cache |
| FP8 | ~80 GB | No | |
| GPTQ-Int4 | N/A | — | Doesn't exist |
| **GGUF Q4_K_XL** | **41.5 GB** | **Yes** | **llama.cpp only option** |

---

## Engine Compatibility Matrix

| Engine | Version | Qwen3.5 | Coder-Next | Status |
|--------|---------|---------|------------|--------|
| **ik_llama.cpp** | Mar 9 | **120 t/s** | Not tested | **PRIMARY** |
| llama.cpp (CUDA) | latest master | 58 t/s (server) | 90 t/s | Working |
| llama.cpp (Vulkan) | b8181 | 114 t/s (raw) | 9 t/s | Windows only |
| **vLLM** | **0.17.0rc1** | **105/155 t/s** | N/A | **Best concurrent** |
| vLLM | 0.16.0 | Not supported | Not supported | Failed |
| llama-cpp-python | 0.3.16 | Not supported | Not supported | Failed |
| LM Studio | v1.104.2 | Not supported | Not supported | Needs v0.4.5+ |
| Ollama | v0.11.4 | SIGSEGV | SIGSEGV | Crashes on dual 3090 |

---

## Sampling Parameters (Official Qwen/Unsloth)

Source: [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)

**Do NOT use greedy decoding (temp=0). Model performance degrades.**

### Qwen3.5-35B-A3B

| Mode | temp | top_p | top_k | min_p | presence_penalty |
|------|-----:|------:|------:|------:|-----------------:|
| **Non-thinking general** | 0.7 | 0.8 | 20 | 0.0 | 1.5 |
| **Non-thinking coding** | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| **Thinking general** | 1.0 | 0.95 | 20 | 0.0 | 1.5 |
| **Thinking coding** | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| **Non-thinking reasoning** | 1.0 | 1.0 | 40 | 0.0 | 2.0 |

### Qwen3-Coder-Next

| Mode | temp | top_p | top_k | min_p |
|------|-----:|------:|------:|------:|
| All | 1.0 | 0.95 | 40 | 0.01 |

### KV Cache

| Engine | Recommended | Notes |
|--------|------------|-------|
| llama.cpp | **bf16** | Unsloth recommended. Prevents gibberish at long context |
| llama.cpp | q4_0/q8_0 | Same speed, saves VRAM. Use if memory-constrained |
| vLLM | auto (bf16) | FP8 safe for 35B variant, saves VRAM |

---

## Optimization Experiments

### Things That Help

| Optimization | Impact | Details |
|-------------|--------|---------|
| **ik_llama.cpp fork** | **2.14x server TPS** | Graph splits 23→3, critical for PCIe |
| **GGML_CUDA_FA_ALL_QUANTS=ON** | +23% raw bench | 98→121 t/s (upstream) |
| **Model on Linux FS** | +5-10% I/O | Copy from /mnt/c to ext4 |
| **Flash attention** | +8% TG | 105→114 t/s (Vulkan bench) |
| **vLLM prefix caching** | Better concurrent | 154→156 agg with repeated prompts |
| **--no-context-shift** | Quality improvement | Qwen docs recommend for long context |
| **--reasoning-budget 0** | No wasted tokens | Disable thinking by default |

### Things That DON'T Help

| Optimization | Result | Details |
|-------------|--------|---------|
| LLAMA_SET_ROWS=1 | No effect | Tested single and concurrent |
| `-sm graph` (graph split mode) | 10 t/s | Only helps NVLink, useless on PCIe |
| Smart Expert Reduction (-ser) | No effect | Experts aren't the bottleneck |
| Fused MoE (-fmoe) | No effect | Already enabled by default |
| vLLM TP=2 | 7x slower | PCIe kills MoE parallelism |
| vLLM enforce-eager | 7x slower | CUDA graphs essential |
| KV cache quantization | Zero speed impact | Only saves VRAM |
| Speculative decoding (MTP) | Not supported | Qwen3.5 MoE not compatible |

---

## Infrastructure

| Component | Status | Details |
|-----------|--------|---------|
| ik_llama.cpp server | Running | Port 8080, 131k ctx, 4 slots, API key auth |
| Cloudflare tunnel | Running | llm.pet → localhost:8080 |
| Model location | Linux FS | ~/models/qwen35-q4.gguf (native ext4) |
| CUDA toolkit | 12.6 | /usr/local/cuda-12.6 |
| cmake | 4.2.3 | ~/.local/bin/cmake (pip installed for ik_llama.cpp) |

---

## VRAM Budget

### Qwen3.5 @ 131k context, bf16 KV, 4 slots

| Component | Size |
|-----------|-----:|
| Model weights | 20.7 GiB |
| KV cache (131k bf16) | 2.56 GiB |
| Compute buffers | ~0.8 GiB |
| Mamba recurrent state | ~0.1 GiB |
| **Total** | **~24.2 GiB / 48 GiB** |

### Qwen3.5 @ 262k context, bf16 KV, 2 slots (unstable)

| Component | Size |
|-----------|-----:|
| Model weights | 20.7 GiB |
| KV cache (262k bf16) | 5.12 GiB |
| Compute + RS | ~0.9 GiB |
| **Total** | **~26.7 GiB / 48 GiB** |

Headroom exists, but 262k is unstable due to CUDA/WSL memory pressure interactions.

---

## System Stability Notes

- **262k context crashes**: WSL VM gets killed by Windows due to combined GPU + system memory pressure. `.wslconfig` sets memory=48GB (all RAM), leaving nothing for Windows.
- **GPU 0 PCIe at 4x** (should be 16x): Degraded link width, possibly WSL/WDDM artifact or physical seating issue.
- **`dxg` ioctl errors**: Constant `dxgkio_is_feature_enabled: -75` in dmesg — known WSL2 issue, harmless but noisy.
- **drop_caches flurries**: WSL's mini_init aggressively reclaims memory when Windows is under pressure. Smoking gun for OOM-adjacent crashes.
