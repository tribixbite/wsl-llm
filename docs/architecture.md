# Architecture

> Last updated: April 2026

## Hardware

| Component | Spec | Notes |
|-----------|------|-------|
| CPU | AMD Ryzen 9 5900X (12c/24t) | Zen 3, CPPC enabled |
| RAM | 64 GB DDR4-3600 CL16 | 4x Crucial Ballistix, 1:1 FCLK @ 1800 MHz |
| GPU 0 | ASUS RTX 3090 24 GB | CPU-direct PCIe 3.0 x16 |
| GPU 1 | NVIDIA RTX 3090 FE 24 GB | X570 chipset slot, x8 max |
| Storage | Samsung 990 PRO 4 TB | NVMe |
| Board | ASUS ROG STRIX X570-E GAMING WIFI II | BIOS 5041+ required |
| OS | Windows 11 + WSL2 Ubuntu 22.04 | NVIDIA Driver 591.86, CUDA Toolkit 12.6 |

No NVLink bridge. Both GPUs communicate over PCIe only.

## Service Architecture

```
Internet
  |
  v
llm.pet (Cloudflare Tunnel)
  |
  v
Open WebUI (:3000)  --->  LiteLLM Proxy (:4000)  --->  llama-server (:8080)
                              |                            ik_llama.cpp
                              v                            Qwen3.5-35B-A3B
                          PostgreSQL (:5434)                RTX 3090 (GPU 0)
                          (Docker, litellm-pg)
```

All services run as systemd units inside WSL2 and start automatically on boot.

| Service | Port | What |
|---------|------|------|
| `llama-server` | 8080 | ik_llama.cpp inference (OpenAI-compatible API) |
| `litellm` | 4000 | API proxy, key management, admin UI |
| `litellm-pg` | 5434 | PostgreSQL (Docker container) |
| `cloudflared` | -- | Cloudflare tunnel (llm.pet -> :4000) |

Windows Task Scheduler starts WSL on boot (`setup-autostart.ps1`), which triggers systemd, which starts everything.

## Engine Comparison

| | ik_llama.cpp | upstream llama.cpp | vLLM |
|-|-|-|-|
| **Single-request** | **114 t/s** (262k ctx, single GPU) | 56 t/s | 105 t/s |
| **Concurrent (5x)** | -- | 132 t/s agg | **157 t/s agg** |
| **Max context** | **262k** | 262k | 16k (TP=1) |
| **Graph splits** | **3** | 23 | N/A |
| **Server overhead** | Minimal | 2x penalty vs raw bench | Minimal |
| **Cold start** | ~5 s | ~5 s | ~150 s |
| **Coder-Next** | Not tested | 90 t/s | Not viable (OOM) |

Why ik_llama.cpp is 2x faster in server mode: it reduces cross-GPU synchronization points (graph splits) from 23 to 3. Each split is a PCIe round-trip. Raw bench difference is only ~6% (128 vs 121 t/s), but the server overhead from 23 synchronization points is devastating on PCIe bandwidth.

## Model Architecture: Qwen3.5-35B-A3B

- 34.66B total parameters, **3B active** per token (MoE)
- 40 layers total: **10 attention layers** + **30 DeltaNet layers** (linear recurrence)
- Only the 10 attention layers have KV cache; DeltaNet layers use fixed-size recurrent state
- This is why the KV cache is small even at 262k context
- Quantization: Q4_K_XL GGUF (20.7 GiB on disk)
- Architecture ID: `qwen35moe`

### VRAM Budget

**Production config: 262k context, q4_0 KV + Hadamard, single GPU (GPU 0), 2 slots**

| Component | Size |
|-----------|-----:|
| Model weights (Q4_K_XL) | 20.7 GiB |
| KV cache (262k, q4_0, 10 attn layers) | ~1.3 GiB |
| Compute buffers + recurrent state | ~0.9 GiB |
| **Total** | **~23 GiB / 24 GiB on GPU 0** |

GPU 1 is free for other workloads. The switch from bf16 to q4_0 + Hadamard KV reduced cache from 5.1 GiB to ~1.3 GiB (3.8x), allowing the full 262k context to fit on a single GPU. 262k is stable on single GPU with q4_0 KV.

### Qwen3-Coder-Next (secondary model)

- 79.67B total, 3B active (MoE), 512 experts, 10 active per token
- Q4_K_XL GGUF: 41.5 GiB (requires both GPUs, no room for vLLM)
- 90 t/s on upstream llama.cpp CUDA (only viable engine)

## Sampling Parameters

Source: Qwen/Unsloth official recommendations. **Do NOT use temp=0 (greedy). Performance degrades.**

### Qwen3.5-35B-A3B

| Mode | temp | top_p | top_k | min_p | presence_penalty |
|------|-----:|------:|------:|------:|-----------------:|
| Non-thinking general | 0.7 | 0.8 | 20 | 0.0 | 1.5 |
| Non-thinking coding (server default) | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking general | 1.0 | 0.95 | 20 | 0.0 | 1.5 |
| Thinking coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Non-thinking reasoning | 1.0 | 1.0 | 40 | 0.0 | 2.0 |

### Qwen3-Coder-Next

| Mode | temp | top_p | top_k | min_p |
|------|-----:|------:|------:|------:|
| All modes | 1.0 | 0.95 | 40 | 0.01 |

### Thinking Mode

`--reasoning-budget 0` disables thinking by default. Clients re-enable per-request:
```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

### KV Cache Type

| Engine | Recommended | Notes |
|--------|------------|-------|
| ik_llama.cpp / llama.cpp | **q4_0 + Hadamard** (`-khad -vhad`) | Near-bf16 quality at 3.8x less memory ("Even better Q4_0 KV cache" commits) |
| vLLM | auto (bf16) | FP8 available, saves VRAM, negligible speed difference |

q4_0 + Hadamard KV gives near-bf16 quality at 3.8x less memory; combined with single GPU it is also slightly faster (+7%).

## Critical Lessons

### PCIe Lane Allocation

- GPU 0 (PCIEX16_1): CPU-direct, PCIe 3.0 x16
- GPU 1 (PCIEX16_2): X570 chipset, **x8 max by design**
- PCIe link width drops to x4/x8 at idle -- normal power management, scales back under load
- 100+ t/s inference confirms no bandwidth bottleneck at x8

### Single GPU > Dual GPU (When the Model Fits)

Running on a single GPU (GPU 0, `CUDA_VISIBLE_DEVICES=0`) is ~7% faster than splitting across two GPUs, because it eliminates all PCIe synchronization overhead between cards. With q4_0 + Hadamard KV, the full 262k context fits in ~23 GiB on one 24 GB card. The freed GPU 1 can run other workloads independently.

### ASPM Crashes (CUDA error 999)

ASPM L0s causes GPU PCIe link wake failure on idle-to-active transitions. Combined with the default 2-second Windows TDR timeout, this manifested as crashes every ~7 hours.

**Fix:**
1. BIOS: Disable ASPM (AMD CBS -> NBIO -> SMU Common -> ASPM = Disabled)
2. Windows registry: TDR timeout 2s -> 60s
3. NVIDIA Control Panel: Power Management = Prefer Maximum Performance
4. Windows Power Options: PCI Express Link State Power Management = Off

### What Helps

| Optimization | Impact |
|-------------|--------|
| ik_llama.cpp fork | **2.14x server TPS** (graph splits 23 -> 3) |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | +23% raw bench |
| Model on Linux FS (ext4) | +5-10% I/O vs /mnt/c |
| Flash attention | +8% TG speed |
| `--no-context-shift` | Better quality at long context |
| vLLM prefix caching | Better concurrent throughput |
| ReBAR (BIOS) | Free perf for GPU memory access |
| `q4_0 KV + Hadamard` | 3.8x KV memory reduction, near-bf16 quality |
| `--fit (auto-fit)` | Optimal VRAM utilization, replaces manual -ngl |
| `Single GPU (CUDA_VISIBLE_DEVICES=0)` | +7% speed (eliminates PCIe cross-GPU transfers) |

### What Does NOT Help

| Optimization | Result |
|-------------|--------|
| `LLAMA_SET_ROWS=1` | No effect |
| `-sm graph` (graph split mode) | 10 t/s -- only helps NVLink |
| Smart Expert Reduction (`-ser`) | No effect -- experts are not the bottleneck |
| Fused MoE (`-fmoe`) | Already enabled by default |
| vLLM TP=2 | 7x slower -- PCIe kills MoE parallelism |
| vLLM enforce-eager | 7x slower -- CUDA graphs are essential |
| KV cache quantization (alone) | Negligible speed impact alone, but enables single-GPU fit which gives +7% |
| Speculative decoding (MTP) | Not supported for Qwen3.5 MoE |

### BIOS Essentials (X570-E)

| Setting | Value | Why |
|---------|-------|-----|
| Above 4G Decoding | Enabled | Required for 2x 24 GB GPUs |
| Re-Size BAR | Auto | Free perf |
| CSM | Disabled | Required for ReBAR |
| ASPM | Disabled | Prevents GPU crash on idle -> active |
| PCIe Speed | Gen 3 | Eliminates X570 WHEA errors |
| SVM Mode | Enabled | Required for WSL2/Hyper-V |
| DOCP | Profile 1 | DDR4-3600 CL16 @ 1.35V |
| FCLK | 1800 MHz | 1:1 ratio with memory |
| DF C-States | Disabled | Prevents deep-idle reboots |

See [BIOS_AND_SYSTEM_OPTIMIZATION.md](BIOS_AND_SYSTEM_OPTIMIZATION.md) for the full checklist with voltage hierarchy and memory subtimings.

## Detailed References

- [BENCHMARKS.md](BENCHMARKS.md) -- Full benchmark tables, all engines, all configs
- [OPTIMIZATION_FINDINGS.md](OPTIMIZATION_FINDINGS.md) -- vLLM tuning, Coder-Next optimization, why things did not help
- [BIOS_AND_SYSTEM_OPTIMIZATION.md](BIOS_AND_SYSTEM_OPTIMIZATION.md) -- Complete BIOS settings, Windows registry, root cause analysis
