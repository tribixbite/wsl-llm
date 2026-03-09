# Claude Learnings: WSL LLM Setup

## Machine Specifications

### Hardware
- **GPUs**: 2x NVIDIA GeForce RTX 3090 (24GB VRAM each, 48GB total), Compute 8.6, VMM supported
- **CPU**: 12 physical cores, 24 threads
- **RAM**: 64GB DDR4
- **OS**: Windows 11 with WSL2 (Ubuntu 22.04)

### Software Versions (as of March 5, 2026)
- **NVIDIA Driver**: 591.74 (Windows host)
- **CUDA Version**: 13.1 (nvidia-smi) / Toolkit 12.6 (dev tools in WSL)
- **llama.cpp**: b8204 (CUDA build in WSL), b8181 (Vulkan on Windows)
- **Python venv**: `~/bench_env` with PyTorch 2.9.1+cu124, vLLM 0.16.0
- **WSL User**: matilda
- **Node.js**: 22 (NodeSource), claude-code at `/usr/bin/claude`, bun at `~/.bun/bin/bun`

### Current Models
| Model | File | Size | Params | Active | Arch | Context |
|-------|------|------|--------|--------|------|---------|
| Qwen3.5-35B-A3B | Q4_K_XL | 20.7 GiB | 34.66B | 3B | qwen35moe | 262K |
| Qwen3-Coder-Next | Q4_K_XL | 41.5 GiB | 79.67B | 3B | qwen3next | 262K |

Both Unsloth Dynamic v2.0 quantized. Qwen3.5 re-downloaded March 4 (MXFP4 replaced with Q4_K layers).
Models stored at `/mnt/c/Users/Will/.lmstudio/models/unsloth/`, symlinked to `~/models/`.

## Critical Lessons Learned

### 1. WSL PATH Configuration (Changed from Previous Setup)
Previously set `appendWindowsPath = false` to avoid PATH pollution.
Now changed to `appendWindowsPath = true` because `claude` and `bun` binaries need Windows PATH.
Installed Node.js 22 natively in WSL for claude-code to work.

### 2. CUDA in WSL -- Only Install Toolkit
WSL2 uses Windows NVIDIA driver via stub library. Only install `cuda-toolkit-12-6`, NEVER `cuda-12-6`.
Installing full CUDA package conflicts with the WSL driver architecture.

### 3. Driver Version Sync
After Windows NVIDIA driver update, WSL libs can get out of sync. Symptom: `nvidia-smi` shows "Unable to determine device handle". Fix: full Windows reboot. `wsl --shutdown` alone is NOT sufficient.

### 4. Vulkan vs CUDA Performance
**Vulkan beats CUDA for single-GPU models** (Qwen3.5: 114 vs 98 t/s TG).
Multi-GPU overhead hurts TG when the model fits on 1 GPU.
**CUDA essential for multi-GPU models** (Coder-Next: 90 vs 9 t/s TG).
Vulkan can only use 1 GPU effectively -- dual-GPU Vulkan split is broken (~30 t/s).

### 5. KV Cache Quantization Has Zero Speed Impact
Tested F16, Q8_0, Q4_0 -- all give identical throughput (~113 t/s).
Use Q4_0 freely to save VRAM for longer contexts without any performance cost.

### 6. Flash Attention
Gives ~8% TG boost on Vulkan (114 vs 105 t/s). Always enable.
b8204 changed syntax from `-fa` to `-fa on` (now requires explicit value).

### 7. Model Architecture Details
**Qwen3.5-35B-A3B**: MoE with 3B active params. Fits on 1x 3090.
**Qwen3-Coder-Next**: 80B MoE (512 experts, 10 active, hybrid attn+Mamba SSM).
Needs 2x 3090 for full GPU offload. Max ngl on single Vulkan GPU: 26 (OOM at 28+).

### 8. Inference Engine Compatibility

| Engine | qwen35moe | qwen3next | Notes |
|--------|-----------|-----------|-------|
| llama.cpp b8181+ | Yes | Yes | Best for GGUF. 114 t/s (Vulkan) / 90 t/s (CUDA) |
| **vLLM 0.17.0rc1 (nightly)** | **Yes (GPTQ-Int4)** | No (AWQ too large) | **105 t/s single, 155 t/s concurrent** |
| vLLM 0.16.0 | No | No | Qwen3.5 not supported until nightly |
| llama-cpp-python 0.3.16 | No | No | Bundled llama.cpp too old |
| LM Studio | No | No | Needs v0.4.5+ |
| Ollama v0.11.4 | No | No | Crashes on dual 3090 |

### 9. WSL Configuration

**`/etc/wsl.conf`**:
```ini
[user]
default=matilda

[interop]
appendWindowsPath = true

[boot]
systemd=true
```

**`C:\Users\Will\.wslconfig`**:
```ini
[wsl2]
# networkingMode=mirrored  # disabled -- causes error 0x8007054f
dnsTunneling=true
autoProxy=true
memory=48GB
processors=24
gpuSupport=true
```

### 10. Sampling Parameters (from Unsloth docs)

**Qwen3.5 instruct**: temp=0.7, top_p=0.8, top_k=20
**Qwen3.5 thinking**: temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
**Qwen3.5 precise coding**: temp=0.6, top_p=0.95, top_k=20
**Coder-Next** (non-thinking only): temp=1.0, top_p=0.95, top_k=40, min_p=0.01
All models: repetition_penalty=1.0 (disabled)

### 11. Recommended Launch Commands

**Qwen3.5 (vLLM, WSL, 105 t/s single / 155 t/s concurrent, BEST for multi-user):**
```bash
source ~/bench_env/bin/activate
export HF_HOME=/mnt/c/Users/Will/.cache/huggingface
export PYTHONPATH=/home/matilda/bench_env/lib/python3.10/site-packages:$PYTHONPATH
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --quantization moe_wna16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8080 \
  --language-model-only \
  --enable-prefix-caching
```

**Qwen3.5 (Vulkan, Windows, 114 t/s, BEST single-request):**
```bash
llama-server -m Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 -fa 1 -ts 1,0 -ctk q4_0 -ctv q4_0 -c 8192 \
  --host 0.0.0.0 --port 8080 --jinja
```

**Qwen3.5 (CUDA, WSL, 98 t/s):**
```bash
~/llama.cpp/build/bin/llama-server -m ~/models/qwen35-q4.gguf \
  -ngl 99 -fa on -c 8192 --host 0.0.0.0 --port 8080 --jinja
```

**Coder-Next (CUDA, WSL, 90 t/s, dual-GPU, only viable option):**
```bash
~/llama.cpp/build/bin/llama-server -m ~/models/coder-next-q4.gguf \
  -ngl 99 -fa on -c 16384 --host 0.0.0.0 --port 8080 --jinja
```

### 12. Build Configuration (llama.cpp b8204, WSL CUDA)
```bash
export PATH="/usr/local/cuda-12.6/bin:$PATH"
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) -- llama-bench llama-server llama-cli
```

### 13. Files and Paths

| Path | Description |
|------|-------------|
| `~/git/wsl-llm/` | This repo (docs, scripts, configs) |
| `~/llama.cpp/` | llama.cpp b8204 source + CUDA build |
| `~/models/qwen35-q4.gguf` | Symlink to Qwen3.5 GGUF |
| `~/models/coder-next-q4.gguf` | Symlink to Coder-Next GGUF |
| `~/bench_env/` | Python venv |
| `~/server_bench.py` | Server benchmark script |
| `/etc/wsl.conf` | WSL config |
| `C:\Users\Will\.wslconfig` | WSL2 resource config |
| `C:\Users\Will\llama-cpp-vulkan\` | Windows Vulkan llama.cpp b8181 |

### 14. vLLM Critical Notes
- **vLLM 0.17.0rc1 nightly required** for Qwen3.5 support (0.16.0 doesn't support the architecture)
- **Install**: `uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly`
- **TP=1 only**: TP=2 is 7x slower due to PCIe overhead on RTX 3090 (no NVLink)
- **CUDA graphs required**: `--enforce-eager` drops from 105 to 15 TPS
- **fp8 KV cache**: No TPS impact, saves VRAM for concurrent sequences; requires `--max-num-batched-tokens 4096` to avoid Mamba block alignment error
- **Model cache**: HF downloads to `/mnt/c/Users/Will/.cache/huggingface` (set `HF_HOME`)
- **First request is slow**: CUDA graph compilation ~10s; subsequent requests are fast
- **Coder-Next impossible on vLLM**: AWQ-4bit is 45.9GB, no room for KV cache in 48GB

## Performance Summary

| Model | Backend | TG (t/s) | Concurrent (agg) | Context | Notes |
|-------|---------|----------|----------|---------|-------|
| **Qwen3.5** | **vLLM GPTQ-Int4** | **105** | **155** | **16k** | **Best concurrent throughput** |
| Qwen3.5 | Vulkan 1 GPU | 114 | 132 | 8k+ | Best single-request TPS |
| Qwen3.5 | CUDA 2 GPU | 98 | 132 | 8k+ | Second option |
| Coder-Next | CUDA 2 GPU | 90 | - | 16k+ | Only viable option |
| Coder-Next | Vulkan 1 GPU | 9 | - | - | Too slow |
