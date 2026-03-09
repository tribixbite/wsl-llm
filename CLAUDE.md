# WSL LLM Project - Claude Instructions

## Project Location
- **Repo**: `~/git/wsl-llm` (WSL Ubuntu-20.04, user `matilda`)
- **Models**: `/mnt/c/Users/Will/.lmstudio/models/unsloth/` (Windows FS, symlinked to `~/models/`)
- **llama.cpp**: `~/llama.cpp` (built from source, tag b8204, CUDA enabled)
- **Python venv**: `~/bench_env` (PyTorch 2.10.0+cu130, vLLM 0.17.0rc1 nightly, llama-cpp-python 0.3.16)

## Hardware
- 2x RTX 3090 (24GB each, 48GB total), Compute 8.6
- 12c/24t CPU, 64GB DDR4
- Windows 11 + WSL2 Ubuntu 22.04
- NVIDIA Driver 591.74, CUDA 13.1, Toolkit 12.6

## Current Models (as of March 6, 2026)

| Model | File | Size | Architecture | Engine |
|-------|------|------|-------------|--------|
| Qwen3.5-35B-A3B | `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` | 22.2 GB (20.7 GiB) | qwen35moe | llama.cpp |
| Qwen3.5-35B-A3B | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (HF) | ~23 GB | qwen35moe | **vLLM** |
| Qwen3-Coder-Next | `Qwen3-Coder-Next-UD-Q4_K_XL.gguf` | 44.6 GB (41.5 GiB) | qwen3next | llama.cpp |

GGUF models use Unsloth Dynamic v2.0 quantization. GPTQ-Int4 is official Qwen safetensors (auto-downloaded by vLLM).

## Environment Setup

```bash
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
```

## Key Commands

```bash
# vLLM Server (Qwen3.5 GPTQ, 105 t/s single / 155 t/s concurrent, BEST multi-user)
source ~/bench_env/bin/activate
export HF_HOME=/mnt/c/Users/Will/.cache/huggingface
export PYTHONPATH=/home/matilda/bench_env/lib/python3.10/site-packages:$PYTHONPATH
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --quantization moe_wna16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8080 \
  --language-model-only --enable-prefix-caching

# llama.cpp Server (CUDA, Qwen3.5)
~/llama.cpp/build/bin/llama-server -m ~/models/qwen35-q4.gguf \
  -ngl 99 -fa on -c 8192 --host 0.0.0.0 --port 8080 --jinja

# llama.cpp Server (CUDA, Coder-Next, dual-GPU, 90 t/s)
~/llama.cpp/build/bin/llama-server -m ~/models/coder-next-q4.gguf \
  -ngl 99 -fa on -c 16384 --host 0.0.0.0 --port 8080 --jinja

# Benchmark (CUDA dual-GPU)
~/llama.cpp/build/bin/llama-bench -m ~/models/qwen35-q4.gguf -ngl 99 -fa on -p 256,512,1024 -n 128
```

## Critical Notes

- **vLLM GPTQ is best for concurrent** -- 105 t/s single, 155 t/s aggregate (vs llama.cpp 132 agg)
- **vLLM needs nightly 0.17.0rc1+** -- v0.16.0 does NOT support Qwen3.5 architecture
- **vLLM TP=2 is 7x slower** -- PCIe overhead kills MoE perf; always use TP=1
- **vLLM enforce-eager is 7x slower** -- CUDA graphs essential for this model
- **Vulkan beats CUDA for Qwen3.5 TG** (114 vs 98 t/s) because model fits on 1 GPU
- **CUDA essential for Coder-Next** (90 vs 9 t/s) because it needs both GPUs
- **Coder-Next impossible on vLLM** -- AWQ-4bit is 45.9GB, no room for KV cache in 48GB
- **Vulkan dual-GPU is broken** -- always use `-ts 1,0` to force single GPU on Windows Vulkan
- **b8204 changed `-fa` syntax** -- now requires `-fa on` not just `-fa`
- **KV cache quant has no effect on speed** -- use Q4_0/fp8 freely for longer contexts
- **Flash attention gives ~8% TG boost** on Vulkan (114 vs 105 t/s)
- **Models are on Windows FS** (`/mnt/c/...`) via symlinks -- consider copying to Linux FS for better I/O
- **WSL sudo password**: (not stored in repo — ask user)

## Sampling Parameters

- **Qwen3.5 instruct**: temp=0.7, top_p=0.8, top_k=20
- **Qwen3.5 thinking**: temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
- **Coder-Next**: temp=1.0, top_p=0.95, top_k=40, min_p=0.01

## Next Steps (Pending)

1. Wait for vLLM 0.17.0 stable release (currently using nightly)
2. Test speculative decoding with MTP (multi-token prediction) on vLLM
3. Test longer contexts (32k+) when vLLM adds better memory management
4. Update llama.cpp when newer builds improve Vulkan multi-GPU
5. Update LM Studio to v0.4.6, Ollama to latest
6. Consider copying models to Linux FS for better I/O

## File Structure

```
~/git/wsl-llm/          # This repo
~/llama.cpp/             # llama.cpp b8204 (CUDA build)
~/models/                # Symlinks to Windows model files
  qwen35-q4.gguf        -> /mnt/c/Users/Will/.lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/...
  coder-next-q4.gguf    -> /mnt/c/Users/Will/.lmstudio/models/unsloth/Qwen3-Coder-Next-GGUF/...
~/bench_env/             # Python venv (vLLM 0.17.0rc1, PyTorch 2.10, llama-cpp-python)
~/server_bench.py        # Server benchmark script
/mnt/c/Users/Will/.cache/huggingface/  # HF model cache (GPTQ-Int4 safetensors)
```

## Reference Docs

- `docs/claude.md` -- Detailed learnings and architecture notes
- `BENCHMARKS.md` -- Full benchmark results table
- `OPTIMIZATION_FINDINGS.md` -- Optimization experiments and conclusions
