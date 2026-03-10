# WSL LLM Project - Claude Instructions

## Project Location
- **Repo**: `~/git/wsl-llm` (WSL Ubuntu-20.04, user `matilda`)
- **Models**: `~/models/` (native Linux FS for Qwen3.5, Windows FS symlinks for Coder-Next)
- **ik_llama.cpp**: `~/ik_llama.cpp` (PRIMARY, built from source, CUDA + FA_ALL_QUANTS)
- **llama.cpp**: `~/llama.cpp` (upstream, built from source, CUDA + FA_ALL_QUANTS)
- **Python venv**: `~/bench_env` (PyTorch 2.10.0+cu130, vLLM 0.17.0rc1 nightly, llama-cpp-python 0.3.16)

## Hardware
- 2x RTX 3090 (24GB each, 48GB total), Compute 8.6, PCIe (no NVLink)
- 12c/24t CPU, 64GB DDR4
- Windows 11 + WSL2 Ubuntu 22.04
- NVIDIA Driver 591.74, CUDA 13.1, Toolkit 12.6

## Current Models (as of March 9, 2026)

| Model | File | Size | Architecture | Engine |
|-------|------|------|-------------|--------|
| Qwen3.5-35B-A3B | `~/models/qwen35-q4.gguf` (Linux FS) | 20.7 GiB | qwen35moe | **ik_llama.cpp** |
| Qwen3.5-35B-A3B | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (HF) | ~23 GB | qwen35moe | vLLM |
| Qwen3-Coder-Next | `~/models/coder-next-q4.gguf` (symlink) | 41.5 GiB | qwen3next | llama.cpp |

## Environment Setup

```bash
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
```

## Key Commands

```bash
# ik_llama.cpp Server (BEST: 122 t/s server TG, 131k ctx, 4 slots)
~/ik_llama.cpp/build/bin/llama-server -m ~/models/qwen35-q4.gguf \
  -ngl 99 -fa 1 -c 131072 -np 4 \
  --cache-type-k bf16 --cache-type-v bf16 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --host 0.0.0.0 --port 8080 --jinja \
  --api-key "<your-api-key>" --alias "qwen3.5-35b-a3b"

# vLLM Server (BEST concurrent: 155 t/s aggregate, but max 16k ctx)
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

# llama.cpp Server (CUDA, Coder-Next, dual-GPU, 90 t/s)
~/llama.cpp/build/bin/llama-server -m ~/models/coder-next-q4.gguf \
  -ngl 99 -fa on -c 16384 --host 0.0.0.0 --port 8080 --jinja

# Cloudflare tunnel (llm.pet -> localhost:8080)
cloudflared tunnel --config ~/.cloudflared/config.yml run wsl-llm
```

## Critical Notes

- **ik_llama.cpp is 2x faster** than upstream llama.cpp in server mode (122 vs 58 t/s)
- **ik_llama.cpp graph splits = 3** vs upstream's 23 (key optimization)
- **`-sm graph` is USELESS on PCIe** -- 10 t/s (only helps NVLink); always use layer split
- **LLAMA_SET_ROWS=1 has no effect** on this model
- **Smart Expert Reduction (-ser) has no effect** -- experts aren't the bottleneck
- **vLLM GPTQ is best for high concurrency** -- 155 t/s aggregate (but max 16k ctx)
- **vLLM needs nightly 0.17.0rc1+** -- v0.16.0 does NOT support Qwen3.5
- **vLLM TP=2 is 7x slower** -- PCIe overhead; always use TP=1
- **Coder-Next impossible on vLLM** -- AWQ-4bit is 45.9GB, no room for KV cache
- **262k bf16 KV can crash Windows** -- uses ~30GB total, leaves no headroom
- **Model on Linux FS** -- copied from Windows FS for better I/O
- **Speculative decoding NOT supported** for Qwen3.5 MoE (GitHub Issue #20039)
- **WSL sudo password**: (not stored in repo -- ask user)

## Sampling Parameters

- **Qwen3.5 Unsloth recommended**: temp=0.6, top_p=0.95, top_k=20
- **Qwen3.5 thinking**: temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5
- **Coder-Next**: temp=1.0, top_p=0.95, top_k=40, min_p=0.01

## Build Commands

```bash
# ik_llama.cpp (requires cmake 3.25+: pip install cmake --user)
cd ~/ik_llama.cpp
~/.local/bin/cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc
~/.local/bin/cmake --build build -j$(nproc)

# upstream llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)
```

## File Structure

```
~/git/wsl-llm/          # This repo
~/ik_llama.cpp/          # ik_llama.cpp fork (PRIMARY, 2x faster server)
~/llama.cpp/             # Upstream llama.cpp (CUDA build)
~/models/
  qwen35-q4.gguf        # Native Linux FS copy (20.7 GiB)
  coder-next-q4.gguf    -> /mnt/c/.../Qwen3-Coder-Next-GGUF/... (symlink)
~/bench_env/             # Python venv (vLLM 0.17.0rc1, PyTorch 2.10)
~/.cloudflared/          # Cloudflare tunnel config + credentials
/mnt/c/Users/Will/.cache/huggingface/  # HF model cache (GPTQ-Int4)
```

## Reference Docs

- `docs/claude.md` -- Detailed learnings and architecture notes
- `BENCHMARKS.md` -- Full benchmark results table
- `OPTIMIZATION_FINDINGS.md` -- Optimization experiments and conclusions
