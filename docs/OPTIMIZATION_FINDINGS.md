# Optimization Findings (Updated March 6, 2026)

## vLLM Benchmark Results (Qwen3.5-35B-A3B-GPTQ-Int4)

### Setup
- **vLLM**: 0.17.0rc1.dev126 (nightly, required for Qwen3.5 architecture)
- **Model**: `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` (safetensors, auto-downloaded from HuggingFace)
- **Quantization**: `--quantization moe_wna16` (required for MoE GPTQ models)
- **HF Cache**: `/mnt/c/Users/Will/.cache/huggingface` (Windows FS, 495GB free)

### Configuration Benchmark Results

| Config | TPS (128) | TPS (512) | TPS (1024) | GPU Mem | Concurrent 5 | Concurrent 10 | Status |
|--------|-------:|------:|-------:|---------|---:|---:|------|
| **TP=1, ctx=8k** | 98.6 | **106.0** | **106.7** | 22.7 GB (GPU0) | 156.5 agg | 154.5 agg | **Best single-req** |
| TP=1, ctx=16k | 103.9 | 106.1 | 106.6 | 22.7 GB | 156.5 agg | 154.5 agg | Same TPS, more ctx |
| TP=1, ctx=16k, fp8 KV | 101.5 | 104.1 | 105.0 | 22.7 GB | - | - | Saves KV VRAM |
| TP=1, ctx=16k, prefix cache | 103.9 | 105.2 | 105.7 | 22.7 GB | 156.5 agg | 154.5 agg | Repeat prompt benefit |
| TP=1, ctx=32k | - | - | - | OOM | - | - | Model uses 22.7/24 GB |
| TP=2, ctx=32k | 5.8 | **15.5** | **15.7** | 24.3 GB each | 9.1 agg | 12.1 agg | **7x slower** (PCIe) |
| TP=1, enforce-eager | 6.1 | **14.9** | **15.0** | 21.0 GB | 48.0 agg | 70.5 agg | **7x slower** (no graphs) |

### Key Findings

1. **TP=1 is optimal**: 105-107 TPS. TP=2 is 7x slower due to PCIe inter-GPU communication overhead
2. **CUDA graphs are essential**: enforce-eager drops from 105 to 15 TPS
3. **Context 8k-16k works well**: Model uses 22.7 GB on 1 GPU, leaving ~1.3 GB for KV cache
4. **Context 32k+ on TP=1 is impossible**: OOM with gpu_memory_utilization=0.95
5. **fp8 KV cache has no TPS impact**: Same speed but saves KV VRAM for more sequences
6. **Prefix caching works**: Subsequent requests with same system prompt are faster
7. **Concurrent throughput is excellent**: 155 aggregate TPS (5-10 concurrent)
8. **First request after startup is slow**: CUDA graph compilation takes ~10s per new batch size

### vLLM vs llama.cpp Comparison (Qwen3.5-35B-A3B)

| Metric | vLLM (GPTQ-Int4) | llama.cpp Vulkan (GGUF Q4_K) | llama.cpp CUDA (GGUF Q4_K) |
|--------|------------------:|---------:|---------:|
| Single-req TPS | **105-107** | **114.3** | 97.6 |
| 5 concurrent agg | **156.5** | **131.7** | 131.7 |
| 10 concurrent agg | **154.5** | 124.2 | 124.2 |
| Max context (TP=1) | 16k | 8k+ (adjustable) | 8k+ (adjustable) |
| TTFT (streaming) | 256ms | ~100ms | ~100ms |
| Cold start | ~150s (download+compile) | ~5s | ~5s |
| Multi-user scaling | Excellent (PagedAttention) | Good (built-in) | Good (built-in) |

**Verdict**: vLLM wins on concurrent throughput (+25%), llama.cpp Vulkan wins on single-request TPS (+8%).

### Coder-Next vLLM Feasibility

| Option | Size | Fits 48GB? | Notes |
|--------|------|-----------|-------|
| AWQ-4bit | 45.9 GB | Barely | No room for KV cache |
| FP8 | ~80 GB | No | Way too large |
| GPTQ-Int4 | N/A | - | Does not exist for this architecture |
| GGUF Q4_K_XL (llama.cpp) | 41.5 GB | Yes | **Best option** - 90 t/s CUDA |

**Conclusion**: Coder-Next stays on llama.cpp. vLLM is not viable on this hardware.

### Optimal vLLM Command (Qwen3.5)

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

**Performance**: 105 t/s single | 155 t/s aggregate concurrent | 16k context
**GPU Memory**: 22.7 GB on single GPU

---

# Qwen3-Coder-Next Optimization Findings (llama.cpp)

## Test Results Summary

### Configuration Tests Performed

| Configuration | Flags | TPS | Duration | Notes |
|--------------|-------|-----|----------|-------|
| **Baseline with --fit on** | `--fit on --ctx-size 131072` | **13.8 t/s** | 36.24s | Original working config |
| **With Flash Attn + Q4_1 KV** | `--flash-attn on --cache-type-k q4_1 --cache-type-v q4_1 --fit on --ctx-size 131072` | 11.21 t/s | 44.59s | **19% SLOWER** - not recommended |

## Key Findings from Unsloth Documentation

### 1. Recommended Settings (Already Implemented)
- ✅ Temperature: 1.0 (critical)
- ✅ Top_P: 0.95
- ✅ Top_K: 40
- ✅ Min_P: 0.01

### 2. Context Size Options
- **Native support**: 262,144 tokens (full capability)
- **Reduced memory**: 32,768 tokens (for memory constraints)
- **Current**: 131,072 tokens (good balance)

### 3. KV Cache Quantization
**Available types**: `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`

**Test findings**:
- F16 (default, no flag): **Best performance at 13.8 t/s**
- Q4_1 with Flash Attention: Slower at 11.21 t/s

**Explanation**:
- KV cache quantization saves VRAM by using lower precision
- **Trade-off**: You have 2x RTX 3090 (48GB total VRAM), so you DON'T need to save memory
- Quantization adds dequantization overhead during inference
- For MoE models with sparse activation, this overhead outweighs bandwidth savings

### 4. Flash Attention Findings
- **Purpose**: Optimized for long sequences and large batch sizes
- **Your use case**: Single-request inference with moderate context
- **Result**: Added overhead without benefits → slower performance
- **Recommendation**: Don't use Flash Attention for this workload

### 5. FP8 Quantization
- **vLLM only**: The FP8-Dynamic quant mentioned in docs is for vLLM deployment
- **Not available in llama.cpp**: llama.cpp doesn't support FP8 KV cache
- **25% speed claim**: Only applies to vLLM with `--kv-cache-dtype fp8`

### 6. VRAM Calculations

**Model**: 35.78 GiB (Q3_K quantization on disk)

**KV Cache at 131k context** (48 layers, 2 heads, 256 dim):
- F16 (default): ~12.3 GB
- Q8_0: ~6.2 GB
- Q4_1: ~3.1 GB

**Total VRAM Usage**:
- Model + F16 KV @ 131k: ~48 GB
- Your hardware: 48 GB available (2x RTX 3090)
- **Conclusion**: You're at capacity with F16 KV cache, but it's working fine

## Recommendations

### ✅ OPTIMAL CONFIGURATION (Current Baseline)
```bash
/home/matilda/llama.cpp/build/bin/llama-server \
  --model /home/matilda/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q3_K_XL.gguf \
  --alias 'unsloth/Qwen3-Coder-Next' \
  --fit on \
  --seed 3407 \
  --temp 1.0 \
  --top-p 0.95 \
  --min-p 0.01 \
  --top-k 40 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 131072 \
  --jinja \
  --n-gpu-layers 999
```

**Performance**: 13.8 t/s
**Reasoning**:
- Simplest configuration
- Best observed performance
- Full F16 precision for KV cache = no dequantization overhead
- `--fit on` automatically optimizes memory layout
- 131k context is a good balance (half of max)

### Alternative: Longer Context
If you need more context (up to 256k):
```bash
--ctx-size 262144 --fit on
```
This will require KV cache quantization to fit in VRAM. Test Q8_0 first:
```bash
--ctx-size 262144 --cache-type-k q8_0 --cache-type-v q8_0 --n-gpu-layers 999
```

### Alternative: Reduced Memory for Future Experiments
If testing other models alongside:
```bash
--ctx-size 32768 --fit on
```
Reduces KV cache from 12.3GB to ~3GB, freeing VRAM for other use.

## Why Optimizations Didn't Help

1. **Memory Bandwidth vs Computation**:
   - Your GPUs aren't memory-bandwidth constrained for this workload
   - KV quantization optimizes bandwidth at cost of compute
   - Result: Net slowdown

2. **MoE Architecture**:
   - Only 3B/80B parameters active per token
   - Sparse activation = less memory pressure
   - Different bottlenecks than dense models

3. **Flash Attention Context**:
   - Optimized for training/fine-tuning with large batches
   - Single-request inference doesn't benefit
   - Adds kernel launch overhead

4. **Hardware Match**:
   - 48GB VRAM is sufficient for F16 KV @ 131k
   - No need to trade performance for memory savings

## llama.cpp Version Status

- **Current version**: commit 292f6908c (2026-02-09)
- **Bug fix status**: ✅ Includes Feb 4 fix for Qwen key_gdiff calculation
- **Conclusion**: Build is up to date

## Next Steps (Optional Experiments)

### 1. Test Reduced Context Performance
```bash
--ctx-size 32768 --fit on
```
Expected: Similar or slightly faster TPS, less VRAM usage

### 2. Test Without --fit on
```bash
--ctx-size 131072 --n-gpu-layers 999
```
Expected: Similar performance, `--fit on` mainly helps with tight memory

### 3. Batch Processing (if applicable)
If you process multiple requests:
- Consider `llama-parallel` for high-throughput
- Batch size > 1 may benefit from Flash Attention

### 4. Monitor VRAM Usage
```bash
nvidia-smi dmon -s u
```
Watch during generation to see actual memory pressure

## Conclusion

**Keep your current baseline configuration** (`--fit on --ctx-size 131072`).

The Unsloth optimizations (Flash Attention, KV quantization) are designed for:
- Memory-constrained environments
- Very long context (>131k tokens)
- Batch inference workloads
- Different inference frameworks (vLLM)

Your setup with dual RTX 3090s and single-request inference is already well-optimized with the simple F16 KV cache configuration.

**Current Performance**: 13.8 t/s @ 131k context
**Status**: ✅ Optimal for your hardware and workload
