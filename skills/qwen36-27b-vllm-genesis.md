---
name: Qwen 3.6 27B vLLM+Genesis serving recipe
description: Production-validated config to serve Qwen3.6-27B Dense at 67 t/s code / 46 t/s prose via vLLM 0.17 + Sandermage Genesis patches + MTP n=3 + fp8_e5m2 KV on a single RTX 3090
---

# Qwen 3.6 27B Dense — vLLM + Genesis serving recipe

## When to use this stack
- Coding-quality matters (Qwen claims SWE-bench Verified 77.2, LiveCodeBench v6 83.9 — best dense model under 30B)
- You want speculative decoding (MTP n=3 — ~2× speedup over no-spec)
- Single RTX 3090 (24 GB). Won't fit FP8 27B without INT4 quant.

## Speed expectations on RTX 3090

| Workload | Decode TPS |
|----------|-----------:|
| Highly-repetitive (JSON sequences, generated tables) | 70 |
| Normal code (BST, doubly linked list, etc.) | 55-67 |
| Boilerplate code (CRUD APIs) | 54 |
| Free-form prose (essays) | 45-50 |

Bandwidth math: 14 GB/token at INT4 / 936 GB/s = 67 t/s no-spec ceiling. MTP gives modest multiplier on top.

## Setup (one-time)

### 1. Download model
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 ~/bench_env/bin/hf download \
    Lorbus/Qwen3.6-27B-int4-AutoRound \
    --local-dir ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound

# Patch tokenizer config — vLLM doesn't recognize Genesis "TokenizersBackend"
python3 -c "
import json
p = '$HOME/models/Lorbus-Qwen3.6-27B-int4-AutoRound/tokenizer_config.json'
d = json.load(open(p))
if d.get('tokenizer_class') == 'TokenizersBackend':
    d['tokenizer_class'] = 'Qwen2TokenizerFast'
    json.dump(d, open(p, 'w'), indent=2)
    print('patched')
"
```

### 2. Install Sandermage Genesis patches
```bash
# Clone noonghunna recipe (pulls Genesis as submodule via setup.sh)
git clone https://github.com/noonghunna/qwen36-27b-single-3090.git ~/qwen36-noon
cd ~/qwen36-noon && SKIP_MODEL=1 bash scripts/setup.sh

# Mount Genesis package into vLLM site-packages
VLLM_PKG=~/bench_env/lib/python3.10/site-packages/vllm
[ -d "$VLLM_PKG/_genesis" ] && mv "$VLLM_PKG/_genesis" "$VLLM_PKG/_genesis.bak.$(date +%s)"
cp -r ~/qwen36-noon/patches/genesis/vllm/_genesis "$VLLM_PKG/_genesis"

# Install Genesis as vLLM plugin entry-point (auto-loads in API server + engine + workers)
cd ~/qwen36-noon/patches/genesis/genesis_vllm_plugin
~/bench_env/bin/pip install --no-deps -e .

# Verify entry-point registration
~/bench_env/bin/python -c "
from importlib.metadata import entry_points
print([e.name for e in entry_points(group='vllm.general_plugins')])
"
# → ['genesis_v7', 'lora_filesystem_resolver', 'lora_hf_hub_resolver']
```

## Launch command (production)

```bash
PATH=$HOME/bench_env/bin:$PATH \
GENESIS_ENABLE_P75_SUFFIX_DECODING=1 \
GENESIS_ENABLE_P77_ADAPTIVE_NGRAM_K=1 \
GENESIS_ENABLE_P81_FP8_BLOCK_SCALED_M_LE_8=1 \
GENESIS_ENABLE_P79B_ASYNC_PROPOSER_SYNC=1 \
GENESIS_ENABLE_P79C_STALE_SPEC_TOKEN_CLEANUP=1 \
GENESIS_ENABLE_P40=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
VLLM_FLOAT32_MATMUL_PRECISION=high \
VLLM_USE_FLASHINFER_SAMPLER=1 \
OMP_NUM_THREADS=1 \
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
VLLM_MARLIN_USE_ATOMIC_ADD=1 \
NCCL_CUMEM_ENABLE=0 NCCL_P2P_DISABLE=1 \
VLLM_NO_USAGE_STATS=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0 \
~/bench_env/bin/python -m vllm.entrypoints.cli.main serve \
    ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --quantization auto_round --dtype float16 \
    --tensor-parallel-size 1 --max-model-len 32000 \
    --gpu-memory-utilization 0.93 --max-num-seqs 1 \
    --max-num-batched-tokens 2048 \
    --kv-cache-dtype fp8_e5m2 \
    --trust-remote-code \
    --enable-prefix-caching --enable-chunked-prefill \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --port 8081
```

## Key flags explained

| Flag | Why |
|------|-----|
| `--quantization auto_round` | Lorbus uses GPTQ-style INT4 with AutoRound calibration — Marlin kernel path |
| `--kv-cache-dtype fp8_e5m2` | NOT fp8_e4m3 — sidesteps tool-call cudagraph cascade bug (vllm#40880) |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` | MTP head trained for n=3; n=2 worse, n=4 crashes (CUDA illegal memory) |
| `--gpu-memory-utilization 0.93` | 0.95 OOMs on a 24 GB card with 22.7 GB free at startup; 0.92 fights KV cache |
| `--max-num-seqs 1 --max-num-batched-tokens 2048` | Single-stream config; minimum batch ensures cudagraph capture sizes match decode shape |

## Sampling params (per-request)

```json
{
  "model": "qwen3.6-27b",
  "messages": [{"role": "user", "content": "..."}],
  "max_tokens": 1000,
  "temperature": 0.6,
  "top_p": 0.95,
  "top_k": 20,
  "stream": true,
  "stream_options": {"include_usage": true},
  "chat_template_kwargs": {"enable_thinking": false}
}
```

**`enable_thinking: false`** is critical: without it, the model emits long reasoning blocks that crater MTP acceptance and double wall time.

## Measurement methodology

To compare against published numbers (e.g. noonghunna's "84 code TPS"):
- Use streaming with `stream_options:{include_usage:true}`
- Compute `decode_TPS = completion_tokens / (wall_time - TTFT)` (excludes prefill)
- 3 warmups + 5 measured runs averaged

The simple `tokens / wall_time` measurement underestimates by 5-10% due to prefill.

## Why this isn't 80 t/s

27B Dense has all-active params; INT4 ≈ 14 GB/token. RTX 3090 = 936 GB/s memory bandwidth → **67 t/s theoretical no-spec ceiling**. We hit 67 on real code, 70 on max-acceptance JSON sequences — at the wall.

To break 67 you'd need either:
- Custom Qwen3.6-27B EAGLE-3 head (not yet published; train via [SpecForge](https://github.com/sgl-project/SpecForge), needs 2× GPU + ~24h)
- Hardware upgrade (RTX 4090 = 1008 GB/s, +8% bandwidth → ~72 no-spec)
- Switch to 35B-A3B MoE (3B active params/token = much higher ceiling, our existing 102 t/s daily driver)

## See also

- [docs/QWEN36_27B_BENCHMARKS.md §11](../docs/QWEN36_27B_BENCHMARKS.md#11-sandermage-genesis-vllm-patches--correct-measurement-methodology) — full Genesis bench writeup
- [bench/results/qwen36-27b/sandermage-genesis/](../bench/results/qwen36-27b/sandermage-genesis/) — raw bench data
- [Sandermage/genesis-vllm-patches](https://github.com/Sandermage/genesis-vllm-patches) — upstream patches
- [noonghunna/qwen36-27b-single-3090](https://github.com/noonghunna/qwen36-27b-single-3090) — recipe source
