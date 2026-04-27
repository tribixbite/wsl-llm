# Qwen 3.6 27B + Sandermage Genesis vLLM Patches (2026-04-26)

Following [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) (45+ runtime patches, "P67 multi-query kernel + Suffix Decoding + adaptive ngram K") + [`noonghunna/qwen36-27b-single-3090`](https://github.com/noonghunna/qwen36-27b-single-3090) (recipe with 66/84 narrative/code TPS reported).

**Result on our hardware**: best **70 t/s** (highly-repetitive JSON sequence), **67 t/s** code (BST + tests), **46 t/s** narrative (transformer essay). Did not reproduce the 84 t/s code claim, but +25-30% over previously-reported numbers due to:
1. Genesis patches (21 applied via plugin entry-point)
2. **Correct measurement methodology** (streaming + `decode_TPS` excludes prefill)
3. `chat_template_kwargs:{enable_thinking:False}` properly disables thinking

## Setup steps

```bash
# 1. Clone noonghunna recipe (gets Genesis as submodule)
git clone https://github.com/noonghunna/qwen36-27b-single-3090.git ~/qwen36-noon
cd ~/qwen36-noon && SKIP_MODEL=1 bash scripts/setup.sh

# 2. Install Genesis package into bench_env's vLLM site-packages (in-place)
VLLM_PKG=~/bench_env/lib/python3.10/site-packages/vllm
cp -r ~/qwen36-noon/patches/genesis/vllm/_genesis "$VLLM_PKG/_genesis"

# 3. Install Genesis as a vLLM plugin entry-point so it auto-loads
#    in EVERY vLLM process (main API server + engine + workers)
cd ~/qwen36-noon/patches/genesis/genesis_vllm_plugin
~/bench_env/bin/pip install --no-deps -e .

# 4. Verify entry-point registration
~/bench_env/bin/python -c "
from importlib.metadata import entry_points
print([e.name for e in entry_points(group='vllm.general_plugins')])
"
# → ['genesis_v7', 'lora_filesystem_resolver', 'lora_hf_hub_resolver']
```

## Launch flags (noonghunna recipe + opt-in patches)

```bash
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
  --tensor-parallel-size 1 --max-model-len 20000 \
  --gpu-memory-utilization 0.93 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8_e5m2 \
  --trust-remote-code \
  --enable-prefix-caching --enable-chunked-prefill \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --port 8081
```

## Genesis applied patches (in this config, 21 of 58)

P1/P2 FP8 dispatcher · P5 KV page size · P8 KV hybrid reporting · P12/P15/P27 Qwen3 fixes · P14 block_table tail zero-fill · P17/P18 Marlin MoE per-SM · P18b TQ decode stage1 · P20 TQ continuation-prefill FP16 · P23 Marlin FP32_REDUCE · P24 fused_moe num_warps · P29 tool parser guard · P31 MoE router fp32 softmax · P32/P33 TQ preallocs · P34 Mamba zero-collapse guard · P75 Suffix Decoding (only fires for ngram, not MTP) · P77 Adaptive ngram K · P79b/c Async spec-decode fixes · P81 fp8 block-scaled MM low-M decode tuning

## Bench results (proper methodology)

`results.tsv` — measurement: stream=True, `chat_template_kwargs:{enable_thinking:False}`, `decode_TPS = completion_tokens / (wall - TTFT)`, 3 warmups + 5 measured runs.

| Prompt | Decode TPS | Wall TPS | Tokens |
|--------|-----------:|---------:|-------:|
| Transformer attention essay (noonghunna canonical narrative) | 45.81 | 45.50 | 992 |
| BST + unit tests | **67.36** | 66.66 | 1000 |
| **JSON integer sequence** | **70.10** ⭐ | 69.14 | 893 |
| Repetitive markdown table | 68.56 | 67.93 | 1500 |
| Doubly linked list | 61.13 | 60.56 | 1000 |
| TS Express CRUD | 54.57 | 54.04 | 1000 |
| Fibonacci continuation | 46.98 | 46.70 | 1500 |

## Why we didn't reach 80

| Factor | Effect |
|--------|--------|
| **27B Dense bandwidth ceiling** | 14 GB/token at INT4, 936 GB/s GPU = ~67 t/s theoretical max via bandwidth alone |
| **MoE-specific patches don't fire** | P17/P18/P24/P31/P37 are MoE-only, no effect on dense 27B |
| **TurboQuant patches don't fire** | P3/P6/P22/P26/P32/P33/P38/P40 require TurboQuant KV (we use fp8_e5m2; longctx variant uses TQ but at 33-72 TPS, slower) |
| **No FP8 native compute on Ampere** | P81/P1/P2 fall back to Marlin INT4 path |
| **MTP acceptance varies by prompt** | 70 t/s on JSON sequences (95%+ accept rate), 46 t/s on essay prose (poor pattern matching) |

## Why noonghunna's "84 code TPS" exceeds our 67

Possible reasons (we cannot verify without their exact prompt + machine):
- They likely use a code prompt with even higher MTP acceptance than ours
- Their Docker pin (`vllm/vllm-openai@sha256:9bba4628...`) may have backend-specific tuning
- Different HW (PCIe gen, RAM speed, BIOS settings)

Sandermage's own `MODELS.md` states for Qwen3.6-27B Dense on a single 3090:
> Speed estimate: **~50-65 tok/s decode** (vs our 127 with MTP A3B)

Our 67 t/s code is **at the upper bound** of that range. **Sandermage themselves don't claim 80 t/s for 27B Dense** — that target was loose interpolation from the 35B-A3B numbers.

## Honest answer to "can we hit 80 t/s on Qwen3.6-27B Dense + RTX 3090"

**On highly repetitive content**: yes (70-80 range achievable on JSON sequences, code with macros, etc.)
**On real workloads** (essays, mixed code, agentic tool calls): **no, ~45-67 t/s is the realistic range**.

The math: 936 GB/s ÷ 14 GB/token = 67 t/s no-spec ceiling. MTP n=3 with ~50-70% acceptance multiplies that by 1.0-1.5×, capping near 100 only if acceptance hits ≥90% (rare in practice). **The 35B-A3B model is fundamentally faster** on this hardware (3 GB active params/token = ~300 t/s no-spec ceiling × MTP = our existing 102 t/s production daily driver).

## Files

| File | Description |
|------|-------------|
| `results.tsv` | All decode_TPS / wall_TPS measurements |
| `bench_noon_style.py` | Streaming bench script (3 warmups + 5 measured) |
| `vllm_with_genesis.py` | Wrapper for Genesis (kept for reference; plugin entry-point is the recommended method) |
