#!/bin/bash
set -uo pipefail
# Production launcher for Qwen3.6-27B Dense via vLLM 0.17 + Sandermage Genesis patches.
# See skills/qwen36-27b-vllm-genesis.md for the full recipe.
#
# Speeds (decode_TPS, streaming, enable_thinking=false):
#   ~67 t/s code / ~46 t/s prose / 70 t/s peak on JSON sequences
#
# Usage:
#   ./scripts/serve-27b-vllm-genesis.sh
#   PORT=8082 ./scripts/serve-27b-vllm-genesis.sh
#   TENSOR_PARALLEL=2 ./scripts/serve-27b-vllm-genesis.sh   # if you want both 3090s
#
# Prereqs (one-time):
#   - ~/bench_env Python venv with vLLM 0.17.0rc1+ + flash-attn + auto-round
#   - ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound downloaded
#   - Genesis patches installed via:
#       git clone https://github.com/noonghunna/qwen36-27b-single-3090.git ~/qwen36-noon
#       cd ~/qwen36-noon && SKIP_MODEL=1 bash scripts/setup.sh
#       cp -r ~/qwen36-noon/patches/genesis/vllm/_genesis ~/bench_env/lib/python3.10/site-packages/vllm/
#       cd ~/qwen36-noon/patches/genesis/genesis_vllm_plugin && ~/bench_env/bin/pip install --no-deps -e .

PORT="${PORT:-8081}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lorbus-Qwen3.6-27B-int4-AutoRound}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.93}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
CUDA_VISIBLE_DEVICES_VAR="${CUDA_VISIBLE_DEVICES:-0}"

# Verify prereqs
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: Model not found at $MODEL_DIR"
    echo "  Download with: HF_HUB_ENABLE_HF_TRANSFER=1 ~/bench_env/bin/hf download Lorbus/Qwen3.6-27B-int4-AutoRound --local-dir $MODEL_DIR"
    exit 1
fi

if ! ~/bench_env/bin/python -c "from importlib.metadata import entry_points; assert 'genesis_v7' in [e.name for e in entry_points(group='vllm.general_plugins')]" 2>/dev/null; then
    echo "ERROR: Genesis plugin not registered. Install via:"
    echo "  cd ~/qwen36-noon/patches/genesis/genesis_vllm_plugin && ~/bench_env/bin/pip install --no-deps -e ."
    exit 1
fi

# Check tokenizer is patched
if grep -q '"tokenizer_class": "TokenizersBackend"' "$MODEL_DIR/tokenizer_config.json" 2>/dev/null; then
    echo "Patching tokenizer_class TokenizersBackend → Qwen2TokenizerFast..."
    cp "$MODEL_DIR/tokenizer_config.json" "$MODEL_DIR/tokenizer_config.json.bak"
    python3 -c "
import json
p = '$MODEL_DIR/tokenizer_config.json'
d = json.load(open(p))
d['tokenizer_class'] = 'Qwen2TokenizerFast'
json.dump(d, open(p, 'w'), indent=2)
"
fi

# Launch
exec env \
    PATH="$HOME/bench_env/bin:$PATH" \
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
    NCCL_CUMEM_ENABLE=0 \
    NCCL_P2P_DISABLE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VAR" \
    "$HOME/bench_env/bin/python" -m vllm.entrypoints.cli.main serve \
    "$MODEL_DIR" \
    --served-model-name qwen3.6-27b \
    --quantization auto_round --dtype float16 \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs 1 \
    --max-num-batched-tokens 2048 \
    --kv-cache-dtype fp8_e5m2 \
    --trust-remote-code \
    --enable-prefix-caching --enable-chunked-prefill \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --port "$PORT"
