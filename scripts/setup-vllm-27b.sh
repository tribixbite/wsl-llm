#!/bin/bash
set -euo pipefail

# Set up vLLM nightly + Lorbus AutoRound INT4 + MTP n=3 + fp8 KV for Qwen 3.6 27B.
#
# Why this setup: native MTP (multi-token prediction) speculative decoding gives
# ~54 t/s on a single RTX 3090 — ~2.7× faster than llama.cpp-family backends for
# the dense 27B Dense Qwen3.6 model. See docs/QWEN36_27B_BENCHMARKS.md.

LORBUS_DIR="${LORBUS_DIR:-$HOME/models/Lorbus-Qwen3.6-27B-int4-AutoRound}"
VENV_DIR="${VENV_DIR:-$HOME/bench_env}"

if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Python venv not found at $VENV_DIR"
    echo "Run scripts/setup-venv.sh first to create it (vLLM nightly + PyTorch + huggingface_hub)."
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Verify vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "ERROR: vllm not importable in $VENV_DIR"
    exit 1
fi

# Install hf_transfer for fast HF downloads (idempotent)
pip install --quiet hf_transfer

# Download Lorbus AutoRound INT4 model (~14 GB) — the model whose MTP head is BF16
if [ ! -f "$LORBUS_DIR/config.json" ]; then
    echo "[setup-vllm-27b] Downloading Lorbus/Qwen3.6-27B-int4-AutoRound to $LORBUS_DIR..."
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
        Lorbus/Qwen3.6-27B-int4-AutoRound \
        --local-dir "$LORBUS_DIR"
else
    echo "[setup-vllm-27b] Lorbus model already present at $LORBUS_DIR"
fi

# Patch tokenizer config — vLLM doesn't recognize the Genesis "TokenizersBackend"
# class so we substitute the standard Qwen2TokenizerFast (the underlying tokenizer
# data is compatible).
TOK_CFG="$LORBUS_DIR/tokenizer_config.json"
if grep -q '"tokenizer_class": "TokenizersBackend"' "$TOK_CFG" 2>/dev/null; then
    echo "[setup-vllm-27b] Patching tokenizer_class to Qwen2TokenizerFast..."
    cp "$TOK_CFG" "$TOK_CFG.bak"
    python3 -c "
import json
p = '$TOK_CFG'
d = json.load(open(p))
d['tokenizer_class'] = 'Qwen2TokenizerFast'
json.dump(d, open(p, 'w'), indent=2)
"
elif grep -q '"tokenizer_class": "Qwen2TokenizerFast"' "$TOK_CFG" 2>/dev/null; then
    echo "[setup-vllm-27b] tokenizer_config already patched"
else
    echo "[setup-vllm-27b] WARNING: tokenizer_class not recognized in $TOK_CFG"
fi

# Print recommended launch command
cat <<EOF

[setup-vllm-27b] Setup complete.

To launch (pick one):

# At full 262k context — ~54 t/s, 21.3 GB VRAM
CUDA_VISIBLE_DEVICES=0 \\
  $VENV_DIR/bin/python -m vllm.entrypoints.cli.main serve \\
    "$LORBUS_DIR" \\
    --served-model-name qwen3.6-27b \\
    --quantization auto_round --dtype float16 \\
    --tensor-parallel-size 1 --max-model-len 262144 \\
    --gpu-memory-utilization 0.92 --max-num-seqs 1 \\
    --kv-cache-dtype fp8 \\
    --enable-prefix-caching --enable-chunked-prefill \\
    --port 8081 --trust-remote-code \\
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

# Without MTP (cleaner, ~25 t/s baseline)
CUDA_VISIBLE_DEVICES=0 \\
  $VENV_DIR/bin/python -m vllm.entrypoints.cli.main serve \\
    "$LORBUS_DIR" --served-model-name qwen3.6-27b \\
    --quantization auto_round --dtype float16 \\
    --max-model-len 262144 --gpu-memory-utilization 0.92 \\
    --max-num-seqs 1 --kv-cache-dtype fp8 \\
    --port 8081 --trust-remote-code

If GPU 0 doesn't have ~22.5 GB free, lower --gpu-memory-utilization or --max-model-len.
EOF
