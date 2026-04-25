#!/bin/bash
set -euo pipefail

# Download Qwen GGUF model from HuggingFace.
# Default: Qwen 3.6 35B-A3B UD-Q4_K_XL (~21 GB, current production model).
# Pass an alias to override: ./download-model.sh coder-next

MODEL_DIR="$HOME/models"
MODEL_FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
HF_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
HF_FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_SIZE="22.4 GiB"

# Allow overriding for different models
if [ -n "${1:-}" ]; then
    case "$1" in
        qwen36-27b)
            HF_REPO="unsloth/Qwen3.6-27B-GGUF"
            HF_FILE="Qwen3.6-27B-UD-Q4_K_XL.gguf"
            MODEL_FILE="Qwen3.6-27B-UD-Q4_K_XL.gguf"
            MODEL_SIZE="17.6 GiB"
            ;;
        qwen36-27b-iq4xs)
            HF_REPO="unsloth/Qwen3.6-27B-GGUF"
            HF_FILE="Qwen3.6-27B-IQ4_XS.gguf"
            MODEL_FILE="Qwen3.6-27B-IQ4_XS.gguf"
            MODEL_SIZE="15.4 GiB"
            ;;
        coder-next)
            HF_REPO="unsloth/Qwen3-Coder-Next-GGUF"
            HF_FILE="Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
            MODEL_FILE="Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
            MODEL_SIZE="41.5 GiB"
            ;;
        qwen35)
            HF_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"
            HF_FILE="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
            MODEL_FILE="Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"
            MODEL_SIZE="20.7 GiB"
            ;;
        *)
            echo "Unknown model: $1"
            echo "Available: (default) qwen3.6, qwen36-27b, qwen36-27b-iq4xs, coder-next, qwen35"
            exit 1
            ;;
    esac
fi

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "Model already exists: $MODEL_DIR/$MODEL_FILE"
    ls -lh "$MODEL_DIR/$MODEL_FILE"
    exit 0
fi

echo "Model: $HF_REPO / $HF_FILE"
echo "Size:  $MODEL_SIZE"
echo "Dest:  $MODEL_DIR/$MODEL_FILE"
echo ""
read -rp "Download now? [Y/n] " answer
if [[ "${answer:-Y}" =~ ^[Nn] ]]; then
    echo "Skipped."
    exit 0
fi

mkdir -p "$MODEL_DIR"

# Prefer the new `hf` CLI (huggingface_hub >= 0.29) over deprecated huggingface-cli.
if command -v hf &>/dev/null; then
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$HF_REPO" "$HF_FILE" --local-dir "$MODEL_DIR"
elif command -v huggingface-cli &>/dev/null; then
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir "$MODEL_DIR"
else
    echo "Neither 'hf' nor 'huggingface-cli' found. Install with:"
    echo "  uv tool install --with hf-transfer huggingface_hub"
    exit 1
fi

if [ -f "$MODEL_DIR/$HF_FILE" ] && [ "$HF_FILE" != "$MODEL_FILE" ]; then
    mv "$MODEL_DIR/$HF_FILE" "$MODEL_DIR/$MODEL_FILE"
fi

echo ""
echo "Downloaded: $MODEL_DIR/$MODEL_FILE"
ls -lh "$MODEL_DIR/$MODEL_FILE"
