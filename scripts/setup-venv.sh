#!/bin/bash
set -euo pipefail

# Set up Python virtual environment with vLLM, PyTorch, and LiteLLM

VENV_DIR="${1:-$HOME/bench_env}"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Virtual environment already exists: $VENV_DIR"
    echo "To recreate, delete it first: rm -rf $VENV_DIR"
    echo ""
    echo "Installed packages:"
    "$VENV_DIR/bin/pip" list 2>/dev/null | grep -E "torch|vllm|litellm|llama" || true
    exit 0
fi

echo "Creating Python venv: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "Installing vLLM (nightly for Qwen3.5 support)..."
pip install vllm --extra-index-url https://wheels.vllm.ai/nightly

echo ""
echo "Installing LiteLLM proxy..."
pip install "litellm[proxy]" prisma

echo ""
echo "Installing benchmark utilities..."
pip install requests aiohttp

echo ""
echo "Generating Prisma client..."
cd "$VENV_DIR/lib/python3.*/site-packages/litellm/proxy/db/" 2>/dev/null || true
prisma generate 2>/dev/null || echo "  Prisma generate skipped (will run on first litellm start)"

echo ""
echo "Virtual environment ready: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
