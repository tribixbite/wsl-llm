#!/bin/bash
set -euo pipefail

# Build Madreag's TurboQuant CUDA fork — the primary inference engine.
#
# Why this fork: Madreag (https://github.com/Madreag/turbo3-cuda) is a fork of
# TheTom's base TurboQuant llama.cpp implementation with CUDA kernel
# optimizations specifically validated on RTX 3090, 3090 Ti, 4090M and 5090.
# It enables turbo3/turbo4 KV cache compression (3-5× smaller KV at near-q8_0
# quality), restoring full 262k context on a single 24 GB GPU at ~100 t/s decode.
#
# The compiled binary is copied to ~/llama-cpp-turboquant/llama-server, which
# is the path referenced from ~/llama-server.conf via LLAMA_BIN.

CMAKE="${HOME}/.local/bin/cmake"
command -v "$CMAKE" &>/dev/null || CMAKE="cmake"

CUDA_ARCH="${CUDA_ARCH:-86}"  # 86 = RTX 3090/3090Ti/4090M (sm_86 = Ampere)
CUDA_COMPILER="/usr/local/cuda-12.6/bin/nvcc"
JOBS="${JOBS:-$(nproc)}"
SRC_DIR="${HOME}/llama-cpp-turboquant-src"
BIN_DIR="${HOME}/llama-cpp-turboquant"

if [ ! -f "$CUDA_COMPILER" ]; then
    echo "ERROR: CUDA compiler not found at $CUDA_COMPILER"
    echo "Install CUDA toolkit first: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "============================================"
echo "  Building Madreag/turbo3-cuda (PRIMARY engine)"
echo "============================================"
echo "  CUDA arch:    sm_$CUDA_ARCH"
echo "  Source:       $SRC_DIR"
echo "  Install bin:  $BIN_DIR/llama-server"
echo ""

if [ ! -d "$SRC_DIR" ]; then
    echo "Cloning Madreag/turbo3-cuda..."
    git clone https://github.com/Madreag/turbo3-cuda.git "$SRC_DIR"
else
    echo "Updating existing checkout..."
    git -C "$SRC_DIR" pull --ff-only || echo "  (pull skipped — local changes)"
fi

cd "$SRC_DIR"
"$CMAKE" -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release

"$CMAKE" --build build --target llama-server -j"$JOBS"

mkdir -p "$BIN_DIR"
install -m 0755 "$SRC_DIR/build/bin/llama-server" "$BIN_DIR/llama-server"

echo ""
echo "Built successfully:"
ls -lh "$BIN_DIR/llama-server"
echo ""
echo "Verify:"
"$BIN_DIR/llama-server" --version 2>&1 | head -3
echo ""
echo "Use --cache-type-k turbo3 --cache-type-v turbo3 -fa on for compressed KV."
