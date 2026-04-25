#!/bin/bash
set -euo pipefail

# Build ik_llama.cpp (primary) and llama.cpp (upstream) with CUDA support

CMAKE="${HOME}/.local/bin/cmake"
command -v "$CMAKE" &>/dev/null || CMAKE="cmake"

CUDA_ARCH="${CUDA_ARCH:-86}"
CUDA_COMPILER="/usr/local/cuda-12.6/bin/nvcc"
JOBS="${JOBS:-$(nproc)}"

if [ ! -f "$CUDA_COMPILER" ]; then
    echo "ERROR: CUDA compiler not found at $CUDA_COMPILER"
    echo "Install CUDA toolkit first: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "============================================"
echo "  Building ik_llama.cpp (PRIMARY engine)"
echo "============================================"
if [ ! -d "$HOME/ik_llama.cpp" ]; then
    echo "Cloning ik_llama.cpp..."
    git clone https://github.com/ikawrakow/ik_llama.cpp "$HOME/ik_llama.cpp"
else
    echo "Updating ik_llama.cpp..."
    git -C "$HOME/ik_llama.cpp" pull --ff-only || echo "  (pull skipped — local changes)"
fi

cd "$HOME/ik_llama.cpp"
"$CMAKE" -B build \
    -DGGML_NATIVE=ON \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER"
"$CMAKE" --build build -j"$JOBS"
echo "ik_llama.cpp built successfully."
echo ""

echo "============================================"
echo "  Building llama.cpp (upstream)"
echo "============================================"
if [ ! -d "$HOME/llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp "$HOME/llama.cpp"
else
    echo "Updating llama.cpp..."
    git -C "$HOME/llama.cpp" pull --ff-only || echo "  (pull skipped — local changes)"
fi

cd "$HOME/llama.cpp"
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build build -j"$JOBS"
echo "llama.cpp built successfully."
echo ""

echo "Both engines built. Binaries:"
echo "  ik_llama.cpp: $HOME/ik_llama.cpp/build/bin/llama-server"
echo "  llama.cpp:    $HOME/llama.cpp/build/bin/llama-server"
