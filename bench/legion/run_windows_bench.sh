#!/usr/bin/env bash
# Native-Windows llama.cpp benchmark, driven from WSL via interop.
#
# Purpose: measure the WSL2-vs-native-Windows delta on identical work. Windows
# also honours NVIDIA's "Prefer No Sysmem Fallback" setting, which WSL2 ignores
# (microsoft/WSL#11050), so a config that silently degrades under WSL2 should
# fail loudly here instead.
#
# Caveat recorded with the results: the Windows binaries are release b10659
# while the WSL build is from commit cb30059. Both are 2026-08-27 but they are
# not the identical commit, so treat small deltas as noise.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PS="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
OUT_DIR="${OUT_DIR:-$REPO_DIR/bench/results/legion-qwen38/windows}"
WIN_MODEL='C:\llm\models\Qwen3.8-27B-UD-Q3_K_XL.gguf'
WIN_MTP='C:\llm\models\MTP\mtp-Qwen3.8-27B-Q4_0.gguf'
mkdir -p "$OUT_DIR"

echo "=== GPU state (Windows side) ==="
timeout 60 "$PS" -NoProfile -Command \
  'nvidia-smi --query-gpu=name,compute_cap,memory.total,enforced.power.limit,power.max_limit --format=csv,noheader' \
  2>&1 | tr -d '\r'

echo
echo "=== llama-bench: pp512 / tg128, flash attention on ==="
timeout 1800 "$PS" -NoProfile -Command "
\$env:Path = 'C:\llm\bin;' + \$env:Path
C:\llm\bin\llama-bench.exe -m '$WIN_MODEL' -ngl 99 -fa 1 -p 512 -n 128 -r 3
" 2>&1 | tr -d '\r' | tee "$OUT_DIR/llama-bench.txt" | grep -E "pp512|tg128|model|error|CUDA"

echo
echo "=== llama-fit-params: what does Windows think fits? ==="
timeout 900 "$PS" -NoProfile -Command "
\$env:Path = 'C:\llm\bin;' + \$env:Path
C:\llm\bin\llama-fit-params.exe -m '$WIN_MODEL' 2>&1 | Select-Object -Last 25
" 2>&1 | tr -d '\r' | tee "$OUT_DIR/fit-params.txt" | tail -20

echo
echo "=== done -> $OUT_DIR ==="
