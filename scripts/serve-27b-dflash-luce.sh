#!/bin/bash
set -uo pipefail
# Qwen3.6-27B speculative decoding via the Luce DFlash fork (Luce-Org/lucebox-hub).
# Custom GatedDeltaNet tree CUDA kernels make DFlash spec-decode work on the
# Qwen3.6 hybrid target where the spiritbuun fork underperformed (18/45 t/s).
#
# Reference (their RESULTS.md / README, single RTX 3090):
#   Q4_K_M target:       HumanEval 129.52 t/s (3.43x), AL 8.31
#   UD-Q4_K_XL target:   HumanEval  78.16 t/s (2.24x), mean 69.19 t/s (1.98x)
#
# Usage:
#   ./scripts/serve-27b-dflash-luce.sh                 # UD-Q4_K_XL target (already present)
#   TARGET=~/models/Qwen3.6-27B-Q4_K_M.gguf ./scripts/serve-27b-dflash-luce.sh
#
# Prereqs: lucebox-hub built (cmake --build build --target dflash_server), draft GGUF downloaded.

SERVER="${SERVER:-$HOME/lucebox-hub/server/build/bin/dflash_server}"
[ -x "$SERVER" ] || SERVER="$HOME/lucebox-hub/server/build/dflash_server"
TARGET="${TARGET:-$HOME/models/Qwen3.6-27B-UD-Q4_K_XL.gguf}"
DRAFT="${DRAFT:-$HOME/models/lucebox-draft/dflash-draft-3.6-q4_k_m.gguf}"
PORT="${PORT:-18080}"
MAX_CTX="${MAX_CTX:-32768}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
FA_WINDOW="${FA_WINDOW:-2048}"
DDTREE_BUDGET="${DDTREE_BUDGET:-22}"

[ -x "$SERVER" ] || { echo "ERROR: dflash_server not built at $SERVER"; exit 1; }
[ -f "$TARGET" ] || { echo "ERROR: target GGUF not found: $TARGET"; exit 1; }
[ -f "$DRAFT" ]  || { echo "ERROR: draft GGUF not found: $DRAFT"; exit 1; }

echo "DFlash server: $SERVER"
echo "  target: $TARGET"
echo "  draft:  $DRAFT"
echo "  port=$PORT ctx=$MAX_CTX fa-window=$FA_WINDOW ddtree-budget=$DDTREE_BUDGET KV=TQ3_0"

exec env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    DFLASH27B_KV_TQ3=1 \
    PATH="/usr/local/cuda-12.6/bin:$PATH" \
    "$SERVER" "$TARGET" \
    --draft "$DRAFT" \
    --host 127.0.0.1 --port "$PORT" \
    --max-ctx "$MAX_CTX" --max-tokens "$MAX_TOKENS" \
    --fa-window "$FA_WINDOW" \
    --ddtree --ddtree-budget "$DDTREE_BUDGET" \
    --model-name qwen3.6-27b
