#!/bin/bash
set -uo pipefail
# Qwen3.8-27B with VISION via llama.cpp + the Unsloth mmproj projector.
#
# Verified: correctly read "TEST-7429" and described both shapes in a test image.
#
# Why this config for the always-on endpoint:
#   - vision works (the syv-ai vLLM stack launches language_model_only=true, no vision)
#   - Q3_K_XL (13 GB) + mmproj (0.93 GB) fits ONE card with headroom, so it can sit on
#     GPU 1 and leave GPU 0 free for benchmarks/experiments
#   - MTP speculative decoding is on (roughly doubles decode)
#
# ⚠️ `-np 1` (one slot) is deliberate. WSL2 has no OOM guardrail: more slots push VRAM
# over the line and WDDM silently evicts weights to system RAM — /health keeps
# answering while decode collapses ~700x. See docs/QWEN38_27B_LEGION_BENCHMARKS.md §3a.
# Validate any change with a TIMED GENERATION, never just /health.

BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/qwen38/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
MMPROJ="${MMPROJ:-$HOME/models/qwen38/mmproj-F16.gguf}"
PORT="${PORT:-8085}"
CTX="${CTX:-16384}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"

[ -x "$BIN" ]     || { echo "ERROR: llama-server not built at $BIN"; exit 1; }
[ -f "$MODEL" ]   || { echo "ERROR: model not found: $MODEL"; exit 1; }
[ -f "$MMPROJ" ]  || { echo "ERROR: vision projector not found: $MMPROJ"; exit 1; }

exec env CUDA_VISIBLE_DEVICES="$GPU" "$BIN" \
    -m "$MODEL" --mmproj "$MMPROJ" \
    --alias qwen3.8-27b-vision \
    -c "$CTX" -np 1 -ngl 999 -fa on \
    --spec-type draft-mtp \
    --temp 0.7 --top-p 0.8 --top-k 20 \
    --host 0.0.0.0 --port "$PORT" --jinja
