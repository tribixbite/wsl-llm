#!/usr/bin/env bash
# Vision (multimodal) serving check for Qwen3.8-27B on the 16 GB Legion.
#
# The model is natively multimodal (27-layer ViT, out_hidden 5120) and Unsloth
# ships mmproj-F16.gguf (0.86 GiB). llama-server exposes it through the ordinary
# OpenAI /v1/chat/completions endpoint using image_url content parts.
#
# VRAM: weights 12.24 + mmproj 0.86 + KV. That does NOT leave room for the MTP
# draft head as well, so vision and speculative decoding are mutually exclusive
# on this card — pick one. Context is trimmed to 16k here to keep ~1.4 GiB slack.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
MMPROJ="${MMPROJ:-$HOME/models/Qwen3.8-27B-GGUF/mmproj-F16.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-16384}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/bench/results/legion-qwen38/vision}"
mkdir -p "$OUT_DIR"

stop_server() {
  pkill -9 -f "$LLAMA_BIN" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "$LLAMA_BIN" >/dev/null || break; sleep 0.5; done
  sleep 2
}
trap stop_server EXIT
stop_server

echo "=== starting vision-enabled llama-server ==="
setsid "$LLAMA_BIN" -m "$MODEL" --mmproj "$MMPROJ" \
    -ngl 99 -fa on -c "$CTX" -ctk q8_0 -ctv q8_0 --parallel 1 \
    --host 127.0.0.1 --port "$PORT" --jinja --no-webui \
    > "$OUT_DIR/server.log" 2>&1 < /dev/null &

for _ in $(seq 1 180); do
  curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || {
  echo "!! server failed to start"; grep -viE "unused tensor" "$OUT_DIR/server.log" | tail -15; exit 1; }

echo "server up — VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
grep -iE "clip|vision|mmproj|projector" "$OUT_DIR/server.log" | grep -viE "unused tensor" | head -8

python3 "$REPO_DIR/bench/legion/vision_probe.py" --url "http://127.0.0.1:$PORT" \
    --out "$OUT_DIR/results.json" "$@" 2>&1 | tee "$OUT_DIR/probe.txt"

echo "peak VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
stop_server
