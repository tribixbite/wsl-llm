#!/usr/bin/env bash
# Start Qwen3.8-27B with speculative decoding AND vision behind an
# OpenAI-compatible endpoint (WSL2 / Linux). Windows equivalent:
# windows/start-qwen38.ps1
#
#   ./serve-qwen38.sh                 # MTP + vision, 16k ctx  (default)
#   ./serve-qwen38.sh --mode fast     # MTP only, 32k ctx, no images
#   ./serve-qwen38.sh --mode vision   # vision only, 16k ctx, no MTP
#   ./serve-qwen38.sh --port 9000 --ctx 8192 --no-thinking
#
# Everything below is measured on the RTX 5080 Laptop (16 GB, sm_120) —
# see docs/QWEN38_27B_LEGION_BENCHMARKS.md.
#
#   --parallel 1          MANDATORY. The default of 4 slots gives each slot its
#                         own DeltaNet recurrent state, pushing peak VRAM 13.4 ->
#                         15.9 GiB. WSL2 has no OOM guardrail: the driver evicts
#                         the weights to system RAM and decode collapses ~700x
#                         (39.8 -> 0.04 t/s) while /health still returns ok.
#   --spec-type draft-mtp Biggest speed lever: 1.89x overall, 2.14x on code.
#   --no-mmproj-offload   Keeps the 0.86 GiB vision projector on the CPU, which
#                         is the only way vision and MTP both fit in 16 GB.
#                         Costs ~3 s to encode an image the first time; llama.cpp
#                         then caches it (repeat queries ~0.2 s).
#   --reasoning-effort medium
#                         Thinking is worth ~2x on coding (58.8% vs 38.2% pass@2);
#                         the model's default 'xhigh' costs ~220 s/exercise.
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-27B-GGUF}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$MODEL_DIR/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
MTP="${MTP:-$MODEL_DIR/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj-F16.gguf}"

MODE=both
PORT=8080
BIND=127.0.0.1
KV=q8_0
ALIAS=qwen3.8-27b
APIKEY=
CTX=0
THINKING=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    --bind)        BIND="$2"; shift 2 ;;
    --kv)          KV="$2"; shift 2 ;;
    --alias)       ALIAS="$2"; shift 2 ;;
    --api-key)     APIKEY="$2"; shift 2 ;;
    --ctx)         CTX="$2";  shift 2 ;;
    --no-thinking) THINKING=0; shift ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -x "$LLAMA_BIN" ]] || { echo "missing llama-server: $LLAMA_BIN" >&2; exit 1; }
[[ -f "$MODEL" ]]     || { echo "missing model: $MODEL" >&2; exit 1; }
# Measured ceilings (16,303 MiB total, keep >=1 GiB slack):
#   both  32k q4_0=14806  48k=15254  64k=15645(tight)
#   long 128k q4_0=15731 (no MTP)  96k=15565  64k=14829
if [[ $CTX -eq 0 ]]; then
  case "$MODE" in fast) CTX=32768 ;; long) CTX=131072 ;; *) CTX=32768 ;; esac
fi
# >32k only fits with q4_0 KV. q4_0 on K is the quality-sensitive half
# (llama.cpp#21591); prefer q8_0 K wherever it fits.
# MTP + vision dominate the budget; past 16k the KV must be q4_0. Measured:
# both@32k q8_0 = 15,291 MiB, decode 88.6 -> 65.4 t/s; q4_0 = 14,806 and full speed.
[[ "$KV" == q8_0 && $CTX -gt 16384 ]] && KV=q4_0

ARGS=(-m "$MODEL" -ngl 99 -fa on -c "$CTX" -ctk "$KV" -ctv "$KV"
      --parallel 1 --host "$BIND" --port "$PORT"
      --alias "$ALIAS"          # else clients see the full .gguf path as the model id
      --jinja)
[[ -n "$APIKEY" ]] && ARGS+=(--api-key "$APIKEY")

case "$MODE" in
  both|fast) [[ -f "$MTP" ]] || { echo "missing MTP head: $MTP" >&2; exit 1; }
             ARGS+=(--spec-type draft-mtp -md "$MTP") ;;&
  both|vision|long) [[ -f "$MMPROJ" ]] || { echo "missing projector: $MMPROJ" >&2; exit 1; }
             ARGS+=(--mmproj "$MMPROJ") ;;&
  both)      ARGS+=(--no-mmproj-offload) ;;
  fast|vision|long) ;;
  *) echo "--mode must be both|fast|vision|long" >&2; exit 2 ;;
esac

if [[ $THINKING -eq 1 ]]; then ARGS+=(--reasoning-effort medium)
else                           ARGS+=(--reasoning-budget 0); fi

case "$MODE" in
  both)   EXPECT="MTP + vision  (~87 t/s code, ~48 t/s with an image, images supported)" ;;
  fast)   EXPECT="MTP only      (~75-87 t/s, no images)" ;;
  vision) EXPECT="vision only   (~38 t/s, images supported)" ;;
  long)   EXPECT="long context  (128k, ~24-38 t/s, vision, no MTP)" ;;
esac

echo
echo "  Qwen3.8-27B  ->  $EXPECT"
echo "  context $CTX | OpenAI endpoint: http://${BIND}:$PORT/v1"
echo "  model id  : $ALIAS  |  auth: ${APIKEY:+api key set}${APIKEY:-none (any dummy key works)}"
[[ "$BIND" == "127.0.0.1" ]] && echo "  note: bound to localhost. For Docker/LAN clients use --bind 0.0.0.0"
nvidia-smi --query-gpu=name,enforced.power.limit,memory.used --format=csv,noheader
lim=$(nvidia-smi --query-gpu=enforced.power.limit --format=csv,noheader | tr -dc '0-9.' | cut -d. -f1)
if [[ -n "$lim" && "$lim" -lt 150 ]]; then
  echo "  note: power limit is ${lim} W, not 175 W. Fn+Q (Performance) is worth"
  echo "        +32% decode / +42% prefill. Starting anyway - low TGP is only slower."
fi
echo

exec "$LLAMA_BIN" "${ARGS[@]}"
