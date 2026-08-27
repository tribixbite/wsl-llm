#!/usr/bin/env bash
# llama.cpp config matrix for the Legion RTX 5080 (16 GB, sm_120).
#
# Launches llama-server once per config, waits for /health, runs the house
# streaming decode-TPS bench, records peak VRAM and the GPU power envelope,
# then tears the server down before the next config.
#
# The power cap on this laptop DRIFTS between runs (84-94 W observed against a
# 175 W hardware max), so every row logs enforced.power.limit and peak draw.
# Never compare two rows without checking those columns.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
MTP="${MTP:-$HOME/models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
PORT="${PORT:-8080}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/bench/results/legion-qwen38}"
TSV="$OUT_DIR/llamacpp_matrix.tsv"

mkdir -p "$OUT_DIR"

# name|extra llama-server flags
#
# --parallel 1 is MANDATORY on this box: the default of 4 slots allocates a
# separate DeltaNet recurrent state + compute buffers per slot, pushing peak
# VRAM from 13384 -> 15941 MiB. That crosses the 16303 MiB ceiling, WDDM
# silently evicts the weights to system RAM (no OOM under WSL2), and decode
# collapses from 27 t/s to 0.04 t/s. See memory/pm.md.
#
# Override the whole list by exporting CONFIGS_FILE=<path>, one "name|flags"
# entry per line, to run a focused sweep without editing this file.
CONFIGS=(
  "baseline_32k_q8kv|-c 32768 -ctk q8_0 -ctv q8_0 --parallel 1"
  "mtp_32k_q8kv|-c 32768 -ctk q8_0 -ctv q8_0 --parallel 1 --spec-type draft-mtp -md $MTP"
  "ngrammod_32k_q8kv|-c 32768 -ctk q8_0 -ctv q8_0 --parallel 1 --spec-type ngram-mod"
  "ngramsimple_32k_q8kv|-c 32768 -ctk q8_0 -ctv q8_0 --parallel 1 --spec-type ngram-simple"
  "ngrammapk_32k_q8kv|-c 32768 -ctk q8_0 -ctv q8_0 --parallel 1 --spec-type ngram-map-k"
  "baseline_32k_f16kv|-c 32768 -ctk f16 -ctv f16 --parallel 1"
  # Asymmetric KV: q4_0 on K alone reproduces full quality collapse, while q4_0
  # on V alone costs ~1/500 (llama.cpp#21591). Needs GGML_CUDA_FA_ALL_QUANTS=ON,
  # otherwise K->type != V->type disables flash attention entirely.
  "baseline_64k_q8k_q4v|-c 65536 -ctk q8_0 -ctv q4_0 --parallel 1"
  "baseline_64k_q4kv|-c 65536 -ctk q4_0 -ctv q4_0 --parallel 1"
)

if [[ -n "${CONFIGS_FILE:-}" ]]; then
  mapfile -t CONFIGS < <(grep -vE '^\s*(#|$)' "$CONFIGS_FILE")
  echo "using ${#CONFIGS[@]} configs from $CONFIGS_FILE"
fi

stop_server() {
  pkill -f "$LLAMA_BIN" 2>/dev/null
  for _ in $(seq 1 40); do
    pgrep -f "$LLAMA_BIN" >/dev/null || break
    sleep 0.5
  done
  pkill -9 -f "$LLAMA_BIN" 2>/dev/null
  sleep 2
}
trap stop_server EXIT

[[ -f "$TSV" ]] || printf 'ts\tconfig\tprompt\tttft_s\tdecode_tps\tcompletion_tokens\tpeak_vram_mib\tpwr_limit_w\tpeak_draw_w\tmax_temp_c\n' > "$TSV"

for entry in "${CONFIGS[@]}"; do
  name="${entry%%|*}"
  flags="${entry#*|}"
  echo "================ $name ================"
  echo "flags: $flags"

  stop_server
  log="$OUT_DIR/${name}.server.log"
  # CUDA graphs are left ENABLED: measured +20% decode on this box and no Xid 8
  # hang, contrary to llama.cpp#27330 which reports hangs on RTX 5090 Laptop.
  # shellcheck disable=SC2086
  "$LLAMA_BIN" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port "$PORT" \
      --jinja --no-webui $flags > "$log" 2>&1 &

  ok=0
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
    sleep 2
  done
  if [[ $ok -ne 1 ]]; then
    echo "!! server failed to become healthy for $name — see $log"
    grep -iE "error|failed|out of memory|CUDA" "$log" | tail -5
    printf '%s\t%s\tLOAD_FAILED\t\t\t\t\t\t\t\n' "$(date -Is)" "$name" >> "$TSV"
    continue
  fi

  pwr_csv="$OUT_DIR/${name}.power.csv"
  nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,enforced.power.limit,temperature.gpu,memory.used \
             --format=csv,noheader -l 1 > "$pwr_csv" 2>&1 &
  pwr_pid=$!

  python3 "$REPO_DIR/bench/stream_bench.py" \
      --url "http://127.0.0.1:$PORT" --model qwen3.8-27b --label "$name" \
      --tsv "$OUT_DIR/${name}.stream.tsv" 2>&1 | tee "$OUT_DIR/${name}.bench.txt"

  kill "$pwr_pid" 2>/dev/null; wait "$pwr_pid" 2>/dev/null

  read -r peak_vram pwr_limit peak_draw max_temp < <(
    awk -F', ' '{gsub(/ W|MiB| C/,"");
                 if($6+0>mm)mm=$6+0; if($3+0>md)md=$3+0; if($5+0>mt)mt=$5+0; lim=$4+0}
                END{printf "%d %.1f %.1f %d", mm, lim, md, mt}' "$pwr_csv")
  echo "peak_vram=${peak_vram}MiB pwr_limit=${pwr_limit}W peak_draw=${peak_draw}W max_temp=${max_temp}C"

  # stream_bench.py appends: label \t prompt \t decode_tps \t ttft_ms \t tokens \t wall
  # (no header). Join those with the GPU telemetry captured above.
  if [[ -f "$OUT_DIR/${name}.stream.tsv" ]]; then
    awk -v n="$name" -v v="$peak_vram" -v l="$pwr_limit" -v d="$peak_draw" -v t="$max_temp" \
        -v ts="$(date -Is)" 'BEGIN{FS=OFS="\t"} NF>=6 {
          print ts, n, $2, $4/1000, $3, $5, v, l, d, t }' \
        "$OUT_DIR/${name}.stream.tsv" >> "$TSV"
    rm -f "$OUT_DIR/${name}.stream.tsv"   # keep re-runs from double-counting
  fi

  # Thermal cooldown so the next config does not start heat-soaked.
  sleep 45
done

stop_server
echo "=== matrix complete -> $TSV ==="
column -t -s $'\t' "$TSV"
