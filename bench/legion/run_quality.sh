#!/usr/bin/env bash
# Quality (pass@1) runs for the Legion box, using the house aider_lite harness
# so results are directly comparable to the existing 27B / 35B-A3B datapoints.
#
# Each entry launches its own llama-server, runs the 34-exercise Python subset,
# then tears down. Sampling is Qwen3.8's official non-thinking preset unless the
# config enables thinking.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
MTP="${MTP:-$HOME/models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
PORT="${PORT:-8080}"
N="${N:-34}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/bench/results/legion-qwen38/quality}"
mkdir -p "$OUT_DIR"

stop_server() {
  pkill -9 -f "$LLAMA_BIN" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "$LLAMA_BIN" >/dev/null || break; sleep 0.5; done
  sleep 3
}
trap stop_server EXIT

# tag|server flags|aider_lite extra flags
RUNS=(
  "nothink_baseline|-c 32768 -ctk q8_0 -ctv q8_0 --parallel 1 --reasoning-budget 0|"
  "nothink_mtp|-c 32768 -ctk q8_0 -ctv q8_0 --parallel 1 --reasoning-budget 0 --spec-type draft-mtp -md $MTP|"
  "think_mtp|-c 32768 -ctk q8_0 -ctv q8_0 --parallel 1 --spec-type draft-mtp -md $MTP|--think"
)
if [[ -n "${RUNS_FILE:-}" ]]; then
  mapfile -t RUNS < <(grep -vE '^\s*(#|$)' "$RUNS_FILE")
fi

for entry in "${RUNS[@]}"; do
  tag="${entry%%|*}"; rest="${entry#*|}"
  sflags="${rest%%|*}"; aflags="${rest#*|}"
  echo "================ $tag ================"
  stop_server

  # shellcheck disable=SC2086
  setsid "$LLAMA_BIN" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port "$PORT" \
      --jinja --no-webui $sflags > "$OUT_DIR/${tag}.server.log" 2>&1 < /dev/null &

  up=0
  for _ in $(seq 1 150); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    sleep 2
  done
  [[ $up -eq 1 ]] || { echo "!! server failed for $tag"; tail -5 "$OUT_DIR/${tag}.server.log"; continue; }

  start=$(date +%s)
  # shellcheck disable=SC2086
  python3 "$REPO_DIR/bench/aider_lite.py" --url "http://127.0.0.1:$PORT" \
      --model "qwen3.8-27b-$tag" --n "$N" --out "$OUT_DIR/${tag}.json" $aflags \
      2>&1 | tee "$OUT_DIR/${tag}.txt" | tail -4
  echo "  elapsed: $(( $(date +%s) - start ))s"
  sleep 30
done

stop_server
echo "=== quality summary ==="
python3 - "$OUT_DIR" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
rows = []
for f in sorted(d.glob("*.json")):
    o = json.load(open(f))
    rows.append((f.stem, o["pass"], o["n"], 100 * o["pass"] / o["n"]))
w = max((len(r[0]) for r in rows), default=10)
for name, p, n, pct in rows:
    print(f"{name:<{w}}  {p:>2}/{n:<3} = {pct:5.1f}%")
PY
