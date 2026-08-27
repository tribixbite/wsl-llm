#!/usr/bin/env bash
# Correctness gate on the MTP speculative-decoding speedup.
#
# Speculative decoding is only a free win if it is distribution-preserving: the
# draft head proposes tokens and the target model verifies them, so greedy
# output MUST be byte-identical with and without the draft. If it is not, MTP
# is trading accuracy for speed and the 1.9x headline is not free.
#
# We run the same prompts at temperature 0 through both configs and diff.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
MTP="${MTP:-$HOME/models/Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
PORT="${PORT:-8080}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/bench/results/legion-qwen38/mtp-equivalence}"
mkdir -p "$OUT_DIR"

stop_server() {
  pkill -9 -f "$LLAMA_BIN" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "$LLAMA_BIN" >/dev/null || break; sleep 0.5; done
  sleep 3
}
trap stop_server EXIT

start_server() {  # start_server <tag> <extra flags...>
  local tag="$1"; shift
  stop_server
  setsid "$LLAMA_BIN" -m "$MODEL" -ngl 99 -fa on -c 16384 -ctk q8_0 -ctv q8_0 \
      --parallel 1 --host 127.0.0.1 --port "$PORT" --jinja --no-webui "$@" \
      > "$OUT_DIR/${tag}.server.log" 2>&1 < /dev/null &
  for _ in $(seq 1 150); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "!! server failed to start for $tag"; tail -5 "$OUT_DIR/${tag}.server.log"; return 1
}

probe() { python3 "$REPO_DIR/bench/legion/greedy_probe.py" --url "http://127.0.0.1:$PORT" --out "$1"; }

# Two baseline samples establish whether the engine is deterministic AT ALL on
# this hardware. Without that control, a baseline-vs-MTP difference is
# uninterpretable: it could just be run-to-run noise.
start_server baseline || exit 1
probe "$OUT_DIR/baseline.txt"
probe "$OUT_DIR/baseline_rerun.txt"

start_server mtp --spec-type draft-mtp -md "$MTP" || exit 1
probe "$OUT_DIR/mtp.txt"
stop_server

echo "=============================================="
if diff -q "$OUT_DIR/baseline.txt" "$OUT_DIR/baseline_rerun.txt" >/dev/null; then
  echo "CONTROL: baseline is DETERMINISTIC across runs."
  det=yes
else
  echo "CONTROL: baseline is NONDETERMINISTIC across runs — engine-level, not MTP's fault."
  det=no
fi

if diff -q "$OUT_DIR/baseline.txt" "$OUT_DIR/mtp.txt" >/dev/null; then
  echo "RESULT : MTP output is byte-identical to baseline at temperature 0."
  echo "         The speedup is free — no accuracy traded."
elif [[ $det == no ]]; then
  echo "RESULT : MTP differs from baseline, but so does baseline from itself."
  echo "         Inconclusive — the engine is not reproducible at temp 0."
else
  echo "RESULT : baseline reproduces exactly, but MTP DIFFERS."
  echo "         MTP changes numerics (batch-shape-dependent FP reduction order)."
  echo "         Quantify with a downstream quality benchmark, not this probe."
fi
echo "=============================================="
python3 - "$OUT_DIR" <<'PY'
import sys, pathlib
d = pathlib.Path(sys.argv[1]); SEP = "\n<<<===PROMPT-SEP===>>>\n"
runs = {n: (d / f"{n}.txt").read_text().split(SEP)
        for n in ("baseline", "baseline_rerun", "mtp") if (d / f"{n}.txt").exists()}
n = len(next(iter(runs.values())))
print(f"{'prompt':>7}  " + "  ".join(f"{k:>16}" for k in runs))
for i in range(n):
    cells = "  ".join(f"{len(v[i]):>16}" for v in runs.values())
    same = "SAME" if len({v[i] for v in runs.values()}) == 1 else "DIFF"
    print(f"{i:>7}  {cells}   {same}")
PY
