#!/usr/bin/env bash
# Official Aider polyglot benchmark against the local llama-server.
#
# Differs from bench/aider_lite.py in two ways that matter:
#   1. **diff edit format** — the model must emit SEARCH/REPLACE blocks that apply
#      cleanly, not a whole rewritten file. This measures edit fidelity, which the
#      whole-file harness cannot see at all.
#   2. multi-language (python/go/rust/...) rather than the Python subset.
# It also runs 2 tries by default, so the headline number is pass_rate_2.
#
# AIDER_DOCKER=1 only bypasses a warning-and-return gate (benchmark.py:251); the
# generated code then executes on the host. That is the same exposure as
# aider_lite's local pytest runs. Podman is available if you want to sandbox it.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AIDER_DIR="${AIDER_DIR:-$HOME/aider}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
PORT="${PORT:-8080}"
LANGS="${LANGS:-python,go,rust}"
NUM_TESTS="${NUM_TESTS:-30}"
EDIT_FORMAT="${EDIT_FORMAT:-diff}"
RUN_NAME="${RUN_NAME:-qwen38-polyglot}"
# CONT=1 resumes the single matching testdir instead of starting a new one.
# This box bugchecks at random (see docs §10), and the harness keeps a
# .aider.results.json per exercise, so resuming is exact.
CONT="${CONT:-0}"
# benchmark.py does random.shuffle(test_dnames) with NO seed (~line 343), so
# --num-tests N picks a DIFFERENT random N every run. Any A/B across runs is
# invalid unless the set is pinned. KEYWORDS pins it to exact
# "<lang>/exercises/practice/<name>" paths via --keywords.
KEYWORDS="${KEYWORDS:-}"
SERVER_FLAGS="${SERVER_FLAGS:--c 32768 -ctk q8_0 -ctv q8_0 --parallel 1 --reasoning-effort medium}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/bench/results/legion-qwen38/aider-polyglot}"

mkdir -p "$OUT_DIR" "$AIDER_DIR/tmp.benchmarks"
# The harness resolves exercises as  $AIDER_BENCHMARK_DIR/<exercises-dir>
[[ -e "$AIDER_DIR/tmp.benchmarks/polyglot-benchmark" ]] || \
  ln -s "$HOME/polyglot-benchmark" "$AIDER_DIR/tmp.benchmarks/polyglot-benchmark"

stop_server() {
  pkill -9 -f "$LLAMA_BIN" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -f "$LLAMA_BIN" >/dev/null || break; sleep 0.5; done
  sleep 2
}
trap stop_server EXIT

stop_server
echo "=== starting llama-server ==="
# shellcheck disable=SC2086
setsid "$LLAMA_BIN" -m "$MODEL" -ngl 99 -fa on --host 127.0.0.1 --port "$PORT" \
    --jinja --no-webui $SERVER_FLAGS > "$OUT_DIR/server.log" 2>&1 < /dev/null &

for _ in $(seq 1 150); do
  curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "!! server failed"; tail -5 "$OUT_DIR/server.log"; exit 1; }
echo "server up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"

cd "$AIDER_DIR"
export AIDER_DOCKER=1
export AIDER_BENCHMARK_DIR="$AIDER_DIR/tmp.benchmarks"
export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY="dummy"
export AIDER_MODEL_SETTINGS_FILE=/dev/null
export AIDER_ANALYTICS=false
# npm-test.sh (de-containerised by setup_aider_multilang.sh) reads this to
# find the shared node_modules the JS exercises need.
export NPM_INSTALL_DIR="${NPM_INSTALL_DIR:-$HOME/npm-install}"

"$HOME/aiderbench/bin/python" benchmark/benchmark.py "$RUN_NAME" \
  --model "openai/qwen3.8-27b" \
  --edit-format "$EDIT_FORMAT" \
  --languages "$LANGS" \
  --num-tests "$NUM_TESTS" \
  --threads 1 \
  --tries 2 \
  $([ "$CONT" = "1" ] && echo --cont || echo --new) \
  ${KEYWORDS:+--keywords "$KEYWORDS"} \
  --exercises-dir polyglot-benchmark 2>&1 | tee "$OUT_DIR/run.log" | tail -40

stop_server
echo "=== results under $AIDER_DIR/tmp.benchmarks ==="
