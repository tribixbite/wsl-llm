#!/bin/bash
set -uo pipefail
# Qwen3.8-27B via the syv-ai vLLM W4A16 stack — the throughput champion on this box.
#
#   105 t/s avg (74.5 prose / 116.4 code / 124.1 json), TTFT ~130-170 ms
#   ~2x the best llama.cpp config. aider pass@2 = 50.0%.
#
# ⚠️ MAX_SEQS=1 IS MANDATORY.
# With the default 8 slots at gpu_memory_utilization 0.93 (~22.3/24 GB) the model
# loads, LISTENs, logs "Application startup complete" — and then WDDM silently
# evicts the weights to system RAM (WSL2 has no OOM guardrail). /health keeps
# answering while every real request times out. Same trap as llama.cpp
# --parallel 4; see docs/QWEN38_27B_LEGION_BENCHMARKS.md §3a.
# Always validate a config change with a TIMED GENERATION, never just /health.

REPO_DIR="${QWEN38_VLLM_DIR:-$HOME/qwen38-vllm}"
export MODEL="${MODEL:-$REPO_DIR/models2/Qwen3.8-27B-W4A16-AutoRound}"
export MAX_SEQS="${MAX_SEQS:-1}"          # do not raise; see above
export CTX="${CTX:-fast}"                 # 'fast' = 64k, 'long' = 150k
export PREFIX_CACHE="${PREFIX_CACHE:-1}"

[ -d "$REPO_DIR" ]  || { echo "ERROR: vLLM stack not found at $REPO_DIR"; exit 1; }
[ -d "$MODEL" ]     || { echo "ERROR: model not found at $MODEL"; exit 1; }
[ -x "$REPO_DIR/venv/bin/vllm" ] || { echo "ERROR: venv missing at $REPO_DIR/venv"; exit 1; }

cd "$REPO_DIR"
exec bash single-user/start_qwen.sh
