#!/bin/bash
set -uo pipefail
# Phase 1: compare three llama.cpp-family engines on Qwen3.6-27B-IQ4_XS @ 64k.
# Same model, same prompt, same KV type per-engine (best supported).

MODEL=$HOME/models/Qwen3.6-27B-IQ4_XS.gguf
PORT=8090
CTX=65536
RESULTS=/tmp/engine_phase1.tsv
LOGDIR=/tmp/engine_phase1_logs
mkdir -p "$LOGDIR"
echo -e "engine\tkv\tgen_t_per_s\tprompt_t_per_s\tvram_mib\ttokens" > "$RESULTS"

read -r -d '' BENCH_PROMPT <<'EOF' || true
Write a concise 800-token Python explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs.
EOF

bench_one() {
    local name="$1"
    local bin="$2"
    local kv="$3"
    local extra="$4"
    local log="$LOGDIR/${name}_${kv}.log"
    [ ! -x "$bin" ] && { echo "[$name] SKIP — binary not found: $bin"; return; }

    pkill -9 -f "llama-server.*--port $PORT" 2>/dev/null || true
    sleep 4

    echo "[$name kv=$kv] launching..."
    CUDA_VISIBLE_DEVICES=0 "$bin" \
        -m "$MODEL" \
        --alias qwen3.6-27b -c "$CTX" -np 1 -ngl 999 -fa on \
        --cache-type-k "$kv" --cache-type-v "$kv" \
        --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
        --host 127.0.0.1 --port $PORT $extra \
        > "$log" 2>&1 &
    local pid=$!
    for i in $(seq 1 90); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$name kv=$kv] CRASH"
            tail -10 "$log"
            echo -e "$name\t$kv\tCRASH\t-\t-\t-" >> "$RESULTS"
            return
        fi
        sleep 1
    done

    local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    local resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json
print(json.dumps({
    'model':'qwen3.6-27b',
    'messages':[{'role':'user','content':'''$BENCH_PROMPT'''}],
    'max_tokens':800,
    'temperature':0.6,'top_p':0.95,'top_k':20
}))
")")

    local tps=$(echo "$resp" | python3 -c "import json,sys;d=json.load(sys.stdin);t=d.get('timings',{});print(round(t.get('predicted_per_second',0),2))" 2>/dev/null || echo 0)
    local prompt_tps=$(echo "$resp" | python3 -c "import json,sys;d=json.load(sys.stdin);t=d.get('timings',{});print(round(t.get('prompt_per_second',0),2))" 2>/dev/null || echo 0)
    local tokens=$(echo "$resp" | python3 -c "import json,sys;print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
    echo "[$name kv=$kv] gen=$tps t/s, prompt=$prompt_tps, vram=${vram} MiB, tokens=$tokens"
    echo -e "$name\t$kv\t$tps\t$prompt_tps\t$vram\t$tokens" >> "$RESULTS"
    kill -9 $pid 2>/dev/null
    sleep 4
}

# Madreag (baseline) with turbo3 KV — already known winner for this fork
bench_one "madreag" "$HOME/llama-cpp-turboquant/llama-server" "turbo3" "--reasoning-budget 0 --jinja --reasoning-format deepseek"
# Madreag with q8_0 KV (fairer comparison to upstream that lacks turbo3)
bench_one "madreag" "$HOME/llama-cpp-turboquant/llama-server" "q8_0"   "--reasoning-budget 0 --jinja --reasoning-format deepseek"
# ik_llama.cpp — has its own KV variants and Hadamard transforms
bench_one "ik_llama" "$HOME/ik_llama.cpp/build/bin/llama-server" "q8_0" "--jinja"
bench_one "ik_llama" "$HOME/ik_llama.cpp/build/bin/llama-server" "q4_0" "--jinja"
# Upstream llama.cpp (latest)
bench_one "upstream" "$HOME/llama.cpp/build/bin/llama-server" "q8_0"   "--jinja"
bench_one "upstream" "$HOME/llama.cpp/build/bin/llama-server" "f16"    "--jinja"

echo
echo "=== Engine Phase 1 Results ==="
column -t -s $'\t' "$RESULTS"
