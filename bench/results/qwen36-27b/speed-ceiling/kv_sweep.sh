#!/bin/bash
set -uo pipefail

# Sweep KV cache types on Madreag fork with Qwen 3.6 27B IQ4_XS @ 64k.
# Goal: prove or disprove that KV format affects 27B Dense throughput.
# Already tested in prior bench: turbo3 (24.18), q8_0 (20.08), bf16 (19.80).
# This sweep covers: q4_0, iq4_nl, q5_0, turbo1.5, turbo2, turbo4, turbo3_tcq, turbo2_tcq

LLAMA=$HOME/llama-cpp-turboquant/llama-server
MODEL=$HOME/models/Qwen3.6-27B-IQ4_XS.gguf
PORT=8090
CTX=65536
RESULTS=/tmp/kv_sweep_27b_results.tsv
LOGDIR=/tmp/kv_sweep_27b_logs
mkdir -p "$LOGDIR"
echo -e "kv_type\tt_per_s\tprompt_t_per_s\tvram_mib\ttokens" > "$RESULTS"

# Standard short bench prompt for throughput measurement
read -r -d '' BENCH_PROMPT <<'EOF' || true
Write a concise 800-token Python explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs.
EOF

bench_one() {
    local kv="$1"
    local log="$LOGDIR/$kv.log"
    echo "[sweep] === KV=$kv ==="

    # Pre-flight: kill any leftover server on port
    pkill -9 -f "llama-cpp-turboquant.*--port $PORT" 2>/dev/null || true
    sleep 3

    # Launch server
    CUDA_VISIBLE_DEVICES=0 "$LLAMA" \
        -m "$MODEL" \
        --alias qwen3.6-27b -c "$CTX" -np 1 -ngl 999 -fa on \
        --cache-type-k "$kv" --cache-type-v "$kv" \
        --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
        --reasoning-budget 0 \
        --host 127.0.0.1 --port $PORT \
        --jinja --reasoning-format deepseek \
        > "$log" 2>&1 &
    local pid=$!

    # Wait for ready (max 90s)
    local ready=0
    for i in $(seq 1 90); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            ready=1; break
        fi
        # Detect early crash
        if ! kill -0 $pid 2>/dev/null; then
            echo "[sweep] $kv FAILED to start"
            tail -8 "$log"
            echo -e "$kv\tFAIL_START\t-\t-\t-" >> "$RESULTS"
            return 1
        fi
        sleep 1
    done
    if [ "$ready" -eq 0 ]; then
        echo "[sweep] $kv timed out"
        tail -8 "$log"
        kill -9 $pid 2>/dev/null
        echo -e "$kv\tFAIL_TIMEOUT\t-\t-\t-" >> "$RESULTS"
        return 1
    fi

    # Capture VRAM
    local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    # Bench: 800 token generation
    local resp
    resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json,sys
print(json.dumps({
    'model':'qwen3.6-27b',
    'messages':[{'role':'user','content':'''$BENCH_PROMPT'''}],
    'max_tokens':800,
    'temperature':0.6,'top_p':0.95,'top_k':20,
    'chat_template_kwargs':{'enable_thinking':False}
}))
")")

    # Parse timings via /v1/chat/completions response (has timings)
    local tps prompt_tps tokens
    tps=$(echo "$resp" | python3 -c "import json,sys;d=json.load(sys.stdin);t=d.get('timings',{});print(round(t.get('predicted_per_second',0),2))" 2>/dev/null || echo "0")
    prompt_tps=$(echo "$resp" | python3 -c "import json,sys;d=json.load(sys.stdin);t=d.get('timings',{});print(round(t.get('prompt_per_second',0),2))" 2>/dev/null || echo "0")
    tokens=$(echo "$resp" | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo "0")

    echo "[sweep] $kv: gen=$tps t/s, prompt=$prompt_tps t/s, vram=${vram} MiB, tokens=$tokens"
    echo -e "$kv\t$tps\t$prompt_tps\t$vram\t$tokens" >> "$RESULTS"

    # Shutdown
    kill -9 $pid 2>/dev/null
    sleep 4
}

# Sweep order: previously tested baselines first to verify reproducibility, then new types
for KV in turbo3 q8_0 bf16 q4_0 iq4_nl q5_0 turbo1.5 turbo2 turbo4 turbo3_tcq turbo2_tcq; do
    bench_one "$KV" || true
done

echo
echo "=== Results ==="
column -t -s $'\t' "$RESULTS"
