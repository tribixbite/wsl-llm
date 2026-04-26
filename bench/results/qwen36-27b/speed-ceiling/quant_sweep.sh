#!/bin/bash
set -uo pipefail
# Bench Madreag turboquant on smaller weight quants for 27B Dense.
# Hypothesis: smaller weights = less bandwidth = faster generation.
# Q3 sizes: UD-Q3_K_XL (14.5 GB), Q3_K_S (12.4 GB) vs IQ4_XS baseline (15.4 GB at ~24 t/s).

LLAMA=$HOME/llama-cpp-turboquant/llama-server
PORT=8090
CTX=65536
KV=turbo3
RESULTS=/tmp/q3_sweep_27b_results.tsv
LOGDIR=/tmp/q3_sweep_27b_logs
mkdir -p "$LOGDIR"
echo -e "model\tt_per_s\tprompt_t_per_s\tvram_mib\ttokens" > "$RESULTS"

read -r -d '' BENCH_PROMPT <<'EOF' || true
Write a concise 800-token Python explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs.
EOF

bench_one() {
    local name="$1"
    local model="$2"
    local log="$LOGDIR/$name.log"
    [ ! -f "$model" ] && { echo "[q3] SKIP $name — file not found: $model"; return; }
    echo "[q3] === $name ==="

    pkill -9 -f "llama-cpp-turboquant.*--port $PORT" 2>/dev/null || true
    sleep 4

    CUDA_VISIBLE_DEVICES=0 "$LLAMA" \
        -m "$model" \
        --alias qwen3.6-27b -c "$CTX" -np 1 -ngl 999 -fa on \
        --cache-type-k "$KV" --cache-type-v "$KV" \
        --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
        --reasoning-budget 0 \
        --host 127.0.0.1 --port $PORT \
        --jinja --reasoning-format deepseek \
        > "$log" 2>&1 &
    local pid=$!
    for i in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
        if ! kill -0 $pid 2>/dev/null; then echo "[q3] $name CRASH"; tail -8 "$log"; echo -e "$name\tFAIL\t-\t-\t-" >> "$RESULTS"; return; fi
        sleep 1
    done

    local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    local resp
    resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "
import json
print(json.dumps({
    'model':'qwen3.6-27b',
    'messages':[{'role':'user','content':'''$BENCH_PROMPT'''}],
    'max_tokens':800,
    'temperature':0.6,'top_p':0.95,'top_k':20,
    'chat_template_kwargs':{'enable_thinking':False}
}))
")")

    local tps prompt_tps tokens
    tps=$(echo "$resp" | python3 -c "import json,sys;d=json.load(sys.stdin);t=d.get('timings',{});print(round(t.get('predicted_per_second',0),2))" 2>/dev/null || echo 0)
    prompt_tps=$(echo "$resp" | python3 -c "import json,sys;d=json.load(sys.stdin);t=d.get('timings',{});print(round(t.get('prompt_per_second',0),2))" 2>/dev/null || echo 0)
    tokens=$(echo "$resp" | python3 -c "import json,sys;print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)

    echo "[q3] $name: gen=$tps t/s, prompt=$prompt_tps t/s, vram=${vram} MiB, tokens=$tokens"
    echo -e "$name\t$tps\t$prompt_tps\t$vram\t$tokens" >> "$RESULTS"

    kill -9 $pid 2>/dev/null
    sleep 4
}

# Baselines for comparison (already known)
bench_one "iq4xs_baseline" "$HOME/models/Qwen3.6-27B-IQ4_XS.gguf"
bench_one "ud_q4_k_xl" "$HOME/models/Qwen3.6-27B-UD-Q4_K_XL.gguf"
# Smaller quant
bench_one "ud_q3_k_xl" "$HOME/models/Qwen3.6-27B-UD-Q3_K_XL.gguf"

echo
echo "=== Results ==="
column -t -s $'\t' "$RESULTS"
