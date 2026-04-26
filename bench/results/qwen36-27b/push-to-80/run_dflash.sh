#!/bin/bash
set -uo pipefail
# Bench DFlash via spiritbuun/buun-llama-cpp on Qwen3.6-27B IQ4_XS.

LLAMA=$HOME/buun-llama-cpp/build/bin/llama-server
[ ! -x "$LLAMA" ] && { echo "ERROR: buun llama-server not built yet"; exit 1; }

TARGET=$HOME/models/Qwen3.6-27B-IQ4_XS.gguf
DRAFT=$HOME/models/dflash-draft/dflash-draft-3.6-q8_0.gguf
PORT=8090
CTX=65536
RESULTS=/tmp/dflash_results.tsv
LOGDIR=/tmp/dflash_logs
mkdir -p "$LOGDIR"
echo -e "config\tprompt\tgen_t_per_s\tprompt_t_per_s\tvram_mib\ttokens" > "$RESULTS"

PROSE='Write a concise 800-token Python explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs.'
CODE='Write a complete TypeScript implementation of a binary search tree with insert, delete, search, and in-order traversal methods. Include unit tests for each method. Aim for 800 tokens of code.'

cleanup() {
    pkill -9 -f "buun-llama-cpp" 2>/dev/null || true
    sleep 5
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null; done
    sleep 3
}

bench_one() {
    local name="$1"
    shift
    local log="$LOGDIR/${name}.log"
    cleanup

    echo "[dflash $name] launching..."
    CUDA_VISIBLE_DEVICES=0 "$LLAMA" \
        -m "$TARGET" -md "$DRAFT" \
        --alias qwen3.6-27b -c "$CTX" -np 1 -ngl 999 -fa on \
        --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
        --reasoning-budget 0 \
        --host 127.0.0.1 --port $PORT \
        --jinja --reasoning-format deepseek \
        "$@" \
        > "$log" 2>&1 &
    local pid=$!

    for i in $(seq 1 180); do
        if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$name] CRASH"
            tail -8 "$log"
            echo -e "$name\tprose\tCRASH\t-\t-\t-" >> "$RESULTS"
            cleanup
            return
        fi
        sleep 2
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "[$name] TIMEOUT"
        kill -9 $pid 2>/dev/null
        echo -e "$name\tprose\tTIMEOUT\t-\t-\t-" >> "$RESULTS"
        cleanup
        return
    fi

    sleep 3
    local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    for label in prose code; do
        if [ "$label" = "prose" ]; then PROMPT="$PROSE"; else PROMPT="$CODE"; fi
        local resp=$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$(python3 -c "
import json
print(json.dumps({
    'model':'qwen3.6-27b',
    'messages':[{'role':'user','content':'''$PROMPT'''}],
    'max_tokens':800,
    'temperature':0.6,'top_p':0.95,'top_k':20,
    'chat_template_kwargs':{'enable_thinking':False}
}))
")")

        local tps=$(echo "$resp" | python3 -c "import json,sys;d=json.load(sys.stdin);t=d.get('timings',{});print(round(t.get('predicted_per_second',0),2))" 2>/dev/null || echo 0)
        local prompt_tps=$(echo "$resp" | python3 -c "import json,sys;d=json.load(sys.stdin);t=d.get('timings',{});print(round(t.get('prompt_per_second',0),2))" 2>/dev/null || echo 0)
        local tokens=$(echo "$resp" | python3 -c "import json,sys;print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
        echo "[$name $label] gen=$tps t/s, prompt=$prompt_tps, tokens=$tokens, vram=${vram}"
        echo -e "$name\t$label\t$tps\t$prompt_tps\t$vram\t$tokens" >> "$RESULTS"
    done

    kill -9 $pid 2>/dev/null
    cleanup
}

# Standard DFlash with explicit type
bench_one "dflash_default" --spec-type dflash
# DFlash with bigger draft topk for more candidates
bench_one "dflash_topk4" --spec-type dflash --draft-topk 4
# DFlash with even bigger topk
bench_one "dflash_topk8" --spec-type dflash --draft-topk 8
# Without --spec-type (auto-detect from drafter)
bench_one "dflash_auto" --draft-max 16

echo
echo "=== DFlash Results ==="
column -t -s $'\t' "$RESULTS"
