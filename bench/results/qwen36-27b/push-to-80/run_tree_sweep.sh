#!/bin/bash
set -uo pipefail
# SGLang NEXTN parameter sweep on Qwen3.6-27B Lorbus AutoRound INT4.
# Goal: break 54 t/s code / 43 t/s prose with bigger spec tree.
# Baseline: --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4

LORBUS=$HOME/models/Lorbus-Qwen3.6-27B-int4-AutoRound
PORT=8082
RESULTS=/tmp/sglang_nextn_sweep.tsv
LOGDIR=/tmp/sglang_nextn_logs
mkdir -p "$LOGDIR"
echo -e "config\tnum_steps\ttopk\tdraft_tok\tprompt\tgen_t_per_s\tvram_mib\ttokens" > "$RESULTS"

PROSE='Write a concise 800-token Python explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs.'
CODE='Write a complete TypeScript implementation of a binary search tree with insert, delete, search, and in-order traversal methods. Include unit tests for each method. Aim for 800 tokens of code.'

cleanup() {
    pkill -9 -f "sglang.launch" 2>/dev/null || true
    sleep 6
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null; done
    sleep 4
}

bench_one() {
    local name="$1"
    local steps="$2"
    local topk="$3"
    local draft="$4"
    local log="$LOGDIR/${name}.log"
    cleanup

    echo "[sg $name] steps=$steps topk=$topk draft=$draft launching..."
    SGLANG_DISABLE_CUDNN_CHECK=1 \
    PATH=$HOME/sglang_env/bin:$PATH \
    CUDA_VISIBLE_DEVICES=0 \
    nohup $HOME/sglang_env/bin/python -m sglang.launch_server \
        --model-path "$LORBUS" --served-model-name qwen3.6-27b \
        --quantization auto-round --dtype bfloat16 \
        --tp 1 --context-length 65536 \
        --mem-fraction-static 0.86 --max-running-requests 1 \
        --kv-cache-dtype fp8_e4m3 \
        --port $PORT --host 0.0.0.0 --trust-remote-code \
        --speculative-algorithm NEXTN \
        --speculative-num-steps "$steps" \
        --speculative-eagle-topk "$topk" \
        --speculative-num-draft-tokens "$draft" \
        > "$log" 2>&1 </dev/null &
    local pid=$!

    for i in $(seq 1 360); do
        if grep -qE "fired up|Uvicorn running|listening on" "$log" 2>/dev/null; then break; fi
        if grep -qE "RuntimeError|Traceback|SIGQUIT|Address already in use|illegal memory|FATAL" "$log" 2>/dev/null; then
            echo "[$name] CRASH"
            tail -5 "$log"
            echo -e "$name\t$steps\t$topk\t$draft\tprose\tCRASH\t-\t-" >> "$RESULTS"
            kill -9 $pid 2>/dev/null
            cleanup
            return
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$name] DIED"
            tail -10 "$log"
            echo -e "$name\t$steps\t$topk\t$draft\tprose\tDIED\t-\t-" >> "$RESULTS"
            cleanup
            return
        fi
        sleep 3
    done
    if ! grep -qE "fired up|Uvicorn running|listening on" "$log" 2>/dev/null; then
        echo "[$name] TIMEOUT"
        kill -9 $pid 2>/dev/null
        echo -e "$name\t$steps\t$topk\t$draft\tprose\tTIMEOUT\t-\t-" >> "$RESULTS"
        cleanup
        return
    fi

    sleep 5
    local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    for label in prose code; do
        if [ "$label" = "prose" ]; then PROMPT="$PROSE"; else PROMPT="$CODE"; fi
        local START=$(date +%s.%N)
        local resp=$(curl -s "http://192.168.1.32:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$(python3 -c "
import json
print(json.dumps({
    'model':'qwen3.6-27b',
    'messages':[{'role':'user','content':'''$PROMPT'''}],
    'max_tokens':800,
    'temperature':0.6,'top_p':0.95,'top_k':20
}))
")")
        local END=$(date +%s.%N)
        local elapsed=$(python3 -c "print(round($END - $START, 2))")
        local tokens=$(echo "$resp" | python3 -c "import json,sys;print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
        local tps="0"
        [ "$tokens" -gt 0 ] && tps=$(python3 -c "print(round($tokens / $elapsed, 2))")
        echo "[$name $label] t/s=$tps, tokens=$tokens, elapsed=${elapsed}s"
        echo -e "$name\t$steps\t$topk\t$draft\t$label\t$tps\t$vram\t$tokens" >> "$RESULTS"
    done

    kill -9 $pid 2>/dev/null
    cleanup
}

# Baseline (already known): 3/1/4 = 43/54 prose/code

# Wider tree: increase topk (breadth)
bench_one "n3_k2_d8"  3 2 8
bench_one "n3_k4_d12" 3 4 12

# Deeper: more spec steps
bench_one "n4_k1_d5"  4 1 5
bench_one "n5_k1_d6"  5 1 6

# Wider+deeper combo
bench_one "n4_k2_d10" 4 2 10

# Smaller tree (validate baseline reproducibility)
bench_one "n2_k1_d3"  2 1 3
bench_one "n3_k1_d4_repro" 3 1 4

echo
echo "=== SGLang NEXTN sweep ==="
column -t -s $'\t' "$RESULTS"
