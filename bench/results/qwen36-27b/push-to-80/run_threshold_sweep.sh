#!/bin/bash
set -uo pipefail
# Push SGLang past 64 t/s code, 50 t/s prose. Best so far: n3_k2_d8 (50/55), n5_k1_d6 (42/64).
# Try: threshold loosening + bigger trees.

LORBUS=$HOME/models/Lorbus-Qwen3.6-27B-int4-AutoRound
PORT=8082
RESULTS=/tmp/sglang_push80.tsv
LOGDIR=/tmp/sglang_push80_logs
mkdir -p "$LOGDIR"
echo -e "config\tnum_steps\ttopk\tdraft_tok\tthresh_s\tthresh_a\tprompt\tgen_t_per_s\ttokens" > "$RESULTS"

PROSE='Write a concise 800-token Python explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs.'
CODE='Write a complete TypeScript implementation of a binary search tree with insert, delete, search, and in-order traversal methods. Include unit tests for each method. Aim for 800 tokens of code.'

cleanup() {
    pkill -9 -f "sglang.launch" 2>/dev/null || true
    sleep 6
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null; done
    sleep 3
}

bench_one() {
    local name="$1"
    local steps="$2"
    local topk="$3"
    local draft="$4"
    local thresh_s="${5:-1.0}"
    local thresh_a="${6:-1.0}"
    local log="$LOGDIR/${name}.log"
    cleanup

    echo "[push80 $name] s=$steps tk=$topk d=$draft ts=$thresh_s ta=$thresh_a"
    SGLANG_DISABLE_CUDNN_CHECK=1 \
    PATH=$HOME/sglang_env/bin:$PATH \
    CUDA_VISIBLE_DEVICES=0 \
    nohup $HOME/sglang_env/bin/python -m sglang.launch_server \
        --model-path "$LORBUS" --served-model-name qwen3.6-27b \
        --quantization auto-round --dtype bfloat16 \
        --tp 1 --context-length 65536 \
        --mem-fraction-static 0.85 --max-running-requests 1 \
        --kv-cache-dtype fp8_e4m3 \
        --port $PORT --host 0.0.0.0 --trust-remote-code \
        --speculative-algorithm NEXTN \
        --speculative-num-steps "$steps" \
        --speculative-eagle-topk "$topk" \
        --speculative-num-draft-tokens "$draft" \
        --speculative-accept-threshold-single "$thresh_s" \
        --speculative-accept-threshold-acc "$thresh_a" \
        > "$log" 2>&1 </dev/null &
    local pid=$!

    for i in $(seq 1 360); do
        if grep -qE "fired up|Uvicorn running|listening on" "$log" 2>/dev/null; then break; fi
        if grep -qE "RuntimeError|Traceback|SIGQUIT|illegal memory" "$log" 2>/dev/null; then
            echo "[$name] CRASH"
            tail -5 "$log"
            echo -e "$name\t$steps\t$topk\t$draft\t$thresh_s\t$thresh_a\tprose\tCRASH\t-" >> "$RESULTS"
            kill -9 $pid 2>/dev/null
            cleanup
            return
        fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$name] DIED"; tail -10 "$log"
            echo -e "$name\t$steps\t$topk\t$draft\t$thresh_s\t$thresh_a\tprose\tDIED\t-" >> "$RESULTS"
            cleanup; return
        fi
        sleep 3
    done

    sleep 5
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
        echo -e "$name\t$steps\t$topk\t$draft\t$thresh_s\t$thresh_a\t$label\t$tps\t$tokens" >> "$RESULTS"
    done

    kill -9 $pid 2>/dev/null
    cleanup
}

# Best balanced from prior sweep + threshold loosening
bench_one "n3_k2_d8_t07" 3 2 8 0.7 0.9
# Best code champion + threshold loosening — push for 80 t/s
bench_one "n5_k1_d6_t07" 5 1 6 0.7 0.9
bench_one "n5_k1_d6_t05" 5 1 6 0.5 0.9
# Tree at depth 5 (was untested)
bench_one "n5_k2_d10" 5 2 10 1.0 1.0
# Tree at depth 5 + thresholds
bench_one "n5_k2_d10_t07" 5 2 10 0.7 0.9
# Aggressive — n6, deepest valid
bench_one "n6_k1_d7" 6 1 7 1.0 1.0
bench_one "n6_k1_d7_t07" 6 1 7 0.7 0.9
# Auto-tune
bench_one "auto_tune" 0 0 0 1.0 1.0

echo
echo "=== SGLang push80 Results ==="
column -t -s $'\t' "$RESULTS"
