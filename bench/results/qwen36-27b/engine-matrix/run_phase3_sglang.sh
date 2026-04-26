#!/bin/bash
set -uo pipefail
# Phase 3: SGLang on Qwen3.6-27B Lorbus AutoRound INT4 with NEXTN (MTP) speculative decoding.

LORBUS=$HOME/models/Lorbus-Qwen3.6-27B-int4-AutoRound
PORT=8082
RESULTS=/tmp/sglang_phase3.tsv
LOGDIR=/tmp/sglang_phase3_logs
mkdir -p "$LOGDIR"
echo -e "config\tprompt\tgen_t_per_s\tvram_mib\ttokens\telapsed_s" > "$RESULTS"

PROSE_PROMPT="Write a concise 800-token Python explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs."
CODE_PROMPT="Write a complete TypeScript implementation of a binary search tree with insert, delete, search, and in-order traversal methods. Include unit tests for each method. Aim for 800 tokens of code."

bench_config() {
    local name="$1"
    local extra_args="$2"
    local log="$LOGDIR/${name}.log"

    pkill -9 -f "sglang.launch" 2>/dev/null || true
    pkill -9 -f "Lorbus-Qwen3.6" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null; done
    sleep 8

    echo "[sglang $name] launching..."
    eval "CUDA_VISIBLE_DEVICES=0 nohup $HOME/sglang_env/bin/python -m sglang.launch_server \
        --model-path '$LORBUS' \
        --served-model-name qwen3.6-27b \
        --quantization auto_round --dtype float16 \
        --tp 1 --context-length 65536 \
        --mem-fraction-static 0.92 --max-running-requests 1 \
        --kv-cache-dtype fp8_e4m3 \
        --port $PORT --host 0.0.0.0 \
        --trust-remote-code \
        $extra_args \
        > $log 2>&1 &"

    # Wait for ready (SGLang prints 'The server is fired up' or 'serving HTTP')
    local ready=0
    for i in $(seq 1 360); do
        if grep -qE "The server is fired up|started server|Uvicorn running|listening on" "$log" 2>/dev/null; then ready=1; break; fi
        if grep -qE "Traceback|ERROR|illegal memory|RuntimeError|FileNotFound|ImportError" "$log" 2>/dev/null; then
            echo "[$name] CRASH"
            grep -E "Error|Traceback|RuntimeError|FileNotFound" "$log" | head -5
            echo -e "$name\tprose\tCRASH\t-\t-\t-" >> "$RESULTS"
            return
        fi
        sleep 2
    done
    [ $ready -eq 0 ] && { echo "[$name] TIMEOUT"; tail -20 "$log"; echo -e "$name\tprose\tTIMEOUT\t-\t-\t-" >> "$RESULTS"; return; }

    sleep 5
    local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    for prompt_label in prose code; do
        if [ "$prompt_label" = "prose" ]; then PROMPT="$PROSE_PROMPT"; else PROMPT="$CODE_PROMPT"; fi
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
        if [ "$tokens" -gt 0 ]; then tps=$(python3 -c "print(round($tokens / $elapsed, 2))"); fi
        echo "[$name $prompt_label] tokens=$tokens, t/s=$tps, elapsed=${elapsed}s, vram=${vram}"
        echo -e "$name\t$prompt_label\t$tps\t$vram\t$tokens\t$elapsed" >> "$RESULTS"
    done

    pkill -9 -f "sglang.launch" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null; done
    sleep 8
}

# Configurations to test
# 1. Plain SGLang, no speculative — direct vLLM-no-MTP comparison
bench_config "default_nospec" ""
# 2. NEXTN (MTP) n=3 — direct vLLM-MTP-n3 comparison
bench_config "nextn_n3" "--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4"
# 3. NGRAM (n-gram speculative — works on any model, no head needed)
bench_config "ngram" "--speculative-algorithm NGRAM --speculative-ngram-min-match-window-size 1 --speculative-ngram-max-match-window-size 6 --speculative-num-draft-tokens 8"

echo
echo "=== SGLang Phase 3 Results ==="
column -t -s $'\t' "$RESULTS"
