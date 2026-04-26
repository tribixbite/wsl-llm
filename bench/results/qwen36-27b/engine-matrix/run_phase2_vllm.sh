#!/bin/bash
set -uo pipefail
# Phase 2: vLLM config sweep. Same model (Lorbus AutoRound), same prompt, same KV (fp8), same ctx (64k).
# Goal: find config tweak that beats the 31 t/s prose / 54 t/s code baseline.

LORBUS=$HOME/models/Lorbus-Qwen3.6-27B-int4-AutoRound
PORT=8081
RESULTS=/tmp/vllm_phase2.tsv
LOGDIR=/tmp/vllm_phase2_logs
mkdir -p "$LOGDIR"
echo -e "config\tprompt\tgen_t_per_s\tvram_mib\ttokens\telapsed_s" > "$RESULTS"

PROSE_PROMPT="Write a concise 800-token Python explanation of how an LSM tree works, including memtable, SSTables, compaction (size-tiered vs leveled), bloom filters, and tradeoffs."
CODE_PROMPT="Write a complete TypeScript implementation of a binary search tree with insert, delete, search, and in-order traversal methods. Include unit tests for each method. Aim for 800 tokens of code."

bench_config() {
    local name="$1"
    local extra_args="$2"
    local extra_env="$3"
    local log="$LOGDIR/${name}.log"

    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    # Kill orphaned EngineCore by PID
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null; done
    sleep 8

    echo "[vllm $name] launching..."
    export PATH=$HOME/bench_env/bin:$PATH
    eval "$extra_env CUDA_VISIBLE_DEVICES=0 nohup $HOME/bench_env/bin/python -m vllm.entrypoints.cli.main serve \
        '$LORBUS' \
        --served-model-name qwen3.6-27b \
        --quantization auto_round --dtype float16 \
        --tensor-parallel-size 1 --max-model-len 65536 \
        --gpu-memory-utilization 0.92 --max-num-seqs 1 \
        --kv-cache-dtype fp8 \
        --port $PORT --trust-remote-code \
        $extra_args \
        > $log 2>&1 &"

    # Wait up to 5 min for ready
    for i in $(seq 1 300); do
        grep -q "Application startup complete" "$log" 2>/dev/null && break
        if grep -q -E "EngineDeadError|FileNotFoundError|illegal memory|Traceback" "$log" 2>/dev/null; then
            echo "[$name] CRASH"
            grep -E "Error|Traceback" "$log" | head -3
            echo -e "$name\tprose\tCRASH\t-\t-\t-" >> "$RESULTS"
            return
        fi
        sleep 2
    done
    if ! grep -q "Application startup complete" "$log" 2>/dev/null; then
        echo "[$name] TIMEOUT"
        echo -e "$name\tprose\tTIMEOUT\t-\t-\t-" >> "$RESULTS"
        return
    fi

    sleep 3  # let it settle
    local vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    # Run both prompts
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
        echo "[$name $prompt_label] tokens=$tokens, t/s=$tps, elapsed=${elapsed}s"
        echo -e "$name\t$prompt_label\t$tps\t$vram\t$tokens\t$elapsed" >> "$RESULTS"
    done

    # Cleanup
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | while read pid; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null; done
    sleep 6
}

# Configurations
# 1. Default no MTP
bench_config "default_nospec" "" ""

# 2. enforce-eager (no CUDA graph compile)
bench_config "eager_nospec" "--enforce-eager" ""

# 3. Triton attention backend
bench_config "triton_nospec" "" "VLLM_ATTENTION_BACKEND=TRITON_ATTN"

# 4. MTP n=3 default (already known but redo here for apples-to-apples)
bench_config "mtp3_default" "--enable-prefix-caching --enable-chunked-prefill --speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":3}" ""

# 5. MTP n=3 + enforce-eager
bench_config "mtp3_eager" "--enforce-eager --speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":3}" ""

# 6. MTP n=3 + Triton
bench_config "mtp3_triton" "--enable-prefix-caching --enable-chunked-prefill --speculative-config={\"method\":\"mtp\",\"num_speculative_tokens\":3}" "VLLM_ATTENTION_BACKEND=TRITON_ATTN"

echo
echo "=== vLLM Phase 2 Results ==="
column -t -s $'\t' "$RESULTS"
