#!/bin/bash
set -euo pipefail

# Unified benchmark runner for wsl-llm
# Usage: bench.sh [quick|full|compare]

LLAMA_CONF="$HOME/llama-server.conf"
[ -f "$LLAMA_CONF" ] && source "$LLAMA_CONF" 2>/dev/null || true

API_URL="http://localhost:${PORT:-8080}/v1/chat/completions"
API_KEY="${API_KEY:-}"

# ── Helpers ──────────────────────────────────────────────────

bench_request() {
    local max_tokens="$1" prompt="$2" label="$3"
    local tmpfile start end elapsed

    tmpfile=$(mktemp)
    local body
    body=$(printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%d,"temperature":0.6,"stream":false}' \
        "${ALIAS:-qwen3.5-35b-a3b}" "$prompt" "$max_tokens")

    start=$(date +%s%N)
    curl -s --max-time 120 \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "$body" \
        "$API_URL" > "$tmpfile" 2>/dev/null
    end=$(date +%s%N)

    elapsed=$(echo "scale=2; ($end - $start) / 1000000000" | bc)
    local tokens
    tokens=$(python3 -c "import json; d=json.load(open('$tmpfile')); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
    local tps
    if [ "$tokens" -gt 0 ] && [ "$(echo "$elapsed > 0" | bc)" -eq 1 ]; then
        tps=$(echo "scale=1; $tokens / $elapsed" | bc)
    else
        tps="N/A"
    fi

    printf "  %-20s %6s tokens  %6ss  %6s t/s\n" "$label" "$tokens" "$elapsed" "$tps"
    rm -f "$tmpfile"
}

bench_concurrent() {
    local n="$1" max_tokens="$2"
    local pids=() tmpfiles=()
    local prompt="Explain the concept of recursion in 2 sentences."

    echo "  Concurrent ($n requests, $max_tokens tok each)..."
    local start
    start=$(date +%s%N)

    for i in $(seq 1 "$n"); do
        local tf
        tf=$(mktemp)
        tmpfiles+=("$tf")
        local body
        body=$(printf '{"model":"%s","messages":[{"role":"user","content":"Request %d: %s"}],"max_tokens":%d,"temperature":0.6,"stream":false}' \
            "${ALIAS:-qwen3.5-35b-a3b}" "$i" "$prompt" "$max_tokens")
        curl -s --max-time 120 \
            -H "Authorization: Bearer $API_KEY" \
            -H "Content-Type: application/json" \
            -d "$body" \
            "$API_URL" > "$tf" 2>/dev/null &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    local end
    end=$(date +%s%N)
    local elapsed
    elapsed=$(echo "scale=2; ($end - $start) / 1000000000" | bc)

    local total_tokens=0
    for tf in "${tmpfiles[@]}"; do
        local t
        t=$(python3 -c "import json; d=json.load(open('$tf')); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
        total_tokens=$((total_tokens + t))
        rm -f "$tf"
    done

    local agg_tps
    if [ "$total_tokens" -gt 0 ]; then
        agg_tps=$(echo "scale=1; $total_tokens / $elapsed" | bc)
    else
        agg_tps="N/A"
    fi

    printf "  %-20s %6s tokens  %6ss  %6s t/s (aggregate)\n" "${n}x concurrent" "$total_tokens" "$elapsed" "$agg_tps"
}

check_server() {
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 \
        -H "Authorization: Bearer $API_KEY" \
        "http://localhost:${PORT:-8080}/health" 2>/dev/null || echo "000")
    if [ "$code" != "200" ]; then
        echo "ERROR: llama-server not responding (HTTP $code)"
        echo "Start it with: llm start"
        exit 1
    fi
}

# ── Benchmark Modes ──────────────────────────────────────────

run_quick() {
    echo "=== Quick Benchmark ==="
    echo "  Server: localhost:${PORT:-8080}"
    echo ""

    echo "  Warmup..."
    bench_request 16 "Hi" "warmup" > /dev/null 2>&1 || true
    echo ""

    echo "  Single-request throughput:"
    bench_request 64  "Write a haiku about programming." "64 tokens"
    bench_request 256 "Explain hash tables in detail." "256 tokens"
    bench_request 512 "Write a merge sort implementation in Python with comments." "512 tokens"
}

run_full() {
    run_quick
    echo ""
    echo "  Concurrent throughput:"
    bench_concurrent 2 128
    bench_concurrent 4 128
    bench_concurrent 8 128
    echo ""
    echo "  Long generation:"
    bench_request 1024 "Write a comprehensive guide to binary search trees." "1024 tokens"
}

run_compare() {
    echo "=== Engine Comparison ==="
    echo ""

    local ik_bin="$HOME/ik_llama.cpp/build/bin/llama-server"
    local up_bin="$HOME/llama.cpp/build/bin/llama-server"

    if [ ! -x "$ik_bin" ] || [ ! -x "$up_bin" ]; then
        echo "ERROR: Both engines must be built."
        echo "Run: scripts/build-engines.sh"
        exit 1
    fi

    echo "This test requires manual setup:"
    echo "  1. Start ik_llama.cpp on port 8080"
    echo "  2. Start llama.cpp on port 8081"
    echo "  3. Re-run this benchmark"
    echo ""
    echo "Or use 'llm bench quick' to benchmark the active server."
}

# ── Main ─────────────────────────────────────────────────────

check_server

case "${1:-quick}" in
    quick)   run_quick ;;
    full)    run_full ;;
    compare) run_compare ;;
    *)       echo "Usage: bench.sh {quick|full|compare}"; exit 1 ;;
esac

echo ""
echo "Done."
