#!/bin/bash
# vLLM Configuration Benchmark Script
# Tests multiple configurations and records results

set -e

export HF_HOME=/mnt/c/Users/Will/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=/home/matilda/bench_env/lib/python3.10/site-packages:$PYTHONPATH

VLLM_CMD="python -m vllm.entrypoints.openai.api_server"
MODEL="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"
RESULTS_FILE="/home/matilda/git/wsl-llm/vllm_results.log"

source ~/bench_env/bin/activate

bench_request() {
    local max_tokens=$1
    local label=$2
    python -c "
import requests, time
start = time.perf_counter()
r = requests.post('http://localhost:8080/v1/chat/completions', json={
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': '/no_think Write a comprehensive guide to Python decorators with examples.'}],
    'max_tokens': $max_tokens,
    'temperature': 0.7,
    'top_p': 0.8,
})
r.raise_for_status()
elapsed = time.perf_counter() - start
toks = r.json()['usage']['completion_tokens']
tps = toks / elapsed
print(f'$label: {toks} tok in {elapsed:.2f}s = {tps:.1f} t/s')
" 2>&1
}

wait_for_server() {
    echo "Waiting for server to start..."
    for i in $(seq 1 120); do
        if curl -s http://localhost:8080/v1/models > /dev/null 2>&1; then
            echo "Server ready!"
            return 0
        fi
        sleep 5
    done
    echo "Server failed to start!"
    return 1
}

run_config() {
    local config_name="$1"
    shift
    local extra_args="$@"

    echo ""
    echo "=========================================="
    echo "CONFIG: $config_name"
    echo "Args: $extra_args"
    echo "=========================================="

    # Start server
    $VLLM_CMD \
        --model $MODEL \
        --quantization moe_wna16 \
        --host 0.0.0.0 --port 8080 \
        --language-model-only \
        $extra_args \
        > /tmp/vllm_server.log 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server; then
        echo "FAILED to start: $config_name"
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        echo "$config_name | FAILED | - | -" >> "$RESULTS_FILE"
        return 1
    fi

    # GPU memory
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')
    echo "GPU Memory: $GPU_MEM"

    # Warmup
    echo "Warming up..."
    python -c "
import requests
requests.post('http://localhost:8080/v1/chat/completions', json={
    'model': '$MODEL',
    'messages': [{'role': 'user', 'content': '/no_think Say hello'}],
    'max_tokens': 32, 'temperature': 0.7,
})
" 2>/dev/null

    # Benchmark
    echo "Running benchmarks..."
    RESULT_128=$(bench_request 128 "128tok")
    RESULT_512=$(bench_request 512 "512tok")
    RESULT_1024=$(bench_request 1024 "1024tok")

    echo "$RESULT_128"
    echo "$RESULT_512"
    echo "$RESULT_1024"

    # Extract TPS from 512tok result
    TPS_512=$(echo "$RESULT_512" | grep -oP '[\d.]+ t/s' | head -1)

    echo "$config_name | $TPS_512 | $GPU_MEM | $extra_args" >> "$RESULTS_FILE"

    # Kill server
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    sleep 5
}

# Clear results
echo "Config | TPS (512tok) | GPU Mem | Args" > "$RESULTS_FILE"
echo "---" >> "$RESULTS_FILE"

# Config 1: Baseline (TP=1, ctx=8192)
run_config "TP1_ctx8k" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90

# Config 2: TP=1, ctx=16384
run_config "TP1_ctx16k" \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90

# Config 3: TP=1, ctx=32768
run_config "TP1_ctx32k" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95

# Config 4: TP=1, fp8 KV cache, ctx=32768
run_config "TP1_ctx32k_fp8kv" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8

# Config 5: TP=2, ctx=8192
run_config "TP2_ctx8k" \
    --max-model-len 8192 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90

# Config 6: TP=2, ctx=32768
run_config "TP2_ctx32k" \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90

# Config 7: TP=1, prefix caching
run_config "TP1_ctx8k_prefix" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enable-prefix-caching

# Config 8: TP=1, enforce eager (no CUDA graphs)
run_config "TP1_ctx8k_eager" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager

echo ""
echo "=========================================="
echo "ALL RESULTS:"
echo "=========================================="
cat "$RESULTS_FILE"
