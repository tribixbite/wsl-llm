#!/bin/bash
# Wrapper script for llama-server — reads ~/llama-server.conf
# Called by systemd service. Edit the conf file, then: llm restart

CONF="${LLAMA_SERVER_CONF:-$HOME/llama-server.conf}"

if [ ! -f "$CONF" ]; then
    echo "ERROR: Config not found: $CONF" >&2
    exit 1
fi

source "$CONF"

# Export CUDA device restriction if set
[ -n "$CUDA_VISIBLE_DEVICES" ] && export CUDA_VISIBLE_DEVICES

# Determine engine binary
# Priority: LLAMA_BIN env/conf > Madreag turboquant fork > ik_llama.cpp > upstream llama.cpp
if [ -z "${LLAMA_BIN:-}" ]; then
    for candidate in \
        "$HOME/llama-cpp-turboquant/llama-server" \
        "$HOME/ik_llama.cpp/build/bin/llama-server" \
        "$HOME/llama.cpp/build/bin/llama-server"
    do
        if [ -x "$candidate" ]; then
            LLAMA_BIN="$candidate"
            break
        fi
    done
fi

if [ -z "${LLAMA_BIN:-}" ] || [ ! -x "$LLAMA_BIN" ]; then
    echo "ERROR: No llama-server binary found" >&2
    exit 1
fi

# GPU offloading: "fit" = auto-fit, number = manual -ngl
if [ "$GPU_LAYERS" = "fit" ]; then
    GPU_ARGS="--fit"
elif [ -n "$GPU_LAYERS" ] && [ "$GPU_LAYERS" != "0" ]; then
    GPU_ARGS="-ngl $GPU_LAYERS"
else
    GPU_ARGS=""
fi

# Kill any orphaned llama-server processes
pgrep -x llama-server | xargs -r kill -9 2>/dev/null
sleep 2

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --alias "$ALIAS" \
    -c "$CONTEXT_SIZE" \
    -np "$NUM_SLOTS" \
    $GPU_ARGS \
    -fa "$FLASH_ATTENTION" \
    --cache-type-k "$KV_TYPE_K" \
    --cache-type-v "$KV_TYPE_V" \
    --temp "$TEMP" \
    --top-p "$TOP_P" \
    --top-k "$TOP_K" \
    --min-p "$MIN_P" \
    --reasoning-budget "$REASONING_BUDGET" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    $EXTRA_FLAGS
