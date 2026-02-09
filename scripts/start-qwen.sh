#!/bin/bash
# Start Qwen3-Coder-Next server

PID_FILE=~/qwen-server.pid
LOG_FILE=~/qwen-server.log

# Check if already running
if [ -f $PID_FILE ]; then
    PID=$(cat $PID_FILE)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Server already running (PID: $PID)"
        exit 1
    fi
fi

echo "Starting Qwen3-Coder-Next server..."
nohup /home/matilda/llama.cpp/build/bin/llama-server \
    --model /home/matilda/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q3_K_XL.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 131072 \
    --n-gpu-layers 999 \
    --jinja \
    > $LOG_FILE 2>&1 &

echo $! > $PID_FILE
echo "Server started! PID: $(cat $PID_FILE)"
echo "Logs: tail -f ~/qwen-server.log"
