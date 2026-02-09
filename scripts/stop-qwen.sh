#!/bin/bash
# Stop Qwen3-Coder-Next server

PID_FILE=~/qwen-server.pid

if [ ! -f $PID_FILE ]; then
    echo "Server not running (no PID file)"
    pkill -f 'llama-server.*Qwen3-Coder-Next' && echo "Killed stray processes"
    exit 0
fi

PID=$(cat $PID_FILE)
if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping server (PID: $PID)..."
    kill $PID
    sleep 2
    if ps -p $PID > /dev/null 2>&1; then
        echo "Force killing..."
        kill -9 $PID
    fi
    rm $PID_FILE
    echo "Server stopped"
else
    echo "Server not running (stale PID file)"
    rm $PID_FILE
    pkill -f 'llama-server.*Qwen3-Coder-Next' && echo "Killed stray processes"
fi
