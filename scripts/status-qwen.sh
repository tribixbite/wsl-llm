#!/bin/bash
# Check Qwen3-Coder-Next server status

PID_FILE=~/qwen-server.pid

echo "=== Server Status ==="
if [ -f $PID_FILE ]; then
    PID=$(cat $PID_FILE)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Status: RUNNING (PID: $PID)"
        echo "Uptime: $(ps -p $PID -o etime= | xargs)"
    else
        echo "Status: STOPPED (stale PID file)"
    fi
else
    echo "Status: STOPPED"
fi

echo ""
echo "=== GPU Memory Usage ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

echo ""
echo "=== API Health ==="
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "API: HEALTHY"
    echo "URL: http://localhost:8080"
    echo "WSL IP: $(hostname -I | awk '{print $1}'):8080"
else
    echo "API: NOT RESPONDING"
fi

echo ""
echo "=== Recent Logs (last 10 lines) ==="
if [ -f ~/qwen-server.log ]; then
    tail -n 10 ~/qwen-server.log
else
    echo "No log file found"
fi
