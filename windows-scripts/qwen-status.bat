@echo off
REM Check Qwen3-Coder-Next server status
echo Checking server status...
echo.
echo === Process Status ===
wsl bash -l -c "ps aux | grep -E 'llama-server.*Qwen3' | grep -v grep || echo 'Server not running'"
echo.
echo === GPU Memory Usage ===
wsl bash -l -c "nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv"
echo.
echo === Server Health Check ===
curl -s http://localhost:8080/health 2>nul && echo Server is responding! || echo Server not responding
echo.
pause
