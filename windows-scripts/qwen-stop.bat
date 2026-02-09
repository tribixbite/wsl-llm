@echo off
REM Stop Qwen3-Coder-Next server in WSL
echo Stopping Qwen3-Coder-Next server...
wsl bash -l -c "pkill -f 'llama-server.*Qwen3-Coder-Next'"
echo Server stopped!
pause
