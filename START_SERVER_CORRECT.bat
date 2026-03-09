@echo off
REM Start Qwen3-Coder-Next with proper Unsloth flags
echo Starting Qwen3-Coder-Next server with Unsloth configuration...

wsl bash -l -c "nohup /home/matilda/llama.cpp/build/bin/llama-server --model /home/matilda/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q3_K_XL.gguf --alias 'unsloth/Qwen3-Coder-Next' --fit on --seed 3407 --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40 --host 0.0.0.0 --port 8080 --ctx-size 131072 --jinja --n-gpu-layers 999 --log-verbose --log-prefix --log-timestamps > /home/matilda/qwen-server.log 2>&1 &"

echo.
echo Server starting with Unsloth-recommended settings:
echo   - Memory optimization: --fit on
echo   - Temperature: 1.0
echo   - Top-P: 0.95, Top-K: 40, Min-P: 0.01
echo   - Context: 128k tokens
echo.
echo Waiting 15 seconds for model to load...
timeout /t 15 /nobreak

echo.
echo Testing server...
curl -s http://localhost:8080/health

echo.
echo Server should be running at:
echo   http://localhost:8080
echo.
echo View logs: wsl cat /home/matilda/qwen-server.log
echo.
pause
