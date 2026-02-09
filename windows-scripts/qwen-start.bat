@echo off
REM Start Qwen3-Coder-Next server in WSL
echo Starting Qwen3-Coder-Next server...
wsl bash -l -c "cd ~ && nohup /home/matilda/llama.cpp/build/bin/llama-server --model /home/matilda/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q3_K_XL.gguf --host 0.0.0.0 --port 8080 --ctx-size 131072 --n-gpu-layers 999 --jinja > ~/qwen-server.log 2>&1 &"
echo Server started! Access at http://localhost:8080
echo Logs: wsl cat ~/qwen-server.log
pause
