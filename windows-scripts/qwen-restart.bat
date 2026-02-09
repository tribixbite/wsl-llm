@echo off
REM Restart Qwen3-Coder-Next server
echo Restarting Qwen3-Coder-Next server...
call qwen-stop.bat
timeout /t 3 /nobreak >nul
call qwen-start.bat
