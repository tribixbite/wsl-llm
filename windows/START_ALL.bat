@echo off
REM Autostart WSL LLM services on Windows boot
REM Add this to Windows Task Scheduler: run at logon, or put in shell:startup
echo Starting WSL LLM services...

REM Ensure WSL is running (launches systemd which starts enabled services)
wsl -e bash -c "echo WSL started, systemd services are auto-starting..."

REM Wait for services to initialize
echo Waiting 30 seconds for model to load...
timeout /t 30 /nobreak

REM Health check
echo.
echo Checking services...
wsl -e bash -c "systemctl status llama-server --no-pager | head -5"
echo.
wsl -e bash -c "systemctl status cloudflared --no-pager | head -5"
echo.
wsl -e bash -c "systemctl status litellm --no-pager | head -5"
echo.

REM Test inference
echo Testing inference...
wsl -e bash -c "curl -s http://localhost:8080/health"
echo.
echo.
echo Services should be running:
echo   - llama-server: http://localhost:8080 (direct)
echo   - LiteLLM:      http://localhost:4000 (proxy + admin UI)
echo   - llm.pet:      https://llm.pet (public)
echo   - Admin UI:     http://localhost:4000/ui
echo.
echo LiteLLM Admin: username=admin, password=sk-litellm-admin-9f8e7d6c5b4a
echo.
pause
