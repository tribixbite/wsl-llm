# WSL LLM Services Autostart Script
# Add to Windows Task Scheduler: run at logon
# Or place shortcut in shell:startup

Write-Host "Starting WSL LLM services..." -ForegroundColor Cyan

# Start WSL (triggers systemd, which starts enabled services)
wsl -e bash -c "echo 'WSL started. Systemd services auto-starting...'"

Write-Host "`nWaiting 30 seconds for model to load..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check services
Write-Host "`nService Status:" -ForegroundColor Cyan
wsl -e bash -c "systemctl is-active llama-server cloudflared litellm"

# Health checks
Write-Host "`nHealth Checks:" -ForegroundColor Cyan
try {
    $health = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing -TimeoutSec 10
    Write-Host "  llama-server: OK ($($health.Content))" -ForegroundColor Green
} catch {
    Write-Host "  llama-server: FAILED" -ForegroundColor Red
}

try {
    $health = Invoke-WebRequest -Uri "http://localhost:4000/health" -UseBasicParsing -TimeoutSec 10 -Headers @{"Authorization"="Bearer sk-litellm-admin-9f8e7d6c5b4a"}
    Write-Host "  LiteLLM: OK" -ForegroundColor Green
} catch {
    Write-Host "  LiteLLM: FAILED (may still be running migrations)" -ForegroundColor Yellow
}

try {
    $health = Invoke-WebRequest -Uri "https://llm.pet/health" -UseBasicParsing -TimeoutSec 10
    Write-Host "  llm.pet: OK" -ForegroundColor Green
} catch {
    Write-Host "  llm.pet: FAILED" -ForegroundColor Red
}

Write-Host "`nEndpoints:" -ForegroundColor Cyan
Write-Host "  Direct API:  http://localhost:8080" -ForegroundColor White
Write-Host "  LiteLLM API: http://localhost:4000" -ForegroundColor White
Write-Host "  Admin UI:    http://localhost:4000/ui" -ForegroundColor White
Write-Host "  Public:      https://llm.pet" -ForegroundColor White
Write-Host "`nLiteLLM Admin: username=admin, password=sk-litellm-admin-9f8e7d6c5b4a" -ForegroundColor Yellow
Write-Host "`nTo manage services:" -ForegroundColor Cyan
Write-Host "  wsl sudo systemctl status llama-server" -ForegroundColor White
Write-Host "  wsl sudo systemctl restart llama-server" -ForegroundColor White
Write-Host "  wsl journalctl -u llama-server -f" -ForegroundColor White
