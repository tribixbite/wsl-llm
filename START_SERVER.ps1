# Start Qwen Server - Run this in PowerShell
Write-Host "Starting Qwen3-Coder-Next server..." -ForegroundColor Cyan

# Start server in WSL
wsl -e bash -l -c "nohup /home/matilda/llama.cpp/build/bin/llama-server --model /home/matilda/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q3_K_XL.gguf --alias 'unsloth/Qwen3-Coder-Next' --fit on --seed 3407 --temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40 --host 0.0.0.0 --port 8080 --ctx-size 131072 --jinja --n-gpu-layers 999 --log-verbose --log-prefix --log-timestamps > /home/matilda/qwen-server.log 2>&1 &"

Write-Host "`nServer starting... waiting 15 seconds for model to load..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host "`nTesting server..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -UseBasicParsing
    Write-Host "SUCCESS! Server is running" -ForegroundColor Green
    Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
    Write-Host "Response: $($response.Content)" -ForegroundColor Green
    Write-Host "`nAccess URLs:" -ForegroundColor Cyan
    Write-Host "  - Localhost: http://localhost:8080" -ForegroundColor White
    $wslIP = (wsl hostname -I).Trim()
    Write-Host "  - LAN: http://$wslIP:8080" -ForegroundColor White
} catch {
    Write-Host "Server not responding yet. Check logs:" -ForegroundColor Red
    Write-Host "  wsl cat /home/matilda/qwen-server.log" -ForegroundColor Yellow
}

Write-Host "`nTo check status: wsl ps aux | grep llama-server" -ForegroundColor Cyan
Write-Host "To view logs: wsl tail -f /home/matilda/qwen-server.log" -ForegroundColor Cyan
Write-Host "To stop: wsl pkill llama-server" -ForegroundColor Cyan
