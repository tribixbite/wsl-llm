# Undo old WSL-LLM autostart setup (port forwarding method)
# Run as Administrator in PowerShell

#Requires -RunAsAdministrator

Write-Host "=== Undoing old WSL-LLM autostart ===" -ForegroundColor Cyan

# 1. Remove Task Scheduler entries
$tasks = @("WSL-LLM-Autostart", "WSL-LLM-PortForward")
foreach ($task in $tasks) {
    $existing = Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue
    if ($existing) {
        Unregister-ScheduledTask -TaskName $task -Confirm:$false
        Write-Host "[OK] Removed task: $task" -ForegroundColor Green
    } else {
        Write-Host "[--] Task not found: $task" -ForegroundColor Gray
    }
}

# 2. Remove port forwarding rules
$proxyCount = (netsh interface portproxy show v4tov4 | Select-String "\d+\.\d+").Count
if ($proxyCount -gt 0) {
    netsh interface portproxy reset | Out-Null
    Write-Host "[OK] Removed $proxyCount port forwarding rules" -ForegroundColor Green
} else {
    Write-Host "[--] No port forwarding rules found" -ForegroundColor Gray
}

# 3. Remove old firewall rule
$fwRule = Get-NetFirewallRule -DisplayName "WSL LLM Services" -ErrorAction SilentlyContinue
if ($fwRule) {
    Remove-NetFirewallRule -DisplayName "WSL LLM Services"
    Write-Host "[OK] Removed firewall rule: WSL LLM Services" -ForegroundColor Green
} else {
    Write-Host "[--] Firewall rule not found" -ForegroundColor Gray
}

# Also remove per-port rules from setup-windows.ps1 if they exist
foreach ($name in @("WSL-LLM-Server", "WSL-LLM-LiteLLM", "WSL-LLM-WebUI")) {
    $rule = Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue
    if ($rule) {
        Remove-NetFirewallRule -DisplayName $name
        Write-Host "[OK] Removed firewall rule: $name" -ForegroundColor Green
    }
}

# 4. Remove generated port forward script
$scriptPath = "$env:USERPROFILE\wsl-llm-portforward.ps1"
if (Test-Path $scriptPath) {
    Remove-Item $scriptPath -Force
    Write-Host "[OK] Removed $scriptPath" -ForegroundColor Green
} else {
    Write-Host "[--] Port forward script not found" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Done. Old autostart removed." -ForegroundColor Green
Write-Host "Now run: .\setup-autostart.ps1" -ForegroundColor Yellow
