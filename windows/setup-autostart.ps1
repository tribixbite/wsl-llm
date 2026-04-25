# WSL LLM - Autostart Setup (mirrored networking)
# Run this script ONCE as Administrator in PowerShell
#
# Sets up:
# 1. .wslconfig with mirrored networking (no port forwarding needed)
# 2. Task Scheduler to keep WSL alive on boot
# 3. Firewall rules for LAN access

#Requires -RunAsAdministrator

Write-Host "=== WSL LLM Autostart Setup ===" -ForegroundColor Cyan
Write-Host ""

# ── 1. Install .wslconfig with mirrored networking ──────────
Write-Host "=== Mirrored Networking ===" -ForegroundColor Yellow

$wslconfigPath = "$env:USERPROFILE\.wslconfig"
$wslconfigContent = @"
[wsl2]
# Mirrored networking: WSL shares Windows network interfaces directly
# No port forwarding needed — LAN devices connect to Windows IP
networkingMode=mirrored
dnsTunneling=true
autoProxy=true

# Memory limit (adjust as needed)
# memory=32GB

# Processor limit (adjust as needed)
# processors=12
"@

if (Test-Path $wslconfigPath) {
    $existing = Get-Content $wslconfigPath -Raw
    if ($existing -match "networkingMode\s*=\s*mirrored") {
        Write-Host "  .wslconfig already has mirrored networking" -ForegroundColor Gray
    } else {
        # Back up existing
        Copy-Item $wslconfigPath "$wslconfigPath.bak" -Force
        Write-Host "  Backed up existing .wslconfig to .wslconfig.bak" -ForegroundColor Yellow
        $wslconfigContent | Out-File -FilePath $wslconfigPath -Encoding UTF8 -Force
        Write-Host "  Installed .wslconfig with mirrored networking" -ForegroundColor Green
        $needsRestart = $true
    }
} else {
    $wslconfigContent | Out-File -FilePath $wslconfigPath -Encoding UTF8 -Force
    Write-Host "  Installed .wslconfig with mirrored networking" -ForegroundColor Green
    $needsRestart = $true
}

# ── 2. Task Scheduler: Keep WSL alive on boot ──────────────
Write-Host ""
Write-Host "=== Task Scheduler ===" -ForegroundColor Yellow

$taskName = "WSL-LLM-Autostart"
$action = New-ScheduledTaskAction -Execute "wsl.exe" -Argument "-e bash -c 'sleep infinity'"
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Highest

Register-ScheduledTask `
    -TaskName $taskName `
    -Description "Start WSL2 on boot for LLM services (llama-server, litellm, cloudflared, open-webui)" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force | Out-Null

Write-Host "  $taskName task registered" -ForegroundColor Green

# ── 3. Firewall Rules ──────────────────────────────────────
Write-Host ""
Write-Host "=== Firewall Rules ===" -ForegroundColor Yellow

# With mirrored networking, WSL binds directly on the Windows network
# interfaces, so we need inbound rules on the Hyper-V firewall
$ports = @(
    @{ Name = "WSL-LLM-Server";  Port = 8080; Desc = "llama-server" },
    @{ Name = "WSL-LLM-LiteLLM"; Port = 4000; Desc = "LiteLLM proxy" },
    @{ Name = "WSL-LLM-WebUI";   Port = 3000; Desc = "Open WebUI" }
)

foreach ($p in $ports) {
    $existing = Get-NetFirewallRule -DisplayName $p.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  $($p.Name) (port $($p.Port)) - already exists" -ForegroundColor Gray
    } else {
        New-NetFirewallRule -DisplayName $p.Name `
            -Direction Inbound -Protocol TCP -LocalPort $p.Port `
            -Action Allow -Profile Private,Domain `
            -Description "Allow $($p.Desc) access from LAN" | Out-Null
        Write-Host "  $($p.Name) (port $($p.Port)) - created" -ForegroundColor Green
    }
}

# Also add Hyper-V firewall rule for mirrored mode
# (Windows 11 may block WSL inbound even with standard firewall rules)
try {
    Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow -ErrorAction Stop | Out-Null
    Write-Host "  Hyper-V firewall: allowed inbound for WSL" -ForegroundColor Green
} catch {
    Write-Host "  Hyper-V firewall: skipped (not applicable or already set)" -ForegroundColor Gray
}

# ── Summary ─────────────────────────────────────────────────
$windowsIP = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.InterfaceAlias -notlike "*Loopback*" -and
                   $_.InterfaceAlias -notlike "*WSL*" -and
                   $_.InterfaceAlias -notlike "*vEthernet*" } |
    Select-Object -First 1).IPAddress

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "How it works:" -ForegroundColor Cyan
Write-Host "  Mirrored networking = WSL shares Windows network directly" -ForegroundColor White
Write-Host "  No port forwarding needed. Services bind on Windows IP." -ForegroundColor White
Write-Host ""
Write-Host "On boot:" -ForegroundColor Cyan
Write-Host "  1. Task Scheduler starts WSL" -ForegroundColor White
Write-Host "  2. systemd starts llama-server, litellm, cloudflared, open-webui" -ForegroundColor White
Write-Host ""
Write-Host "Access from this PC:" -ForegroundColor Cyan
Write-Host "  http://localhost:3000      Open WebUI" -ForegroundColor White
Write-Host "  http://localhost:4000/ui   LiteLLM Admin" -ForegroundColor White
Write-Host ""
Write-Host "Access from LAN:" -ForegroundColor Cyan
Write-Host "  http://${windowsIP}:3000   Open WebUI" -ForegroundColor White
Write-Host "  http://${windowsIP}:4000   LiteLLM API" -ForegroundColor White
Write-Host "  http://${windowsIP}:8080   llama-server direct" -ForegroundColor White

if ($needsRestart) {
    Write-Host ""
    Write-Host "NOTE: WSL restart required for mirrored networking." -ForegroundColor Yellow
    Write-Host "Run: wsl --shutdown   (then reopen your terminal)" -ForegroundColor Yellow
}
