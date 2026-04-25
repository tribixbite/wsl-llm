# WSL-LLM Windows-side setup
# Run as Administrator in PowerShell
#
# Sets up:
# 1. TDR timeout increase (prevents CUDA crash on long GPU operations)
# 2. ASPM disable reminder
# 3. Windows Firewall rules for LLM services

#Requires -RunAsAdministrator

Write-Host "WSL-LLM Windows Setup" -ForegroundColor Cyan
Write-Host ""

# ── TDR Timeout ──────────────────────────────────────────────
Write-Host "=== TDR Timeout ===" -ForegroundColor Yellow

$tdrPath = "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
$currentTdr = (Get-ItemProperty -Path $tdrPath -Name "TdrDelay" -ErrorAction SilentlyContinue).TdrDelay

if ($currentTdr -ge 60) {
    Write-Host "  TdrDelay already set to $currentTdr seconds"
} else {
    New-ItemProperty -Path $tdrPath -Name "TdrDelay" -PropertyType DWord -Value 60 -Force | Out-Null
    New-ItemProperty -Path $tdrPath -Name "TdrDdiDelay" -PropertyType DWord -Value 60 -Force | Out-Null
    Write-Host "  TdrDelay set to 60 seconds (was: $($currentTdr ?? 'default 2'))"
    Write-Host "  TdrDdiDelay set to 60 seconds"
    Write-Host "  NOTE: Reboot required for TDR changes to take effect" -ForegroundColor Yellow
}

# ── Firewall Rules ───────────────────────────────────────────
Write-Host ""
Write-Host "=== Firewall Rules ===" -ForegroundColor Yellow

$ports = @(
    @{ Name = "WSL-LLM-Server";  Port = 8080; Desc = "llama-server" },
    @{ Name = "WSL-LLM-LiteLLM"; Port = 4000; Desc = "LiteLLM proxy" },
    @{ Name = "WSL-LLM-WebUI";   Port = 3000; Desc = "Open WebUI" }
)

foreach ($p in $ports) {
    $existing = Get-NetFirewallRule -DisplayName $p.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  $($p.Name) (port $($p.Port)) - already exists"
    } else {
        New-NetFirewallRule -DisplayName $p.Name `
            -Direction Inbound -Protocol TCP -LocalPort $p.Port `
            -Action Allow -Profile Private,Domain `
            -Description "Allow $($p.Desc) access from LAN" | Out-Null
        Write-Host "  $($p.Name) (port $($p.Port)) - created"
    }
}

# ── BIOS Reminder ────────────────────────────────────────────
Write-Host ""
Write-Host "=== BIOS Recommendations ===" -ForegroundColor Yellow
Write-Host "  For maximum stability with dual-GPU inference:"
Write-Host "  1. Disable ASPM (PCI Subsystem Settings -> ASPM -> Disabled)"
Write-Host "  2. Enable Above 4G Decoding + Resizable BAR"
Write-Host "  3. Set Power Supply Idle Control to Typical Current Idle"
Write-Host "  4. Consider disabling DF C-States"
Write-Host ""

# ── GPU Status ───────────────────────────────────────────────
Write-Host "=== GPU Status ===" -ForegroundColor Yellow
& nvidia-smi --query-gpu=index,name,pcie.link.gen.current,pcie.link.width.current,memory.total --format=csv 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  nvidia-smi not found"
}

Write-Host ""
Write-Host "Done. Reboot if TDR values were changed." -ForegroundColor Green
