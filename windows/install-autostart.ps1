# Register Qwen3.8-27B to start automatically on Windows, with no WSL involved.
#
#   .\install-autostart.ps1                 # run at logon, MTP + vision
#   .\install-autostart.ps1 -Mode fast      # MTP only, 32k ctx
#   .\install-autostart.ps1 -NoBattery      # don't run on battery (saves the battery)
#   .\install-autostart.ps1 -Status
#   .\install-autostart.ps1 -Uninstall
#
# Why a Scheduled Task rather than the Startup folder:
#   * no console window flashing up and sitting in the taskbar
#   * restarts the server automatically if it dies (this box bugchecks — see
#     docs/QWEN38_27B_LEGION_BENCHMARKS.md §10)
#   * a start delay, so the NVIDIA driver is settled before we grab ~14 GB
#   * survives without keeping a shell open, and stops cleanly on logoff
#
# Runs as the logged-on user in the interactive session on purpose. A task set to
# "run whether user is logged on or not" executes in session 0, where CUDA under
# WDDM is unreliable — this needs the interactive session.
#
# Battery: runs on battery and keeps running if you unplug (default). Note the
# GPU is power-limited on battery, so throughput drops well below the 87 t/s
# figure and the battery drains quickly under sustained load. Pass -NoBattery to
# get the conservative behaviour instead.
#
# The task never gates on the GPU's power limit — it starts and serves at
# whatever TGP the machine is currently in. Performance mode (175 W) is worth
# +32% decode, but anything lower simply runs slower, it does not block.
#
# No admin rights required.

[CmdletBinding()]
param(
    [ValidateSet('both', 'fast', 'vision')]
    [string]$Mode = 'both',
    [int]$Port = 8080,
    [int]$DelaySeconds = 30,
    [string]$Root = 'C:\llm',
    [string]$TaskName = 'Qwen38Server',
    [switch]$NoBattery,
    [switch]$Uninstall,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'
$script  = Join-Path $Root 'start-qwen38.ps1'
$logDir  = Join-Path $Root 'logs'
$logFile = Join-Path $logDir 'server.log'

if ($Status) {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $t) { Write-Host "task '$TaskName' is NOT registered"; exit 0 }
    $i = Get-ScheduledTaskInfo -TaskName $TaskName
    Write-Host "task      : $TaskName  [$($t.State)]"
    Write-Host "action    : $($t.Actions[0].Execute) $($t.Actions[0].Arguments)"
    Write-Host "last run  : $($i.LastRunTime)  result=0x$('{0:X}' -f $i.LastTaskResult)"
    Write-Host "next run  : $($i.NextRunTime)"
    $up = Test-NetConnection 127.0.0.1 -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue
    Write-Host "endpoint  : http://127.0.0.1:$Port/v1  ->  $(if ($up) {'UP'} else {'down'})"
    if (Test-Path $logFile) { Write-Host "log       : $logFile ($([math]::Round((Get-Item $logFile).Length/1KB)) KB)" }
    exit 0
}

if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "unregistered '$TaskName'"
    } else { Write-Host "task '$TaskName' was not registered" }
    exit 0
}

if (-not (Test-Path $script)) {
    throw "missing $script - copy windows/start-qwen38.ps1 there first (see README)"
}
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# Redirect through -Command so llama-server's stdout/stderr land in a log we can read.
$inner = "& '$script' -Mode $Mode -Port $Port *>> '$logFile'"
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command `"$inner`""

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT${DelaySeconds}S"

$allowBattery = -not $NoBattery
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries:$allowBattery `
    -DontStopIfGoingOnBatteries:$allowBattery `
    -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew
$settings.Hidden = $true

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Qwen3.8-27B OpenAI-compatible server ($Mode)" `
    -Force | Out-Null

Write-Host ""
Write-Host "  registered '$TaskName'"
Write-Host "    mode      : $Mode  (port $Port)"
Write-Host "    starts    : at logon, ${DelaySeconds}s delay"
Write-Host "    battery   : $(if ($allowBattery) {'runs on battery, keeps running when unplugged'} else {'will NOT start on battery, stops if unplugged'})"
Write-Host "    power     : starts at any GPU power limit (175 W is just faster)"
Write-Host "    restarts  : up to 3x, 1 min apart"
Write-Host "    log       : $logFile"
Write-Host ""
Write-Host "  start now : Start-ScheduledTask -TaskName $TaskName"
Write-Host "  status    : .\install-autostart.ps1 -Status"
Write-Host "  remove    : .\install-autostart.ps1 -Uninstall"
Write-Host ""
