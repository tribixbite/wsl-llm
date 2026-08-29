# Start Qwen3.8-27B on native Windows: fast speculative decoding AND vision,
# behind an OpenAI-compatible endpoint.
#
#   .\start-qwen38.ps1                 # MTP + vision, 16k ctx   (default)
#   .\start-qwen38.ps1 -Mode fast      # MTP only, 32k ctx, no images
#   .\start-qwen38.ps1 -Mode vision    # vision only, 16k ctx, no MTP
#   .\start-qwen38.ps1 -Port 9000 -Ctx 8192
#
# Why the flags are what they are (all measured on this box — RTX 5080 Laptop,
# 16 GB, sm_120; see docs/QWEN38_27B_LEGION_BENCHMARKS.md):
#
#   --parallel 1          MANDATORY. The default of 4 slots gives every slot its
#                         own DeltaNet recurrent state and pushes peak VRAM from
#                         13.4 to 15.9 GiB. There is no OOM guardrail: the driver
#                         silently evicts the weights to system RAM and decode
#                         collapses ~700x (39.8 -> 0.04 t/s) while /health still
#                         reports ok.
#   --spec-type draft-mtp The MTP draft head is the single biggest speed lever:
#                         1.89x overall, 2.14x on code (39.8 -> ~75 t/s).
#   --no-mmproj-offload   Keeps the 0.86 GiB vision projector on the CPU. This is
#                         what lets vision and MTP coexist in 16 GB at all. The
#                         projector runs once per image, so the cost is a small
#                         one-off on image prefill, not on decode.
#   --reasoning-effort medium
#                         Thinking is worth roughly 2x on coding (58.8% vs 38.2%
#                         pass@2). The default 'xhigh' costs ~220 s/exercise.
#   -ctk/-ctv q8_0        KV dtype barely affects speed here (39.9 f16 vs 38.7
#                         q4_0), so pick it for headroom, not throughput.
#
# Verify with:  Invoke-RestMethod http://127.0.0.1:8080/health

[CmdletBinding()]
param(
    [ValidateSet('both', 'fast', 'vision')]
    [string]$Mode = 'both',
    [int]$Port = 8080,
    [int]$Ctx = 0,                       # 0 = pick a safe default for the mode
    [string]$Root = 'C:\llm',
    [switch]$NoThinking
)

$ErrorActionPreference = 'Stop'

$exe    = Join-Path $Root 'bin\llama-server.exe'
$model  = Join-Path $Root 'models\Qwen3.8-27B-UD-Q3_K_XL.gguf'
$mtp    = Join-Path $Root 'models\MTP\mtp-Qwen3.8-27B-Q4_0.gguf'
$mmproj = Join-Path $Root 'models\mmproj-F16.gguf'

foreach ($f in @($exe, $model)) {
    if (-not (Test-Path $f)) { throw "missing: $f" }
}

if ($Ctx -eq 0) { $Ctx = if ($Mode -eq 'fast') { 32768 } else { 16384 } }

$srvArgs = @(
    '-m', $model,
    '-ngl', '99',
    '-fa', 'on',
    '-c', "$Ctx",
    '-ctk', 'q8_0', '-ctv', 'q8_0',
    '--parallel', '1',                   # see header — do not raise this
    '--host', '127.0.0.1',
    '--port', "$Port",
    '--jinja'
)

if ($Mode -in @('both', 'fast')) {
    if (-not (Test-Path $mtp)) { throw "missing MTP draft head: $mtp" }
    $srvArgs += @('--spec-type', 'draft-mtp', '-md', $mtp)
}
if ($Mode -in @('both', 'vision')) {
    if (-not (Test-Path $mmproj)) { throw "missing vision projector: $mmproj" }
    $srvArgs += @('--mmproj', $mmproj)
    # Only keep the projector off the GPU when we also need room for MTP.
    if ($Mode -eq 'both') { $srvArgs += '--no-mmproj-offload' }
}
if ($NoThinking) { $srvArgs += @('--reasoning-budget', '0') }
else             { $srvArgs += @('--reasoning-effort', 'medium') }

$expect = switch ($Mode) {
    'both'   { 'MTP + vision  (~75 t/s text, images supported)' }
    'fast'   { 'MTP only      (~75 t/s, no images)' }
    'vision' { 'vision only   (~38 t/s, images supported)' }
}

Write-Host ""
Write-Host "  Qwen3.8-27B  ->  $expect"
Write-Host "  context $Ctx | OpenAI endpoint: http://127.0.0.1:$Port/v1"
Write-Host ""
& nvidia-smi --query-gpu=name,enforced.power.limit,memory.used --format=csv,noheader
Write-Host "  (if the power limit is not 175 W, press Fn+Q for Performance mode:"
Write-Host "   it is worth +32% decode and +42% prefill)"
Write-Host ""

& $exe @srvArgs
