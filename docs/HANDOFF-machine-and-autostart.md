# Handoff — Legion Pro 7 machine state, power config, and LLM autostart

**Written:** 2026-09-01
**Machine:** Lenovo Legion Pro 7 16IAX10H (type 83F5), Core Ultra 9 275HX, RTX 5080 Laptop, 128 GB, Win11 26220.8544 (Insider Beta)
**Scope:** what another agent changed outside `C:\llm`, and what is still unresolved.

---

## 0. TL;DR for whoever picks this up

- This machine has a **known Windows kernel bug** that bugchecks it. It is *not* your code, *not* bad RAM, *not* the NVIDIA driver.
- A **mitigation stack is currently applied and appears to be working** (~4 days uptime vs. 6 crashes in 24 h before). **Do not undo the power configuration in §2.**
- The GPU is at **175 W**, which required Lenovo Performance mode. That is firmware-locked — see §2.3 before trying to change GPU power from software.
- A dedicated **`llmsvc` account + auto-logon + two scheduled tasks** now exist for running the server headless (§3). **This has never been exercised — no reboot since it was set up.** See §4.
- **Coordination hazard:** re-running `install-autostart.ps1` as `wills` will collide with the new setup. See §5.

---

## 1. The bugcheck bug (essential background)

The machine crashed 14+ times between July and Aug 2026. Root cause is identified with high confidence.

**Signature (from 7 minidumps, analysed in WinDbg):**

```
FAILURE_ID_HASH   : 6f13343d-8edf-14f9-0269-6df067c74f57
FAILURE_BUCKET_ID : AV_nt!ExpPoolTrackerChargeEntry
FAULTING IP       : nt!ExpPoolTrackerChargeEntry+0x40
INSTRUCTION       : lock xadd qword ptr [r14+r8], rbp
BUGCHECKS SEEN    : 0x1E, 0x3B, 0x7E, 0xEF, 0x9F, 0xA — all 0xC0000005
```

It is a **lock-free race on the per-processor `_POOL_TRACKER_TABLE`** in `nt!ExAllocateHeapPool`. A CPU captures its per-processor tracker table pointer, the table is migrated/freed underneath it, and the atomic add lands in freed or wrong memory. The *crashing* component is whatever allocates pool next, so it appears in unrelated subsystems: graphics (`dxgmms2`), registry (`CmpCreateKeyBody`), processor power mgmt (`PpmEventAddAffinityMaskAsSubset`), NTFS (`NtfsReadUsnJournal`).

**This is a documented cross-OEM bug**, not specific to this machine:
- Microsoft Q&A question **5921413**
- Dell community thread for Alienware 16X Aurora AC16251
- Dell KB **000492904** (Dell-model workaround)
- 40+ reports across Alienware, Dell, ASUS, MSI, Acer, Lenovo — common factor is **Core Ultra 9 275HX (Arrow Lake-HX) + Windows 11 25H2**
- Under investigation by Microsoft and Intel; **no fix as of 2026-09-01**

### Things already ruled out — do not re-investigate

| Hypothesis | Why it's dead |
|---|---|
| Bad RAM | Multiple users with the **identical failure hash** ran MemTest86 clean. One dual-booted Win10 on the same hardware for a year with zero crashes. Another ran Linux stress, zero MCEs. **Do not spend days on MemTest.** |
| NVIDIA driver | Crashed on both 32.0.15.9201 (Feb) and 32.0.16.1088 (Jul). Dump confirms the July driver was loaded when it still crashed. |
| Nahimic / VeraCrypt | Both removed/updated; crashed again afterwards with both changes live. |
| Thermals | 171 W / 77 °C under load with `clocks_event_reasons.active = 0x0` — zero throttling. Cooling is fine. |
| WHEA / hardware errors | Log is completely clean. (Note: non-ECC RAM can't report bit flips, so this is weak evidence on its own — but the items above settle it.) |

**Crash correlation:** crashes cluster at **idle / background activity**, not under load — so it is power-state transitions, not allocation throughput. That's what the §2 mitigations target.

---

## 2. Power configuration applied — DO NOT REVERT

### 2.1 Lenovo EC mode

```
SmartFanMode = 3  (Performance)
```

Set via `LENOVO_GAMEZONE_DATA.SetSmartFanMode`. **Required for GPU > 95 W.**

### 2.2 Custom Windows power plan

```
Plan: "Legion PPM-Tamed"  be445bec-2562-41fb-9ffb-25df172f53e4   (ACTIVE)

CPMINCORES     = 100 (AC+DC)   core parking disabled  -> 0 of 24 cores parked
CPMAXCORES     = 100 (AC+DC)
IDLEDISABLE    = 1 on AC, 0 on DC   C-states off on AC, kept on for battery
IDLEPROMOTE    = 100 (AC)      damp idle transition churn
IDLEDEMOTE     = 100 (AC)
PERFBOOSTMODE  = 2 (Aggressive)
```

Rationale: the bug is perturbed by power-state transitions. Core parking changes the active processor set (`PpmEventAddAffinityMaskAsSubset` appears in one dump); C-state entry/exit is most frequent at idle, which is when crashes cluster.

**Intel Dynamic Tuning (DTT) is disabled** — `dptftcs` service Disabled, two DTT PnP devices disabled. Intel IPF (`ipfsvc`) deliberately **left running** (it's the newer Core Ultra framework). Intel APO left enabled.

### 2.3 GPU power is firmware-locked — don't chase it

Verified empirically. `nvidia-smi -pl` is rejected: *"Changing power management limit is not supported in current scope."* Direct EC writes via `LENOVO_OTHER_METHOD.SetFeatureValue` are silently **ignored**:

| Tunable | Quiet | Balance | **Performance** | Custom | Adjustable? |
|---|---|---|---|---|---|
| GPU cTGP | 80 W | 80 W | **150 W** | 80 W | **No** (Step=0, Min=0, Max=0) |
| GPU Power Boost | 0 W | 15 W | **25 W** | 15 W | **No** |
| GPU TPP offset | 10 | 55 | 105 | 55 | Yes (10–135) — *no measurable effect on power cap* |

Balance = 80+15 = **95 W**. Performance = 150+25 = **175 W**. Both confirmed against `nvidia-smi`.

Tested and confirmed useless: Custom mode (255), TPP → 135, cTGP → 150, forced EC re-latch. Enforced limit never moved off 95 W in any non-Performance mode.

**Legion Toolkit cannot work around this and rebuilding it will not help** — LLT's own God Mode preset reads `GPUConfigurableTGP min=0 max=0 step=0`, i.e. the EC declares it read-only. The rejection is below any software layer.

Also note in `nvidia-smi`: **`Max Power Limit: 175 W` is the static VBIOS ceiling and is meaningless here.** The value that tracks the mode is **`Current Power Limit`**.

### 2.4 Result so far

```
Before mitigations (Performance mode): 6 crashes in 24 hours
After  mitigations (Performance mode): 0 crashes, ~4 days uptime (since 2026-08-28 12:25)
```

Encouraging but **not yet proof** — the pre-mitigation record includes an 8.7-day clean streak, so ~1 week of uptime is needed before calling it fixed.

### 2.5 Other machine changes

- Nahimic removed (6 driver packages); Awinic **kept** (it's the speaker amp driver, not bloat)
- VeraCrypt 1.26.20 → 1.26.29
- Lenovo Vantage removed; `LenovoFnAndFunctionKeys` + `LenovoProcessManagement` disabled
- Autostart disabled for Cloudflare WARP, TeamViewer, WhatsApp
- Crash dumps fixed: `CrashDumpEnabled=3`, **`Overwrite=1`** (was 0 — this silently discarded every dump for a year). Minidumps now land in `C:\Windows\Minidump`, backups in the session scratchpad.

---

## 3. LLM autostart setup

### 3.1 Service account

```
LEGION\llmsvc    standard user (NOT admin), SID S-1-5-21-3053374352-2428384167-143889993-1009
Password         C:\llm\llmsvc-credential.txt
                 ACL: inheritance disabled; wills + Administrators only (llmsvc CANNOT read it)
C:\llm           llmsvc granted Modify via (OI)(CI) inheritance
```

### 3.2 Auto-logon

```
AutoAdminLogon    = 1
DefaultUserName   = llmsvc
DefaultDomainName = LEGION
Password storage  = LSA secret (via Sysinternals Autologon64), NOT plaintext registry
```

**How it behaves:** Windows auto-logons **one** account at the console. The machine boots into **llmsvc's desktop**, the server starts there, and the human then switches users (Ctrl+Alt+Del → Switch user) to `wills`. The llmsvc session stays alive in the background, disconnected but running. It is *not* invisible — one extra click per boot. Windows cannot hold two interactive console sessions.

Remove with:
```powershell
& 'C:\Users\wills\AppData\Local\Microsoft\WinGet\Packages\Microsoft.Sysinternals.Autologon_Microsoft.Winget.Source_8wekyb3d8bbwe\Autologon64.exe' /delete
```

### 3.3 Scheduled tasks

| Task | Runs as | Trigger | Settings |
|---|---|---|---|
| `Qwen38Server-Svc` | llmsvc | at llmsvc logon, +30 s | RestartCount **999** @ 1 min, no time limit, battery allowed |
| `Qwen38Watchdog-Svc` | llmsvc | at logon +3 min, then **every 5 min** | 5 min execution limit |
| `Qwen38Server` (original, wills) | wills | at wills logon | **DISABLED** (`Settings.Enabled=False`) to avoid a port-8080 collision |

Both new tasks: `DisallowStartIfOnBatteries=False`, `StopIfGoingOnBatteries=False`, `LogonType=Interactive`, `RunLevel=Limited`.

Action is identical to the original:
```
powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command
  "& 'C:\llm\start-qwen38.ps1' -Mode both -Port 8080 *>> 'C:\llm\logs\server.log'"
```

### 3.4 Watchdog — `C:\llm\watchdog.ps1`

Covers three failure modes:
1. **Process exits / errors** → task restart policy (999× @ 1 min)
2. **Process hangs** (port dead but task still `Running`) → watchdog stops and recycles the task — task-restart alone misses this
3. **Task never fired, or machine bugchecked** → watchdog starts it on the next 5-minute tick

Logs to `C:\llm\logs\watchdog.log`.

---

## 4. OPEN PROBLEMS — things left on the table

### 4.1 ⚠️ The whole autostart setup is UNTESTED

**There has been no reboot since it was configured** (uptime dates to 2026-08-28, setup was 2026-08-29). Evidence:

```
Qwen38Server-Svc      LastRun = 11/30/1999   Result = 0x41303  ("has not yet run")
Qwen38Watchdog-Svc    LastRun = 11/30/1999   Result = 0x41303
Sessions              only  console / wills / Active   (llmsvc has never logged on)
```

Nothing about auto-logon, the llmsvc session, or the watchdog has been exercised even once.

### 4.2 ⚠️ Biggest unknown: does CUDA work in a disconnected session?

When the human switches from llmsvc to wills, llmsvc's session becomes **disconnected**. Session 0 is definitively bad for CUDA under WDDM — that's why the original script's author used an interactive logon trigger. A disconnected Fast-User-Switching session is a *real* interactive session and is usually fine for compute, **but this has not been verified on this machine.**

**If the model fails to load after switching users, this is the cause.** Fallback is running the server in the human's own session and accepting that it stops at logoff.

### 4.3 Core parking could not be disabled globally

`CPMINCORES=100` only takes effect because "Legion PPM-Tamed" is the **active plan with no Windows overlay**. Notes:
- Overlay schemes (e.g. `ded574b5…` Best performance) carry their own `CPMINCORES=50` and **override the plan**
- Those overlay registry keys are ACL-protected — `Set-ItemProperty` fails with *"Requested registry access is not allowed"*, and `powercfg /setacvalueindex <overlay-guid>` silently no-ops
- **Consequence: if anything re-applies a Windows power overlay, core parking silently comes back and the mitigation is weakened.** LLT was switched to `WindowsPowerPlan` mapping mode specifically to stop it setting overlays.

### 4.4 The root bug is unfixed

The mitigations are empirical and target a *hypothesised* mechanism. No published root cause exists. If crashes resume, the trigger is likely in EC firmware below Windows' reach, and 175 W may not be safely attainable on this build. Watch `C:\Windows\Minidump`.

### 4.5 GPU VRAM is shared

The model holds ~14 GB in llmsvc's session; that's unavailable to the human's session.

### 4.6 Battery behaviour is now aggressive

Battery guards were removed as requested. On battery: sleep timeout is **Never**, the server keeps running, and the machine runs to **5 %** then performs a clean shutdown. At LLM load that is not long.

---

## 5. Coordination hazards — read before running anything

1. **Do not re-run `install-autostart.ps1` as `wills`.** `Register-ScheduledTask -Force` will re-enable the wills-owned `Qwen38Server`, and you'll get two servers fighting over port 8080. Either run it as `llmsvc`, or leave `Qwen38Server-Svc` as the live task and edit that.
2. **Always pass `-AllowBattery`** if you do re-register, or the battery guards come back.
3. **Do not switch the Lenovo power mode away from Performance** unless you intend to drop the GPU to 95 W.
4. **Do not re-enable a Windows power overlay** (see §4.3) — it silently re-enables core parking.
5. **LLT automation is currently `IsEnabled: false`, but `automation.json` contains a rule `ACAdapterConnected → Performance`.** Harmless now; if automation is ever enabled, be aware it will force mode changes.
6. `server.log` is written **UTF-16** by PowerShell's `*>>` redirect, so it looks byte-spaced in tools expecting UTF-8. Also, `*>>` only captures the wrapper's streams — `llama-server`'s own stderr was **not** landing in the log during the earlier failure, which made diagnosis harder. Worth fixing.

---

## 6. Quick verification commands

```powershell
# crash check
Get-WinEvent -FilterHashtable @{LogName='System'; Id=41} -MaxEvents 5 | Select TimeCreated
Get-ChildItem C:\Windows\Minidump

# power config intact?
(Invoke-CimMethod -InputObject (Get-CimInstance -Namespace root\wmi -ClassName LENOVO_GAMEZONE_DATA) `
                  -MethodName GetSmartFanMode).Data     # expect 3
powercfg /getactivescheme                                # expect be445bec... Legion PPM-Tamed
(Get-Counter '\Processor Information(*)\Parking Status').CounterSamples |
  Where-Object {$_.InstanceName -notmatch '_total' -and $_.CookedValue -eq 1} | Measure-Object   # expect 0
& 'C:\Windows\System32\nvidia-smi.exe' -q -d POWER | Select-String 'Current Power Limit'         # expect 175 W

# autostart state
Get-ScheduledTask Qwen38Server-Svc,Qwen38Watchdog-Svc,Qwen38Server |
  Select TaskName,State,@{n='Enabled';e={$_.Settings.Enabled}}
query session
Get-Content C:\llm\logs\watchdog.log -Tail 20
```

---

## 7. Appendix — verified 2026-09-01 by a later agent (Claude Code, from WSL)

Appended, not edited: §1–§4 stand and the power work in §2 is confirmed intact
(`enforced.power.limit = 175 W`). Three corrections and one urgent finding.

### 7.1 🔴 URGENT — the llmsvc tasks from §3.3 no longer exist

```
schtasks /query            ->  only \Qwen38Server
C:\Windows\System32\Tasks  ->  no Qwen38Server-Svc, no Qwen38Watchdog-Svc XML
C:\llm\logs\watchdog.log   ->  never created
Qwen38Server (wills)       ->  State=Disabled, Enabled=False
```

The `llmsvc` account and `C:\llm\watchdog.ps1` are still present, but **no task references
them**. Meanwhile `AutoAdminLogon=1 / DefaultUserName=llmsvc` is still set.

**Net effect on next boot: the machine auto-logons to an llmsvc desktop where nothing starts,
and the wills task is disabled, so nothing serves at all.** Worst of both worlds — you get the
extra user-switch click and no server.

Pick one before rebooting:
- **(a) simple** — `Autologon64.exe /delete`, then re-enable `Qwen38Server` for wills.
  Server runs in your session and stops at logoff. Known-good; CUDA definitely works.
- **(b) headless as designed** — recreate both `-Svc` tasks with
  `Register-ScheduledTask -User llmsvc -Password` (credential in `C:\llm\llmsvc-credential.txt`).
  Keeps §4.2's unverified risk: CUDA in a *disconnected* FUS session.

I did not choose for you — (b) is an account-level change to someone else's design.

### 7.2 §5.6 is wrong: `*>>` DOES capture llama-server's stderr

The observation was real but the cause was misattributed. `start-qwen38.ps1` set
`$ErrorActionPreference = 'Stop'`; under redirection PowerShell turns native stderr into error
records, so llama-server's **first log line became a terminating error** and the task exited
`0x1` having written nothing. It was never a redirection-scope problem.

Fixed (`Continue` + `2>&1`) and verified both ways:
- success path: **54 of 84 log lines** are llama-server's own output
- forced failure: `E llama_model_load: error loading model…`, `E … failed to load draft model`,
  `E srv llama_server: exiting due to model loading error` — all captured

### 7.3 Fixed alongside: the wrapper swallowed the exit code

It returned 0 even when llama-server died, so **Task Scheduler saw success and RestartCount
never fired** — a failed start would have left the endpoint silently missing. Now
`exit $LASTEXITCODE` (verified: 1 on failure; the port-busy guard still exits 0 so a duplicate
launch isn't treated as a crash).

### 7.4 `start-qwen38.ps1` gained parameters

`-Bind` (default `127.0.0.1`; use `0.0.0.0` for Docker/LAN clients), `-Alias` (default
`qwen3.8-27b` — without it clients see the full `.gguf` path as the model id), `-ApiKey`.
Named `-Bind` deliberately: `-Host` collides with PowerShell's automatic `$Host`.

Client configuration for Open WebUI / AnythingLLM: `docs/CLIENT_SETUP.md` in the wsl-llm repo.

### 7.5 Also worth knowing

`server.log` being UTF-16 (§5.6) is unfixed. `--parallel 1` is not a tuning choice — raising it
overruns VRAM and the driver silently evicts the model to system RAM (~700× decode collapse), so
Open WebUI's background title/tag requests will queue; disable those in its settings.
