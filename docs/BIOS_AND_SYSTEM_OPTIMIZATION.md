# BIOS & System Optimization Guide
## ASUS ROG STRIX X570-E GAMING WIFI II + 2x RTX 3090 + Ryzen 9 5900X

**Hardware:**
- CPU: AMD Ryzen 9 5900X (12c/24t)
- RAM: 4x Crucial Ballistix BL16G36C16U4B/W 16GB DDR4-3600 CL16 (Micron Rev.B, single-rank)
- GPUs: ASUS RTX 3090 + NVIDIA RTX 3090 FE (24GB each)
- Storage: Samsung 990 PRO 4TB
- Use case: 24/7 llama.cpp inference server via WSL2

**Date:** March 2026
**Previous BIOS:** 0309 (August 2021, initial release — critically outdated)

---

## 1. BIOS Update

Flash to **5041** (stable, August 2025) or **5044** (Jan 2026).
- Download from ASUS support page
- Rename file to `R570EGW2.CAP`, put on FAT32 USB
- Flash via EZ Flash 3 (Tool menu in BIOS)
- Can go directly from 0309 → latest, no incremental updates needed
- **If using BitLocker with fTPM: back up recovery key first** — fTPM firmware update clears keys

### What you were missing on 0309:
- fTPM stuttering fix (random 1-2s freezes — AGESA 1.2.0.7)
- Multiple AMD security patches (Zenbleed, Inception, CVE-2024-36347)
- Sleep/hibernate crash fix
- Improved memory training for 4-DIMM stability
- ReBAR support

---

## 2. BIOS Settings Checklist

### PCIe / GPU Stability

| Setting | Path | Value | Why |
|---------|------|-------|-----|
| Above 4G Decoding | Advanced → PCI Subsystem | **Enabled** | Required for 2x 24GB GPUs |
| Re-Size BAR | Advanced → PCI Subsystem | **Auto** | Free perf (vBIOS already supports it) |
| CSM | Boot → CSM | **Disabled** | Required for ReBAR; UEFI boot only |
| ASPM | AMD CBS → NBIO → SMU Common | **Disabled** | Prevents GPU "fallen off bus" crashes |
| ASPM L0s/L1 | Same area | **Disabled** | L0s specifically crashes 3090s on idle→active |
| PCIe Speed | Advanced → PCIe Config | **Gen 3** | Eliminates X570 WHEA correctable errors |

### CPU / Power (24/7 stability)

| Setting | Path | Value | Why |
|---------|------|-------|-----|
| SVM Mode | Advanced → CPU Config | **Enabled** | Required for WSL2/Hyper-V |
| IOMMU | AMD CBS → NBIO | **Enabled** | Security + Hyper-V |
| Global C-States | AMD CBS | **Enabled** | Saves ~40W idle |
| DF C-States | AMD CBS | **Disabled** | Prevents deep-idle reboots |
| Power Supply Idle Control | AMD CBS → NBIO | **Typical Current Idle** | Fixes "Ryzen random restart at idle" |
| CPPC / Preferred Cores | AMD CBS | **Enabled** | Efficient frequency scaling |

### Memory (4x Crucial Ballistix DDR4-3600 CL16)

| Setting | Path | Value | Why |
|---------|------|-------|-----|
| DOCP | AI Tweaker | **Enabled (Profile 1)** | Activates 3600 CL16 @ 1.35V |
| FCLK | AI Tweaker | **1800 MHz** | 1:1 ratio — Zen 3 sweet spot |
| DRAM Voltage | AI Tweaker | **1.35V** (1.37V if unstable) | XMP spec |
| SoC Voltage | AI Tweaker | **1.10V** (manual) | Stabilizes 4-DIMM config |
| CLDO VDDP | AMD CBS | **0.900V** | Memory controller stability |
| VDDG CCD | AMD CBS | **0.950V** | Must be < SoC voltage |
| VDDG IOD | AMD CBS | **1.050V** | Must be < SoC, > VDDG CCD |
| Gear Down Mode | AI Tweaker → DRAM Timing | **Enabled** | Crucial for 4-DIMM stability |

**Voltage hierarchy:** VSOC (1.10V) > VDDG IOD (1.05V) > VDDG CCD (0.95V) > VDDP (0.90V)

**RAM note:** Black (U4B) and White (U4W) are electrically identical. Install each kit pair in the same channel.

### Misc

| Setting | Path | Value | Why |
|---------|------|-------|-----|
| Chipset SATA Hot Plug | Advanced → SATA Config | **Disabled** | Reduces WHEA errors |
| ErP Ready | Advanced → APM Config | **Disabled** | Allows wake-on-LAN |
| Q-Fan (chassis) | Monitor → Q-Fan | **Turbo** | Dual 3090s dump heat |

---

## 3. Windows Registry Changes

Run in admin PowerShell:
```powershell
# TDR — 60 second timeout (default is 2s, way too short for CUDA)
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -Name "TdrDelay" -PropertyType DWord -Value 60 -Force
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\GraphicsDrivers" -Name "TdrDdiDelay" -PropertyType DWord -Value 60 -Force
```

## 4. Windows Power Settings

- Power Options → PCI Express → Link State Power Management → **Off**
- NVIDIA Control Panel → Manage 3D Settings → Power Management Mode → **Prefer Maximum Performance**

## 5. GPU Startup Script

`C:\Users\Will\gpu-init.bat` runs at login via VBS launcher in Startup folder.
Sets both GPUs to 300W power limit. Shows nvidia-smi output for 30 seconds.

## 6. Auto-Login

```powershell
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" -Name "AutoAdminLogon" -Value "1"
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" -Name "DefaultUserName" -Value "Will"
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" -Name "DefaultDomainName" -Value "NZXT"
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon" -Name "DefaultPassword" -Value "YOUR_PASSWORD"
```

## 7. Verification After Reboot

```powershell
# Check ReBAR is active (should show large value, not 256 MiB)
nvidia-smi -q | findstr BAR1

# Check PCIe link (under load should be Gen 3+ and wider than idle)
nvidia-smi -q | findstr -i "Link Width" 

# Check WHEA errors (should be empty/clean)
Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} -MaxEvents 10
```

## 8. Software to Remove

- ASUS Armoury Crate (use official uninstall tool from ASUS)
- Glary Utilities
- TrueCrypt (if not actively used — VeraCrypt is the successor)

---

## Root Cause Analysis

### CUDA crashes (every ~7 hours)
**Cause:** ASPM L0s causing GPU PCIe link wake failure during idle→active transitions.
Combined with default 2-second TDR timeout. Crashes manifested as CUDA error 999
(cudaErrorUnknown) at cudaMemcpyAsync on GPU 0.

**Fix:** Disable ASPM in BIOS + increase TDR timeout.

### BSOD on March 9 (0x154 UNEXPECTED_STORE_EXCEPTION)
**Cause:** Likely related to the same GPU instability or a third-party kernel driver.
Suspicious drivers: MSIO, IOMap (ASUS), GUBootStartup (Glary), TrueCrypt 7.1a.
No crash dump was saved (volmgr failed). No WHEA hardware errors.

### RustDesk / DWM weirdness
**Cause:** After the BSOD, RustDesk lost NVENC hardware encoding capability on one
GPU adapter (3 LUIDs → 2, ram_encode went empty). GPU context was corrupted.
**Fix:** Restart RustDesk after any crash/reboot.

### PCIe link width (x4/x8 at idle)
**Not a problem.** Normal PCIe power management. Scales back to full width under load.
GPU 1 at x8 max is by design (X570 chipset slot). 100+ t/s inference confirms no
bandwidth bottleneck.
