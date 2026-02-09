# Windows Firewall Setup

## Quick Setup (Recommended)

1. **Right-click** `windows-scripts\setup-firewall.bat`
2. Select **"Run as administrator"**
3. Confirm the UAC prompt
4. Firewall rule will be created automatically

## Manual Setup (PowerShell)

If you prefer PowerShell, run as Administrator:

```powershell
New-NetFirewallRule -DisplayName "Qwen LLM Server" `
  -Direction Inbound `
  -LocalPort 8080 `
  -Protocol TCP `
  -Action Allow `
  -Profile Private,Domain `
  -Description "Allow local network access to Qwen3-Coder-Next LLM server running in WSL"
```

## Verify Firewall Rule

```cmd
netsh advfirewall firewall show rule name="Qwen LLM Server"
```

## Test Access

### From Windows (localhost)
```bash
curl http://localhost:8080/health
```

### From Other Devices on LAN
1. Get your WSL IP: `wsl hostname -I`
2. From another device: `curl http://[WSL_IP]:8080/health`
3. Or open in browser: `http://[WSL_IP]:8080`

## Remove Firewall Rule (if needed)

```cmd
netsh advfirewall firewall delete rule name="Qwen LLM Server"
```

## Troubleshooting

### Port already in use
```cmd
netstat -ano | findstr :8080
```

### Check Windows Firewall status
```cmd
netsh advfirewall show allprofiles
```

### Temporarily disable firewall (testing only)
```powershell
# DO NOT do this on production systems
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False

# Re-enable after testing
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True
```

## Security Notes

- Firewall rule only allows connections on **Private and Domain** networks
- **Public networks are blocked** by default for security
- Server has no authentication - consider adding reverse proxy with auth
- Never expose directly to internet without proper security

## WSL Network Architecture

```
┌─────────────────────────────────────┐
│ Windows Host (Your PC)              │
│  ├─ IP: 192.168.x.x (LAN)          │
│  ├─ Firewall Rule: Port 8080       │
│  └─ WSL2 VM                         │
│      ├─ IP: 172.x.x.x (internal)   │
│      └─ llama-server: 0.0.0.0:8080 │
└─────────────────────────────────────┘
         │
         ├──> localhost:8080 (Windows)
         ├──> 172.x.x.x:8080 (WSL IP)
         └──> 192.168.x.x:8080 (LAN access)
```

## Related Files

- `windows-scripts/setup-firewall.bat` - Automated setup script
- `windows-scripts/qwen-start.bat` - Start server
- `windows-scripts/qwen-status.bat` - Check server status
- `docs/SETUP_GUIDE.md` - Complete setup documentation
