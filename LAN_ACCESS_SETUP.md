# LAN Access Setup for WSL Server

## The Problem

By default, WSL2 uses a virtualized network (NAT), which means:
- ✅ Windows can access WSL via `localhost:8080`
- ❌ Other devices on your LAN cannot directly access WSL services

## Solution: Mirrored Networking Mode

I've configured `.wslconfig` to use **mirrored networking**, which makes WSL share Windows' network adapter.

## Setup Steps

### 1. Restart WSL (Required)

The `.wslconfig` file has been created, but you need to restart WSL for it to take effect:

```cmd
wsl --shutdown
```

Wait 8 seconds, then start WSL again:
```cmd
wsl
```

### 2. Verify Mirrored Mode is Active

After restarting WSL, check your IP:

```bash
# In WSL
ip addr show eth0
```

In mirrored mode, WSL will have an IP on your LAN subnet (192.168.x.x), not an internal one (172.x.x.x).

### 3. Configure Windows Firewall (If Not Done)

Run as Administrator:
```cmd
windows-scripts\setup-firewall.bat
```

### 4. Access from LAN

From device at 192.168.1.32:

```bash
# Get your Windows/WSL IP first (should be same in mirrored mode)
# Example: 192.168.1.100

# Test health
curl http://192.168.1.100:8080/health

# Or open in browser
http://192.168.1.100:8080
```

## Troubleshooting

### Check if mirrored mode is working

```bash
# In WSL
wsl hostname -I
```

You should see your LAN IP (192.168.x.x), not 172.x.x.x

### If mirrored mode isn't working

**Fallback Option: Port Forwarding**

Run as Administrator in PowerShell:

```powershell
# Get your Windows IP
$windowsIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Ethernet*" | Where-Object {$_.IPAddress -like "192.168.*"}).IPAddress

# Get WSL IP
wsl hostname -I

# Forward port (replace WSL_IP with actual IP)
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=<WSL_IP>

# Verify
netsh interface portproxy show all
```

### Check port forwarding

```cmd
netsh interface portproxy show all
```

### Remove port forwarding (if needed)

```cmd
netsh interface portproxy delete v4tov4 listenport=8080 listenaddress=0.0.0.0
```

## Network Architecture

### Before (Default NAT Mode)
```
LAN Device (192.168.1.32)
    ❌ Cannot reach WSL

Windows Host (192.168.1.100)
    ├─ Firewall: blocks external access
    └─ WSL2 VM (172.19.38.33) ← isolated network
        └─ llama-server:8080
```

### After (Mirrored Mode)
```
LAN Device (192.168.1.32)
    └─> 192.168.1.100:8080
        ↓
Windows Host (192.168.1.100)
    ├─ Firewall: allows port 8080 ✅
    └─ WSL2 (same IP: 192.168.1.100) ← shared network
        └─ llama-server:8080
```

## Configuration Files

- `C:\Users\Will\.wslconfig` - WSL global configuration (created)
- `/etc/wsl.conf` - WSL distribution configuration (already configured)
- Windows Firewall Rule: Port 8080 (run setup-firewall.bat)

## Benefits of Mirrored Mode

1. **LAN Access**: Other devices can connect directly
2. **Better Performance**: Less network overhead
3. **Simpler**: No port forwarding needed
4. **Consistent IPs**: WSL and Windows share the same IP

## Security Notes

- Server accessible on LAN without authentication
- Consider adding reverse proxy with auth for production use
- Firewall rule only allows Private/Domain networks (not Public WiFi)

## Testing

### From Windows
```bash
curl http://localhost:8080/health
```

### From LAN Device (192.168.1.32)
```bash
# Replace with your actual Windows IP
curl http://192.168.1.100:8080/health
```

### From Browser
```
http://192.168.1.100:8080
```

## Next Steps

1. **Restart WSL**: `wsl --shutdown` (wait 8s, then restart)
2. **Verify IP**: `wsl hostname -I` (should show 192.168.x.x)
3. **Test LAN access**: From 192.168.1.32, curl the endpoint
4. **Install Web UI**: See `docs/IMPROVEMENTS.md` for Open WebUI setup

---

**Created:** 2026-02-09
**WSL Version:** 2.2.4.0 (supports mirrored networking)
