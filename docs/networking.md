# Networking

> Last updated: March 2026

## Network Architecture

```
                         Internet
                            |
                    Cloudflare Edge
                            |
                    llm.pet (HTTPS)
                            |
                    cloudflared tunnel
                            |
              +-------------+-------------+
              |             |             |
         :3000          :4000         :8080
       Open WebUI    LiteLLM       llama-server
       (optional)    (proxy)       (ik_llama.cpp)
              |             |
              |      PostgreSQL :5434
              |      (Docker)
              |             |
              +------+------+
                     |
              WSL2 VM (Ubuntu 22.04)
                     |
               NAT / mirrored
                     |
              Windows 11 Host
                     |
              LAN (192.168.x.x)
                     |
              Other devices
```

## WSL2 Networking Modes

### NAT Mode (default)

WSL2 gets its own virtual network adapter with a private IP (typically 172.x.x.x). Windows host NATs traffic between the LAN and WSL.

- `localhost` forwarding works from Windows to WSL (built-in since Windows 11)
- LAN devices cannot reach WSL services directly without port forwarding
- WSL IP changes on every restart

### Mirrored Mode

Add to `C:\Users\Will\.wslconfig`:
```ini
[wsl2]
networkingMode=mirrored
```

- WSL shares the Windows host IP
- LAN devices can reach WSL services directly (no port forwarding needed)
- Simpler, but can conflict with Docker networking and VPNs
- Requires Windows 11 22H2+

**This project uses NAT mode** with explicit port forwarding rules.

## Port Forwarding (Windows to WSL)

Since NAT mode hides WSL behind a private IP, forward ports from the Windows host to WSL:

```powershell
# Run as Administrator
# Get current WSL IP
$wslIp = (wsl hostname -I).Trim().Split(" ")[0]

# Forward ports
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=4000 listenaddress=0.0.0.0 connectport=4000 connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=3000 listenaddress=0.0.0.0 connectport=3000 connectaddress=$wslIp

# Verify
netsh interface portproxy show v4tov4
```

The WSL IP changes on restart. If using NAT mode, re-run the portproxy commands after each WSL restart, or add them to a startup script.

**Note:** `localhost` access from Windows works without portproxy (Windows 11 auto-forwards). Portproxy is only needed for LAN device access.

## Windows Firewall Rules

Open the required ports for LAN access:

```powershell
# Run as Administrator
New-NetFirewallRule -DisplayName "LLM Server (8080)" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "LiteLLM Proxy (4000)" -Direction Inbound -LocalPort 4000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "Open WebUI (3000)" -Direction Inbound -LocalPort 3000 -Protocol TCP -Action Allow
```

To verify:
```powershell
Get-NetFirewallRule -DisplayName "LLM*","LiteLLM*","Open WebUI*" | Format-Table DisplayName,Enabled,Direction
```

## Cloudflare Tunnel

The Cloudflare tunnel exposes the LiteLLM proxy to the internet as `llm.pet` without opening any inbound ports on the router.

### How It Works

1. `cloudflared` runs inside WSL as a systemd service
2. It establishes an outbound connection to Cloudflare's edge
3. Cloudflare routes `llm.pet` traffic through the tunnel to `localhost:4000`
4. No port forwarding or static IP needed on the home network

### Setup

```bash
# Install cloudflared
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared jammy main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install cloudflared

# Authenticate (one-time)
cloudflared tunnel login

# Create tunnel (one-time)
cloudflared tunnel create wsl-llm

# Tunnel config lives at ~/.cloudflared/config.yml
```

### Config File

`~/.cloudflared/config.yml`:
```yaml
tunnel: <tunnel-id>
credentials-file: /home/matilda/.cloudflared/<tunnel-id>.json

ingress:
  - hostname: llm.pet
    service: http://localhost:4000
  - service: http_status:404
```

### DNS

In the Cloudflare dashboard for the `llm.pet` domain, add a CNAME record:
- Name: `@` (or subdomain)
- Target: `<tunnel-id>.cfargotunnel.com`
- Proxy: ON (orange cloud)

### Systemd Service

The tunnel runs as a systemd unit (`cloudflared.service`) that starts after `llama-server.service`. See `services/cloudflared.service.template`.

```bash
sudo systemctl status cloudflared
journalctl -u cloudflared -f
```

## LAN Access from Other Devices

### From Windows Host

Services are accessible at `localhost`:
- llama-server: `http://localhost:8080`
- LiteLLM: `http://localhost:4000`
- LiteLLM Admin UI: `http://localhost:4000/ui`

### From LAN Devices (phones, other PCs)

Requires port forwarding + firewall rules (see above).

Access via the Windows host's LAN IP:
```
http://192.168.x.x:4000    # LiteLLM (recommended entry point)
http://192.168.x.x:8080    # Direct llama-server access
http://192.168.x.x:3000    # Open WebUI (if running)
```

Find the Windows host IP:
```powershell
ipconfig | findstr "IPv4"
```

### From the Internet

Use the Cloudflare tunnel domain: `https://llm.pet`

All traffic is authenticated through LiteLLM API keys. No direct internet exposure of the host.

## Security Notes

### Authentication Layers

1. **Cloudflare tunnel**: Only `llm.pet` traffic reaches WSL. No inbound ports open on the router.
2. **LiteLLM proxy**: Requires API key for all requests. Keys are per-user with model access controls.
3. **llama-server**: Has its own `--api-key` flag. Only LiteLLM should talk to it directly.

### What Is NOT Exposed

- PostgreSQL (port 5434): Bound to localhost only
- llama-server (port 8080): Reachable from LAN if firewall rule exists, but requires API key
- WSL SSH: Not running by default

### Recommendations

- Do not expose port 8080 to the internet -- always go through LiteLLM for key management and rate limiting
- Rotate LiteLLM API keys if compromised: `curl -X POST http://localhost:4000/key/delete -H "Authorization: Bearer <master-key>" -d '{"keys":["sk-compromised-key"]}'`
- Cloudflare tunnel credentials (`~/.cloudflared/*.json`) are secrets -- do not commit to git
- LiteLLM master key and database password are in the systemd service environment -- restrict file permissions on the service unit

### Port Summary

| Port | Service | Accessible From | Auth |
|------|---------|----------------|------|
| 8080 | llama-server | WSL, Windows localhost, LAN (if forwarded) | API key |
| 4000 | LiteLLM proxy | WSL, Windows localhost, LAN (if forwarded), Internet (via tunnel) | API key |
| 3000 | Open WebUI | WSL, Windows localhost, LAN (if forwarded) | Login |
| 5434 | PostgreSQL | WSL localhost only | DB password |
