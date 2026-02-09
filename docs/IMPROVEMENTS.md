# Qwen Server - Recommended Improvements

## 1. Web UI for Easy Access (Highly Recommended)

### Option A: Open WebUI (Best Choice)
**Best all-in-one solution with chat interface, model management, and user auth.**

```bash
# Install with Docker
docker run -d -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://172.19.38.33:8080/v1 \
  -e OPENAI_API_KEY=dummy \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```
Access: http://localhost:3000

**Features:**
- ChatGPT-like interface
- Multi-user support with auth
- Conversation history
- Model switching
- RAG (document upload)
- Image generation support
- Mobile-friendly

### Option B: SillyTavern (For Character/Creative Writing)
```bash
git clone https://github.com/SillyTavern/SillyTavern
cd SillyTavern
npm install
node server.js
```
- Great for roleplay, storytelling, character chats
- Extensive prompt templates
- Character cards support

### Option C: Text Generation WebUI
```bash
# One-liner install (Windows)
curl -LO https://github.com/oobabooga/text-generation-webui/releases/download/installers/oobabooga_windows.zip
# Extract and run start_windows.bat
# In UI: Add OpenAI API extension, set endpoint to http://localhost:8080/v1
```

## 2. VSCode Integration

### Continue Extension (Free)
1. Install "Continue" extension in VSCode
2. Configure `.continue/config.json`:
```json
{
  "models": [{
    "title": "Qwen3-Coder",
    "provider": "openai",
    "model": "Qwen3-Coder-Next",
    "apiBase": "http://localhost:8080/v1",
    "apiKey": "dummy"
  }]
}
```

### Codeium Extension
- Add custom LLM endpoint in settings
- Point to http://localhost:8080

## 3. Monitoring Dashboard

### Grafana + Prometheus
**Full monitoring stack for production use:**

```bash
# Install Prometheus
wsl bash -l -c "wget https://github.com/prometheus/prometheus/releases/download/v2.48.0/prometheus-2.48.0.linux-amd64.tar.gz"

# Configure to scrape llama-server metrics at http://localhost:8080/metrics
```

### Simple GPU Monitoring (nvitop)
```bash
wsl bash -l -c "pip3 install nvitop"
wsl nvitop
```
Better than nvidia-smi with colorful real-time graphs.

## 4. Reverse Proxy with Auth (Security)

### Nginx with Basic Auth
```bash
wsl sudo apt install nginx apache2-utils

# Create password
wsl sudo htpasswd -c /etc/nginx/.htpasswd yourusername

# Configure nginx
wsl sudo tee /etc/nginx/sites-available/qwen << 'EOF'
server {
    listen 8081;
    location / {
        auth_basic "Qwen Server";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
    }
}
EOF

wsl sudo ln -s /etc/nginx/sites-available/qwen /etc/nginx/sites-enabled/
wsl sudo systemctl restart nginx
```
Now access with auth at: http://localhost:8081

### Caddy (Simpler, Auto HTTPS)
```bash
wsl bash -l -c "curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg"
wsl bash -l -c "sudo apt install caddy"

# Caddyfile
wsl sudo tee /etc/caddy/Caddyfile << 'EOF'
:8081 {
    basicauth {
        yourusername $2a$14$... # generate with: caddy hash-password
    }
    reverse_proxy localhost:8080
}
EOF

wsl sudo systemctl restart caddy
```

## 5. Performance Optimizations

### A. Increase Context to 256k (if needed)

**Option 1: Enable KV cache quantization** (saves ~50% memory)
Check if your llama.cpp build supports it:
```bash
wsl ~/llama.cpp/build/bin/llama-server --help | grep cache-type
```
If supported, modify start scripts:
```bash
--ctx-size 262144 \
--cache-type-k q4_1 \
--cache-type-v q4_1
```

**Option 2: Use Q2_K model** (smaller, lower quality)
Download and switch model in scripts.

### B. Multi-GPU Load Balancing
Currently both GPUs are used, but you can split explicitly:
```bash
--tensor-split 24,24  # Split evenly across GPUs
```

### C. Prompt Caching for Repeated Requests
Already enabled by default, but verify:
```bash
--cache-prompt  # Cache common prompts
```

## 6. API Rate Limiting (Multi-user)

### Using nginx limit_req
```nginx
http {
    limit_req_zone $binary_remote_addr zone=qwen:10m rate=10r/s;

    server {
        location / {
            limit_req zone=qwen burst=20;
            proxy_pass http://localhost:8080;
        }
    }
}
```

## 7. Logging & Analytics

### Structured Logging
Redirect logs to JSON for easier parsing:
```bash
# In systemd service or start script, pipe through jq
--log-format json | tee -a qwen-server.jsonl
```

### Request Analytics
Use tools like GoAccess to analyze access patterns:
```bash
wsl sudo apt install goaccess
wsl goaccess ~/qwen-server.log -o ~/report.html --log-format=COMBINED
```

## 8. Backup & Disaster Recovery

### Automated Backups
```bash
# Backup script
wsl bash -l -c "cat > ~/backup-qwen.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=/mnt/c/Users/Will/qwen-backups
mkdir -p \$BACKUP_DIR
tar -czf \$BACKUP_DIR/qwen-$(date +%Y%m%d).tar.gz \
  ~/llama.cpp \
  ~/unsloth \
  ~/.bash_profile \
  /etc/wsl.conf \
  /etc/systemd/system/qwen-server.service
EOF
chmod +x ~/backup-qwen.sh"

# Run weekly with cron
wsl bash -l -c "(crontab -l 2>/dev/null; echo '0 0 * * 0 ~/backup-qwen.sh') | crontab -"
```

## 9. Model Hot-Swapping

### Multi-Model Support
Run multiple models on different ports:
```bash
# Terminal 1: Qwen3 Q3 on 8080 (128k context)
~/start-qwen.sh

# Terminal 2: Qwen3 Q4 on 8081 (64k context, higher quality)
~/llama.cpp/build/bin/llama-server \
  --model ~/unsloth/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --port 8081 \
  --ctx-size 65536
```

### Model Router (Smart Load Balancing)
Use LiteLLM to route between models:
```bash
pip install litellm
litellm --config litellm_config.yaml
```

## 10. Windows Firewall Configuration

### Allow LAN Access
```powershell
# Run in PowerShell as Administrator
New-NetFirewallRule -DisplayName "Qwen LLM Server" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
```

### WSL Port Forwarding (If needed)
```powershell
# Forward Windows port 8080 to WSL
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=172.19.38.33
```

## 11. Advanced: Fine-tuning

### Use Unsloth to fine-tune on your own code
```bash
# Install Unsloth
wsl pip3 install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Fine-tune script (requires significant VRAM, may need offloading)
# See: https://unsloth.ai/docs/getting-started/training
```

## 12. Cost Savings vs Cloud

**Your Setup (2x 3090):**
- One-time: Already owned
- Power: ~700W under load = ~$0.10-0.20/hour (depending on electricity rates)
- Unlimited requests

**Equivalent Cloud (Claude Opus / GPT-4):**
- API: ~$15-60 per million tokens
- Your 128k context = $2-8 per request
- 1000 requests/day = $60,000-240,000/month

**Break-even:** Your setup pays for itself in days with heavy usage!

## Summary of Recommended Stack

**Minimal Setup (Current):**
- llama.cpp server
- Start/stop scripts ✅

**Recommended Production Setup:**
1. **Web UI**: Open WebUI (easiest) or Text-gen-webui
2. **Monitoring**: nvitop or Grafana + Prometheus
3. **Security**: Caddy reverse proxy with auth
4. **Integration**: VSCode Continue extension
5. **Backup**: Automated weekly backups
6. **Service**: systemd auto-start ✅

**Enterprise Setup:**
1. All of the above +
2. Rate limiting with nginx
3. Multiple models on different ports
4. LiteLLM router for load balancing
5. Structured JSON logging
6. Analytics dashboard

## Next Steps Priority

1. **Install Open WebUI** - Makes everything accessible via browser
2. **Install nvitop** - Better GPU monitoring
3. **Enable systemd** - Auto-start on boot
4. **Setup VSCode Continue** - Code completion in editor
5. **Configure backups** - Protect your setup

Would you like me to set up any of these for you?
