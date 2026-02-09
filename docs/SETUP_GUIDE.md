# Qwen3-Coder-Next Server Management Guide

## Quick Start Commands

### Windows (Double-click .bat files)
- **Start**: `qwen-start.bat`
- **Stop**: `qwen-stop.bat`
- **Restart**: `qwen-restart.bat`
- **Status**: `qwen-status.bat`

### WSL/Linux (Command line)
```bash
# Start server
~/start-qwen.sh

# Stop server
~/stop-qwen.sh

# Check status
~/status-qwen.sh

# View live logs
tail -f ~/qwen-server.log
```

### Systemd Service (Auto-start on WSL boot)
```bash
# Enable auto-start
sudo systemctl enable qwen-server

# Start service
sudo systemctl start qwen-server

# Stop service
sudo systemctl stop qwen-server

# Check status
sudo systemctl status qwen-server

# View logs
sudo journalctl -u qwen-server -f
```

## Access URLs

- **Localhost**: http://localhost:8080
- **Local Network**: http://[WSL_IP]:8080 (from other devices - use `hostname -I` to find WSL IP)
- **Health Check**: http://localhost:8080/health
- **API Docs**: http://localhost:8080/docs

## API Examples

### Completions API (OpenAI-compatible)
```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def fibonacci(n):\n",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Chat API (OpenAI-compatible)
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to check if a number is prime"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

## Server Configuration

**Current Setup:**
- Model: Qwen3-Coder-Next-UD-Q3_K_XL.gguf (35.78 GB)
- Context Window: 128k tokens
- VRAM Usage: ~44.6 GB (2x RTX 3090)
- Generation Speed: ~34.6 tokens/second
- Flash Attention: Enabled

**To increase context to 256k** (requires more VRAM optimization):
- Edit scripts and change `--ctx-size 131072` to `--ctx-size 262144`
- May require KV cache quantization or smaller model

## Monitoring & Management

### Recommended Web UIs

1. **Open WebUI** (Best all-in-one)
   ```bash
   # Get WSL IP first
   WSL_IP=$(wsl hostname -I | awk '{print $1}')

   docker run -d -p 3000:8080 \
     -e OPENAI_API_BASE_URL=http://$WSL_IP:8080/v1 \
     -e OPENAI_API_KEY=dummy \
     --name open-webui \
     ghcr.io/open-webui/open-webui:main
   ```
   Access at: http://localhost:3000

2. **text-generation-webui** (Feature-rich)
   ```bash
   git clone https://github.com/oobabooga/text-generation-webui
   cd text-generation-webui
   # Add as API endpoint in settings
   ```

3. **LibreChat** (ChatGPT-like interface)
   ```bash
   WSL_IP=$(wsl hostname -I | awk '{print $1}')

   docker run -d -p 3080:3080 \
     -e OPENAI_API_KEY=dummy \
     -e OPENAI_REVERSE_PROXY=http://$WSL_IP:8080/v1 \
     ghcr.io/danny-avila/librechat
   ```

### System Monitoring

**GPU Monitoring:**
```bash
# Real-time GPU stats
wsl watch -n 1 nvidia-smi

# Detailed GPU info
wsl nvidia-smi dmon -s pucvmet
```

**Server Metrics:**
```bash
# Check active requests
curl http://localhost:8080/metrics

# Check slots status
curl http://localhost:8080/slots
```

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
wsl netstat -tulpn | grep 8080

# Check CUDA is working
wsl nvidia-smi

# Check logs
wsl tail -f ~/qwen-server.log
```

### Out of memory errors
- Reduce context size: `--ctx-size 65536` (64k)
- Switch to Q4 model if you have less context needs
- Close other GPU applications

### Slow inference
- Check GPU utilization: `wsl nvidia-smi`
- Ensure Flash Attention is enabled (check logs)
- Reduce `--n-parallel` if memory constrained

## Performance Tuning

### Increase throughput for multiple users:
```bash
--n-parallel 8        # Allow 8 concurrent requests
--n-batch 2048        # Larger batch for better GPU usage
--cache-prompt        # Enable prompt caching
```

### Lower latency for single user:
```bash
--n-parallel 1        # Single request at a time
--n-ubatch 512        # Smaller microbatch
```

### Memory optimization:
```bash
--cache-type-k q4_1   # Quantize KV cache (if Flash Attention supports)
--cache-type-v q4_1   # Can save ~50% KV cache memory
```

## Integration Examples

### Python
```python
import openai

openai.api_base = "http://localhost:8080/v1"
openai.api_key = "dummy"

response = openai.ChatCompletion.create(
    model="Qwen3-Coder-Next",
    messages=[{"role": "user", "content": "Write a hello world in Python"}]
)
print(response.choices[0].message.content)
```

### VSCode Extension
Install "Continue" or "Codeium" extension and configure:
```json
{
  "apiBase": "http://localhost:8080/v1",
  "apiKey": "dummy",
  "model": "Qwen3-Coder-Next"
}
```

### Cursor / Windsurf
Add custom model endpoint in settings:
- API Base: http://localhost:8080/v1
- Model: Qwen3-Coder-Next

## Backup & Updates

### Update llama.cpp:
```bash
wsl bash -l -c "cd ~/llama.cpp && git pull && cmake --build build --config Release -j --target llama-server"
```

### Download different model quantizations:
```bash
# Q4 for better quality (46GB)
wsl bash -l -c "huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF Qwen3-Coder-Next-UD-Q4_K_XL.gguf --local-dir ~/unsloth/Qwen3-Coder-Next-GGUF"

# Q2 for lower memory (smaller, lower quality)
wsl bash -l -c "huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF Qwen3-Coder-Next-UD-Q2_K.gguf --local-dir ~/unsloth/Qwen3-Coder-Next-GGUF"
```

## Security Notes

- Server is exposed on 0.0.0.0 (all interfaces) - accessible on LAN
- No authentication by default - add reverse proxy with auth if exposing to internet
- Consider using nginx or Caddy for HTTPS and authentication
- Never expose directly to internet without proper security

## Support

- llama.cpp issues: https://github.com/ggml-org/llama.cpp/issues
- Model info: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF
- Unsloth docs: https://unsloth.ai/docs
