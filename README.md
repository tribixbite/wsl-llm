# WSL LLM Setup - Qwen3-Coder-Next

Complete setup for running Qwen3-Coder-Next (80B MoE) on WSL2 with NVIDIA GPUs using llama.cpp.

## Hardware Requirements

- **GPU**: 2x NVIDIA RTX 3090 (48GB VRAM total) or similar
- **OS**: Windows 11 with WSL2
- **Driver**: NVIDIA 591.74+ (supports CUDA 13.1)
- **Storage**: ~50GB for models + llama.cpp

## Quick Start

### Windows (Easiest)
```cmd
windows-scripts\qwen-start.bat      # Start server
windows-scripts\qwen-status.bat     # Check status
windows-scripts\qwen-stop.bat       # Stop server
```

### WSL/Linux
```bash
./scripts/start-qwen.sh    # Start
./scripts/status-qwen.sh   # Status
./scripts/stop-qwen.sh     # Stop
```

### Access
- **Local**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **Health**: http://localhost:8080/health

## Installation

### 1. Fix WSL PATH (Important!)
```bash
# Copy config
sudo cp config/wsl.conf /etc/wsl.conf
cp config/bash_profile ~/.bash_profile

# Restart WSL
wsl --shutdown
wsl
```

### 2. Install Dependencies
```bash
# Build tools
sudo apt update
sudo apt install -y build-essential cmake git curl wget

# CUDA toolkit (development only, NO driver)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6

# Hugging Face CLI
pip3 install --upgrade huggingface_hub[cli,hf_transfer]
```

### 3. Build llama.cpp
```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build --config Release -j --target llama-server
```

### 4. Download Model
```bash
# Q3_K_XL (36GB) - Recommended for 128k context
huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
  Qwen3-Coder-Next-UD-Q3_K_XL.gguf \
  --local-dir ~/unsloth/Qwen3-Coder-Next-GGUF

# Q4_K_XL (46GB) - Better quality, less context
huggingface-cli download unsloth/Qwen3-Coder-Next-GGUF \
  Qwen3-Coder-Next-UD-Q4_K_XL.gguf \
  --local-dir ~/unsloth/Qwen3-Coder-Next-GGUF
```

### 5. Install Scripts
```bash
# From this repo root
cp scripts/*.sh ~/
chmod +x ~/*.sh

# Optional: systemd service
sudo cp config/qwen-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable qwen-server
```

## Configuration

### Current Setup
- **Model**: Q3_K_XL (35.78 GB, 3.86 BPW)
- **Context**: 128k tokens (131,072)
- **VRAM Usage**: ~44.6 GB / 48 GB
- **Generation Speed**: ~34.6 tokens/second
- **Flash Attention**: Enabled

### Increase Context to 256k
Edit start scripts and change:
```bash
--ctx-size 131072  →  --ctx-size 262144
```
Note: May require KV cache quantization or smaller model.

### Switch to Q4 Model (Better Quality)
Edit start scripts:
```bash
Qwen3-Coder-Next-UD-Q3_K_XL.gguf  →  Qwen3-Coder-Next-UD-Q4_K_XL.gguf
--ctx-size 131072  →  --ctx-size 65536  # Reduce context
```

## API Usage

### Completions (OpenAI-compatible)
```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "def fibonacci(n):\n",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Chat
```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a Python function to check if a number is prime"}
    ],
    "max_tokens": 200
  }'
```

### Python
```python
import openai

openai.api_base = "http://localhost:8080/v1"
openai.api_key = "dummy"

response = openai.ChatCompletion.create(
    model="Qwen3-Coder-Next",
    messages=[{"role": "user", "content": "Write hello world in Python"}]
)
print(response.choices[0].message.content)
```

## Monitoring

### GPU Usage
```bash
wsl nvidia-smi
wsl watch -n 1 nvidia-smi  # Real-time
```

### Server Logs
```bash
wsl tail -f ~/qwen-server.log
```

### Metrics
```bash
curl http://localhost:8080/metrics
curl http://localhost:8080/slots
```

## Troubleshooting

### Server won't start
```bash
# Check port
wsl netstat -tulpn | grep 8080

# Check CUDA
wsl nvidia-smi

# Check logs
wsl tail -f ~/qwen-server.log
```

### Out of memory
- Reduce context size: `--ctx-size 65536`
- Switch to Q3 or Q2 model
- Close other GPU applications

### Slow inference
- Check GPU utilization: `wsl nvidia-smi`
- Ensure Flash Attention enabled (check logs)
- Verify both GPUs are being used

## Documentation

- [SETUP_GUIDE.md](docs/SETUP_GUIDE.md) - Complete setup walkthrough
- [IMPROVEMENTS.md](docs/IMPROVEMENTS.md) - Web UIs, monitoring, security
- [claude.md](docs/claude.md) - Detailed learnings and architecture notes

## Architecture

- **Model**: Qwen3-Coder-Next (80B MoE, 512 experts, 10 active)
- **Backend**: llama.cpp with CUDA support
- **Quantization**: GGUF Q3_K or Q4_K
- **Context**: RoPE with 5M base frequency
- **Attention**: Flash Attention v2
- **Multi-GPU**: Automatic layer distribution

## Links

- Qwen Model: https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Unsloth Docs: https://unsloth.ai/docs

## License

Configuration files and scripts: MIT
Qwen3 Model: Apache 2.0
llama.cpp: MIT
