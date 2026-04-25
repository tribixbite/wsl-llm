# WSL-LLM

Local LLM inference stack for WSL2 with NVIDIA GPUs. Runs **Qwen3.6-35B-A3B at ~102 t/s with full 262k context** on a single RTX 3090, using [Madreag's TurboQuant CUDA fork](https://github.com/Madreag/turbo3-cuda) (turbo3 KV cache compression).

**Stack:** Madreag turboquant → LiteLLM proxy → Open WebUI (+ optional Cloudflare tunnel)

See [docs/QWEN36_BENCHMARKS.md](docs/QWEN36_BENCHMARKS.md) for the comprehensive benchmark report behind the engine and quant choices.

## Quick Start

```bash
git clone https://github.com/YOUR_USER/wsl-llm.git
cd wsl-llm
./install.sh
```

The installer handles everything: builds engines, downloads the model, sets up Docker containers, installs systemd services, and configures passwordless management.

### Options

```bash
./install.sh --admin-user myname --admin-password mypass     # Set admin credentials
./install.sh --domain llm.example.com                         # Enable Cloudflare tunnel
./install.sh --skip-build --skip-model                        # Config-only reinstall
./install.sh --yes                                            # Non-interactive
```

## What Gets Installed

| Component | Port | Description |
|-----------|------|-------------|
| Madreag turboquant | 8080 | Inference server (turbo3 KV → full 262k ctx at ~100 t/s) |
| LiteLLM | 4000 | OpenAI-compatible proxy with API key management |
| Open WebUI | 3000 | ChatGPT-like web interface |
| PostgreSQL | 5434 | LiteLLM database (Docker) |
| Cloudflare tunnel | — | Optional public access |
| Watchdog | — | Auto-restarts on CUDA crashes |

All services auto-start on boot via systemd.

## Management

```bash
llm status              # All services + GPU state
llm health              # Health check endpoints
llm restart             # Restart llama-server
llm logs                # Tail server logs
llm config edit         # Edit server config (context, model, sampling)
llm bench quick         # Performance benchmark
llm model list          # Available models
llm model switch FILE   # Switch model
llm key create USER     # Create API key for a user
llm info                # System info (GPU, PCIe, engines)
llm update              # Pull latest + rebuild
```

## Hardware Requirements

- **GPU:** 1-2 NVIDIA GPUs with 24GB+ VRAM each (tested: 2x RTX 3090)
- **CPU:** Any modern x86_64 (tested: Ryzen 9 5900X)
- **RAM:** 32GB+ recommended
- **OS:** Windows 11 + WSL2 Ubuntu 22.04+
- **CUDA:** Toolkit 12.x (driver provided by Windows)

## Configuration

Edit `~/llama-server.conf` then `llm restart`:

```bash
LLAMA_BIN=~/llama-cpp-turboquant/llama-server   # Madreag fork
MODEL=~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
CONTEXT_SIZE=262144      # Full long context, single slot
NUM_SLOTS=1
KV_TYPE_K=turbo3         # TurboQuant 3.125 bpv (~5x compression vs q8_0)
KV_TYPE_V=turbo3
REASONING_BUDGET=0       # 0=thinking off by default
```

### Sampling Presets (Qwen/Unsloth Official)

| Mode | temp | top_p | top_k | min_p | presence_penalty |
|------|-----:|------:|------:|------:|-----------------:|
| Non-thinking coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Non-thinking general | 0.7 | 0.8 | 20 | 0.0 | 1.5 |
| Thinking coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking general | 1.0 | 0.95 | 20 | 0.0 | 1.5 |

## API Usage

```bash
# Direct (llama-server)
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Hello"}]}'

# Via LiteLLM (recommended — rate limiting, key management)
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-litellm-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Hello"}]}'
```

## Windows Setup

After install, run in an admin PowerShell once:

```powershell
.\windows\setup-autostart.ps1    # Auto-start WSL + port forwarding on boot
```

### BIOS Recommendations (for stability)

- Disable ASPM (prevents GPU idle crash)
- Enable Above 4G Decoding + ReBAR
- Increase TDR timeout: see `windows/setup-windows.ps1`

## Uninstall

```bash
./uninstall.sh
```

## Docs

- **[Qwen 3.6 35B-A3B Benchmark Report](docs/QWEN36_BENCHMARKS.md)** ⭐ — Comprehensive benchmark of all 5 TurboQuant forks, plain vs imatrix vs UD-Q4_K_XL quants, and from-scratch install commands
- **[Qwen 3.6 27B Dense Benchmark Report](docs/QWEN36_27B_BENCHMARKS.md)** ⭐ — Madreag fork + vLLM MTP speculative decoding, 11 configs tested, 54 t/s speed champion via vLLM + Lorbus AutoRound INT4 + MTP n=3
- [35B-A3B GoL HTMLs](bench/results/gol_full/) — 8 quant/engine variants
- [27B Dense GoL HTMLs](bench/results/qwen36-27b/gol/) — 10 backend×config variants
- [Architecture](docs/architecture.md) — System design and lessons learned
- [Networking](docs/networking.md) — LAN access, firewall, tunnel setup
- [Earlier Benchmarks](docs/BENCHMARKS.md) — Qwen 3.5 era results
- [Earlier Optimization](docs/OPTIMIZATION_FINDINGS.md) — Qwen 3.5 findings

## License

Scripts and configuration: MIT. Models: see respective licenses (Qwen: Apache 2.0).
