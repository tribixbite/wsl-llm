# WSL LLM Project - Claude Instructions

## Project Location
- **Repo**: `~/git/wsl-llm` (WSL Ubuntu-22.04, user `matilda`)
- **Models**: `~/models/` (native Linux FS)
- **llama-cpp-turboquant** (Madreag fork): `~/llama-cpp-turboquant/llama-server` — **PRIMARY engine** (turbo3 KV cache, full 262k ctx)
- **Source for above**: `~/llama-cpp-turboquant-src/` (built by `scripts/build-turboquant.sh`)
- **ik_llama.cpp**: `~/ik_llama.cpp` (fallback, built from source)
- **llama.cpp**: `~/llama.cpp` (upstream fallback, built from source)
- **Python venv**: `~/bench_env` (PyTorch, vLLM nightly, litellm)
- **Install metadata**: `~/.config/wsl-llm/install.env` (secrets, paths — chmod 600)

## Hardware
- 2x RTX 3090 (24GB each) — inference on GPU 0 only (CUDA_VISIBLE_DEVICES=0), GPU 1 free; Compute 8.6, PCIe (no NVLink)
- Ryzen 9 5900X (12c/24t), 64GB DDR4, ASUS ROG STRIX X570-E
- Windows 11 + WSL2 Ubuntu 22.04
- NVIDIA Driver 591.86, CUDA 13.1, Toolkit 12.6

## Quick Reference

```bash
# Install from scratch
./install.sh --admin-user myname --admin-password mypass

# Day-to-day management (no sudo needed after install)
llm status              # All services + GPU state
llm health              # Health check endpoints
llm restart             # Restart llama-server
llm logs                # Tail server logs
llm config edit         # Edit ~/llama-server.conf
llm bench quick         # Performance benchmark
llm model list          # Available models
llm key create USER     # Create API key
llm info                # System info (GPU, PCIe, engines)
```

## Current Models

| Model | File | Size | Architecture | Engine |
|-------|------|------|-------------|--------|
| **Qwen3.6-35B-A3B** ⭐ | `~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | 22.4 GiB | qwen3.6 MoE (10 attn + 30 DeltaNet, head_dim=128) | **Madreag turboquant** |
| Qwen3.6-35B-A3B (legacy plain Q4) | `~/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` (bartowski) | 19 GiB | qwen3.6 MoE | any |
| Qwen3.5-35B-A3B (legacy) | `~/models/qwen35-q4.gguf` | 20.7 GiB | qwen35moe | ik_llama.cpp |
| Qwen3-Coder-Next | `~/models/coder-next-q4.gguf` | 41.5 GiB | qwen3next | llama.cpp |

## Services (systemd, autostart on boot)

| Service | Port | Description |
|---------|------|-------------|
| `llama-server` | 8080 | Madreag turboquant inference server (Qwen 3.6, turbo3 KV, 262k ctx) |
| `litellm` | 4000 | LiteLLM OpenAI-compatible proxy + admin UI |
| `cloudflared` | — | Cloudflare tunnel (optional) |
| `llama-server-watchdog` | — | Auto-restarts on CUDA crashes (timer) |
| `litellm-pg` (Docker) | 5434 | PostgreSQL for LiteLLM |
| `open-webui` (Docker) | 3000 | ChatGPT-like web interface (--network host) |
| `mcpo` (Docker) | 8000 | MCP-to-OpenAPI bridge for Open WebUI |

## Configuration

Server config lives at `~/llama-server.conf` (generated from `config/llama-server.conf.template`).
Edit with `llm config edit`, then `llm restart`.

Key parameters: `LLAMA_BIN`, `MODEL`, `CONTEXT_SIZE`, `NUM_SLOTS`, `KV_TYPE_K` (=`turbo3`), `KV_TYPE_V`, `REASONING_BUDGET`, `LLAMA_API_KEY`, `CUDA_VISIBLE_DEVICES`, `GPU_LAYERS`, `FLASH_ATTENTION`, `EXTRA_FLAGS`.

`EXTRA_FLAGS` defaults to `--jinja --reasoning-format deepseek`. Reasoning is disabled by default (`REASONING_BUDGET=0`) and re-enabled per-request via `chat_template_kwargs`.

## Repo Structure

```
wsl-llm/
├── install.sh                  # Main setup (--admin-user, --admin-password, --domain, etc.)
├── uninstall.sh                # Clean removal
├── bin/llm                     # CLI tool (symlinked to ~/.local/bin/llm)
├── config/
│   ├── llama-server.conf.template    # → ~/llama-server.conf
│   ├── litellm_config.yaml.template  # → ~/litellm_config.yaml
│   ├── cloudflared.yml.template      # → ~/.cloudflared/config.yml
│   ├── mcpo/config.json.template     # → ~/mcpo-config.json
│   ├── wsl.conf                      # Reference WSL config
│   ├── .wslconfig                    # Reference Windows-side config
│   └── bash_profile                  # CUDA env setup
├── services/
│   ├── llama-server.service.template
│   ├── llama-server.sh               # Wrapper (reads ~/llama-server.conf, exports CUDA_VISIBLE_DEVICES, supports GPU_LAYERS=fit)
│   ├── llama-server-watchdog.service.template
│   ├── llama-server-watchdog.timer
│   ├── llama-server-watchdog.sh.template
│   ├── litellm.service.template
│   └── cloudflared.service.template
├── scripts/
│   ├── build-turboquant.sh     # Clone + build Madreag turboquant fork (PRIMARY)
│   ├── build-engines.sh        # Clone + build ik_llama.cpp & llama.cpp (fallback)
│   ├── download-model.sh       # HuggingFace model download
│   ├── setup-docker.sh         # PostgreSQL + Open WebUI + mcpo containers
│   └── setup-venv.sh           # Python venv (vLLM, PyTorch, litellm)
├── bench/
│   ├── bench.sh                # Unified runner (quick/full/compare)
│   ├── vllm_bench.py
│   ├── bench_eager.py
│   └── vllm_config_bench.sh
├── skills/
│   ├── vite-svelte-app.md      # Vite + Svelte 5 + TS + Tailwind conventions
│   ├── cloudflare-worker.md    # CF Workers + Wrangler + Hono patterns
│   └── code-review.md          # Review checklist and feedback format
├── docs/
│   ├── BENCHMARKS.md           # Full performance results
│   ├── OPTIMIZATION_FINDINGS.md
│   ├── architecture.md         # System design, VRAM budget, lessons learned
│   └── networking.md           # LAN, firewall, tunnel setup
└── windows/
    ├── setup-autostart.ps1     # Windows Task Scheduler for WSL boot
    ├── setup-windows.ps1       # TDR fix, firewall rules, BIOS reminders
    ├── START_ALL.ps1
    ├── START_ALL.bat
    └── gpu-init.bat              # Sets both GPUs to 200W power limit
```

## Template System

Templates use `{{VARIABLE}}` placeholders, substituted by `install.sh` via `sed`.
Variables: `{{HOME}}`, `{{USER}}`, `{{REPO_DIR}}`, `{{LLAMA_API_KEY}}`, `{{LITELLM_MASTER_KEY}}`,
`{{LITELLM_SALT_KEY}}`, `{{DB_PASSWORD}}`, `{{WEBUI_SECRET_KEY}}`, `{{ADMIN_USER}}`,
`{{ADMIN_PASSWORD}}`, `{{ADMIN_EMAIL}}`, `{{MODEL_PATH}}`, `{{DOMAIN}}`, `{{TUNNEL_UUID}}`.

Secrets auto-generated with `openssl rand` if not provided as install args.

## Critical Notes

- **Madreag turboquant fork is the production engine** for Qwen 3.6 — turbo3 KV cache enables full 262k ctx at ~102 t/s on single 24 GB GPU. See `docs/QWEN36_BENCHMARKS.md` for the complete benchmark report.
- **`--reasoning-budget 0`** disables thinking by default; re-enable per-request with `chat_template_kwargs: {enable_thinking: true}`
- **ASPM must be disabled in BIOS** — causes CUDA crash (error 999) after idle periods
- **262k context is stable** on a single GPU thanks to turbo3 KV (3.125 bpv, ~5× compression vs q8_0); only 10/40 layers have KV cache (DeltaNet)
- **Speculative decoding is net-negative** on Qwen3.6-35B-A3B + RTX 3090 (verified via thc1006 benchmark)
- **`-sm graph` is USELESS on PCIe** — 10 t/s (only helps NVLink); always use layer split
- **vLLM TP=2 is 7x slower** on PCIe; always use TP=1
- **vLLM needs nightly 0.17.0rc1+** for Qwen3.5/3.6 support
- **Vulkan not available in WSL2** — NVIDIA only provides CUDA/D3D12
- **Model on Linux FS** — better I/O than Windows mount
- **Never commit secrets** — all secrets in templates use `{{PLACEHOLDERS}}`
- **bartowski's Q4_0 is imatrix-calibrated**, NOT a "naive" Q4_0 — meaningful quality preservation despite the simple format name

## Sampling Parameters (Official Qwen/Unsloth)

Do NOT use greedy decoding (temp=0). Model performance degrades.

| Mode | temp | top_p | top_k | min_p | presence_penalty |
|------|-----:|------:|------:|------:|-----------------:|
| **Non-thinking coding** (server default) | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| **Non-thinking general** | 0.7 | 0.8 | 20 | 0.0 | 1.5 |
| **Thinking coding** | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| **Thinking general** | 1.0 | 0.95 | 20 | 0.0 | 1.5 |

## Build Commands

```bash
# Primary engine (Madreag turboquant fork — production)
./scripts/build-turboquant.sh

# Fallback engines (ik_llama.cpp + upstream llama.cpp)
./scripts/build-engines.sh

# Manual Madreag build (requires cmake 3.25+)
cd ~/llama-cpp-turboquant-src
~/.local/bin/cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DLLAMA_CURL=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
~/.local/bin/cmake --build build --target llama-server -j$(nproc)
install -m 0755 build/bin/llama-server ~/llama-cpp-turboquant/llama-server
```

## External Paths

```
~/llama-cpp-turboquant/                          # Madreag fork (PRIMARY) — installed binary
~/llama-cpp-turboquant-src/                      # Madreag fork — source checkout
~/ik_llama.cpp/                                  # ik_llama.cpp fork (fallback)
~/llama.cpp/                                     # Upstream llama.cpp (fallback)
~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf         # Primary model (native Linux FS, 22 GB)
~/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf          # bartowski imatrix-Q4_0 (legacy comparison)
~/bench_env/                                     # Python venv (vLLM, PyTorch)
~/llama-server.conf                              # Server config (generated from template)
~/litellm_config.yaml                            # LiteLLM config (generated from template)
~/mcpo-config.json                               # mcpo config (generated from template)
~/.cloudflared/                                  # Cloudflare tunnel config + credentials
~/.config/wsl-llm/                               # Install metadata (install.env)
```

## Dev Tools

- **bun** 1.3.x: `~/.bun/bin/bun` (JS/TS runtime, package manager)
- **gh** 2.89.x: `~/.local/bin/gh` (GitHub CLI)
- **wrangler** 4.x: `~/.bun/bin/wrangler` (Cloudflare Workers CLI)

## MCP Tool Servers (Open WebUI)

| Server | Type | URL | Notes |
|--------|------|-----|-------|
| DeepWiki | MCP (Streamable HTTP) | `https://mcp.deepwiki.com/mcp` | Direct, no auth |
| GitHub | MCP (Streamable HTTP) | `https://api.githubcopilot.com/mcp/` | Bearer (GitHub PAT) |
| Context7 | OpenAPI (via mcpo) | `http://localhost:8000/context7` | Library docs |
| PAL | OpenAPI (via mcpo) | `http://localhost:8000/pal` | Routes through LiteLLM |

## Open WebUI Features

- **Web Search**: DuckDuckGo (no API key needed)
- **Code Execution**: Pyodide (browser-based, zero infra)
- **Function Calling**: Native (set per-model)
- **Skills**: Importable from `skills/` dir via Workspace > Skills

## Reference Docs

- `docs/QWEN36_BENCHMARKS.md` — **Comprehensive Qwen 3.6 benchmark report** including Madreag turboquant fork analysis, 5 TurboQuant fork comparison, and from-scratch install commands ⭐
- `bench/results/gol_full/` — Conway's Game of Life HTML outputs from 8 quant/engine variants (visual comparison of generation quality)
- `docs/BENCHMARKS.md` — Earlier benchmark results (Qwen 3.5)
- `docs/OPTIMIZATION_FINDINGS.md` — What we tried and what works (Qwen 3.5)
- `docs/architecture.md` — System design, VRAM budget, CUDA crash analysis
- `docs/networking.md` — LAN access, firewall, Cloudflare tunnel
