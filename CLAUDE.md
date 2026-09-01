# WSL LLM Project - Claude Instructions

## ⚠️ This repo now covers TWO machines — check which one you are on first

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

| Reports | Machine | Notes |
|---|---|---|
| `RTX 3090`, `8.6` (×2) | **Desktop** (`matilda`) | Everything below the next section describes this box |
| `RTX 5080 Laptop`, `12.0` | **Legion laptop** (`will`) | See `docs/QWEN38_27B_LEGION_BENCHMARKS.md` — the sections below mostly DO NOT apply |

### Legion RTX 5080 Laptop quick reference (16 GB, Blackwell sm_120)

- Model: `~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf` (12.24 GiB, arch `qwen35`)
- Engine: **mainline llama.cpp** at `~/llama.cpp/build/bin/` — no fork needed, unlike Qwen3.6 on the desktop
- Build with `-DCMAKE_CUDA_ARCHITECTURES=120` and **CUDA 12.8** (12.6 has no sm_120); cmake auto-promotes to `120a`
- **`--parallel 1` is MANDATORY.** The default of 4 slots pushes VRAM to 15.9 GiB, and WSL2 has no OOM
  guardrail — WDDM silently evicts the model to system RAM and decode collapses ~700× (39.8 → 0.04 t/s).
  Keep peak VRAM ≤ ~14.7 GiB and validate every config with a *timed generation*, never just "it loaded".
- **Use the MTP draft head** (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf`): 1.89× overall, 2.14× on code,
  with no measurable accuracy cost (n=102/arm, p=0.59). n-gram speculation gives nothing here.
- **Keep CUDA graphs ON** (+20%); llama.cpp#27330's laptop-Blackwell hang did not reproduce.
- **Power mode matters enormously.** Fn+Q → Performance takes the GPU from ~90 W to 175 W:
  +32% decode, +42% prefill, and the memory clock stops throttling 14001 → 9001 MHz.
  `nvidia-smi -pl` cannot set it; it is a Windows-side/Lenovo setting.
- Sampling for Qwen3.8 differs from Qwen3.6: **thinking** temp 1.0 / top_p 0.95;
  **non-thinking** temp 0.7 / top_p 0.80 / presence_penalty 1.5. `reasoning_effort` is
  `xhigh` (default) | `medium` | `low` — `xhigh` costs ~220 s/exercise, use `medium`.
- vLLM/SGLang are **not usable** for this model here: every 4-bit safetensors quant is 17.7–21.8 GiB.
- **Use the CURRENT `UD-Q3_K_XL`, not revision `408fcc18`.** Despite V3 having 24 two-bit
  tensors (7.33% of params), it beats 408fcc18 on mean KLD (0.0248 vs 0.0271), top-1 agreement
  (93.1% vs 92.8%) and RMS Δp vs a Q8_0 reference — and is 0.28 GiB smaller and 17–30% faster
  (408 is 90.4% i-quant, which dequantizes slower on CUDA). The forum complaint is not supported.
- ExLlamaV3 ties llama.cpp on quality (38.2% pass@2 both) and beats its baseline by 11%, but
  **llama.cpp + MTP is ~70% faster than ExLlamaV3** — no EXL3 MTP draft head exists.
- ⚠️ `aider/benchmark/benchmark.py` calls `random.shuffle` **unseeded** before `--num-tests`,
  so every run tests a different random subset. Pin with `--keywords` before any A/B.
- **Report pass@2, not pass@1** — worth ~2.2x. Best scores: **66.7% pass@2** on the official
  aider polyglot (diff format, py/js/java, 30 tests) and 63.3% (py/go/rust); the house
  whole-file harness gives 58.8% thinking / 38.2% non-thinking.
- **Best throughput: 88.6 t/s** (Windows, MTP, 16k ctx q8_0 KV), 48.1 t/s with an image in
  context. `--no-mmproj-offload` keeps the projector on CPU so **vision and MTP coexist**.
- **Just use the launchers**: `windows/start-qwen38.ps1` (+ `install-autostart.ps1` for a logon
  task) or `scripts/serve-qwen38.sh`. Modes: `both` (default) | `fast` | `vision`.
- ⚠️ **This machine bugchecks randomly** — `nt` and `clipsp.sys` access violations across
  unrelated subsystems/processes, i.e. memory corruption, almost certainly the aftermarket
  2x64 GB DDR5-5600 kit (marginal for Arrow Lake HX). NOT caused by the GPU workload; it
  predates it. Always checkpoint long runs — `aider_lite` appends per-exercise JSONL and
  resumes exactly. See `docs/QWEN38_27B_LEGION_BENCHMARKS.md` §10.

---

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
| **Qwen3.6-35B-A3B** ⭐ (daily driver) | `~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | 22.4 GiB | qwen3.6 MoE (10 attn + 30 DeltaNet, head_dim=128, 3B active) | **Madreag turboquant** |
| Qwen3.6-27B (deep-thinking option) | `~/models/Qwen3.6-27B-UD-Q4_K_XL.gguf` | 17.6 GiB | qwen3.6 Dense hybrid (16 attn + 48 DeltaNet, head_dim=256, 27B all-active) | Madreag turboquant or vLLM+MTP |
| Qwen3.6-27B (smaller/faster) | `~/models/Qwen3.6-27B-IQ4_XS.gguf` | 15.4 GiB | same arch | Madreag turboquant |
| Qwen3.6-27B (vLLM MTP target) | `~/models/Lorbus-Qwen3.6-27B-int4-AutoRound/` | ~14 GiB | INT4 + BF16 MTP head | **vLLM nightly** |
| Qwen3.6-35B-A3B (legacy plain Q4) | `~/models/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf` (bartowski) | 19 GiB | qwen3.6 MoE | any |
| Qwen3.5-35B-A3B (legacy) | `~/models/qwen35-q4.gguf` | 20.7 GiB | qwen35moe | ik_llama.cpp |
| Qwen3-Coder-Next | `~/models/coder-next-q4.gguf` | 41.5 GiB | qwen3next | llama.cpp |

**When to use which model:**
- Qwen3.6-35B-A3B: daily coding, chat, edits — ~102 t/s @ 262k ctx via Madreag turboquant. 3B active params makes it very fast.
- Qwen3.6-27B + Madreag: hard one-shot coding tasks where quality matters — ~20 t/s but Sonnet 4.6-tier output (per Qwen). Same simple stack as 35B.
- **Qwen3.6-27B + vLLM + Genesis** ⭐: best 27B throughput — **~67 t/s code / 46 t/s prose / 70 peak** decode_TPS (proper streaming bench, enable_thinking=false). Launch via `scripts/serve-27b-vllm-genesis.sh`. See [skills/qwen36-27b-vllm-genesis.md](skills/qwen36-27b-vllm-genesis.md). Bandwidth-bound at ~67 t/s; 80 t/s requires custom EAGLE-3 head (none published yet).

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

- `docs/QWEN36_BENCHMARKS.md` — **Comprehensive Qwen 3.6 35B-A3B benchmark report** including Madreag turboquant fork analysis, 5 TurboQuant fork comparison, and from-scratch install commands ⭐
- `docs/QWEN36_27B_BENCHMARKS.md` — **Qwen 3.6 27B Dense benchmark report**: 11 configs across Madreag fork + vLLM nightly + MTP speculative decoding, with vLLM+MTP@262k as speed champion (~54 t/s) ⭐
- `bench/results/gol_full/` — 35B-A3B Conway's Game of Life HTML outputs (8 fork variants)
- `bench/results/qwen36-27b/gol/` — 27B Dense Conway's Game of Life HTML outputs (10 backend×config variants)
- `docs/BENCHMARKS.md` — Earlier benchmark results (Qwen 3.5)
- `docs/OPTIMIZATION_FINDINGS.md` — What we tried and what works (Qwen 3.5)
- `docs/architecture.md` — System design, VRAM budget, CUDA crash analysis
- `docs/networking.md` — LAN access, firewall, Cloudflare tunnel
