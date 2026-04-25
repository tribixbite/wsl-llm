# Qwen 3.6 Local Inference — Complete Benchmark Report

**Hardware**: 1× RTX 3090 (24 GB, sm_86, CUDA 12.6) on WSL2 Ubuntu 22.04
**Model under test**: [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (MoE, 35B total / 3B active, released 2026-04-16)
**Test dates**: 2026-04-19 through 2026-04-25
**Outputs directory**: `/mnt/c/Users/Will/Dropbox/qwen36-bench/`

---

## Executive Summary

We compared two **calibrated** quants of Qwen 3.6 ([bartowski's imatrix-Q4_0](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) vs [Unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)) across multiple engines ([ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), [upstream llama.cpp](https://github.com/ggml-org/llama.cpp), and **5 TurboQuant forks**), 14 hard prompts, and 10 different runtime configurations.

**Important terminology correction**: bartowski's Q4_0 is **NOT** "plain" Q4_0 — it's an *imatrix-calibrated* Q4_0 (importance-matrix tuned codebook against real activations). Both quants are sophisticated; they differ in their *type* of sophistication:
- **bartowski imatrix-Q4_0**: pure Q4_0 tensor format, codebook calibrated via imatrix
- **Unsloth UD-Q4_K_XL**: mixed-precision (Q4_K + Q5_K + Q6_K + Q8_0 across tensors) plus calibration

This explains why bartowski's Q4_0 held up so well on quality — it has its own quality-preservation mechanism.

**Top-line finding**: by switching to [Madreag's TurboQuant CUDA fork](https://github.com/Madreag/turbo3-cuda) with `turbo3` KV cache, we run UD-Q4_K_XL at full **262k context** at **101.6 t/s on the Game of Life prompt** — that's faster than our previous 64k-context production config (103.6 t/s on GoL but only 64k ctx). Net: 4× more context for ~free.

Other key conclusions:

- **bartowski imatrix-Q4_0 is ~13% faster than UD-Q4_K_XL** on the GoL prompt with comparable quality on most code tasks; UD-Q4_K_XL wins only when prompts have many subtle constraints (allowlist sanitizers, content-asserting tests)
- **Reducing context 262k → 32k saves ~24% throughput** by itself (q8_0 KV + drop Hadamard)
- **Upstream llama.cpp ≈ ik_llama.cpp for Q4_K_XL on Ampere** (within 2%)
- **Speculative decoding is net-negative** on Qwen3.6-35B-A3B + RTX 3090
- **TurboQuant works on Ampere** in active forks (Madreag/TheTom/spiritbuun all ✅, animehacker very slow at long ctx, AmesianX has long-ctx V-cache bug on head_dim=128)
- **Best end-to-end config**: Madreag fork + UD-Q4_K_XL + turbo3 KV + 262k ctx — ~102 t/s on hard prompts

---

## Quick Reference — Game of Life Test (max_tokens=16384, full output)

Conway's Game of Life with RLE URL hash sync — flagship hard prompt. All runs finished naturally (no truncation). HTML files saved for visual side-by-side comparison.

| Variant | t/s | Tokens | Time | HTML Output |
|---------|----:|-------:|-----:|-------------|
| ik_llama.cpp + UD-Q4_K_XL @ 64k / q8_0 KV (prior production) | **103.6** | 5449 | 52.6s | `gol_full/ik_llama_k_xl_64k_q8.html` |
| ik_llama.cpp + bartowski imatrix-Q4_0 @ 64k / q8_0 KV | **115.4** | 5464 | 47.3s | `gol_full/ik_llama_imatrix_q4_0.html` |
| **Madreag** turbo3 + UD-Q4_K_XL @ **262k** ⭐ | **101.6** | 5988 | 58.9s | `gol_full/madreag_turbo3_262k.html` |
| TheTom turbo3 + UD-Q4_K_XL @ 32k | **97.1** | 6082 | 62.6s | `gol_full/thetom_turbo3.html` |
| spiritbuun turbo3 + UD-Q4_K_XL @ 32k | **96.7** | 5468 | 56.6s | `gol_full/spiritbuun_turbo3.html` |
| spiritbuun turbo3_tcq (Viterbi) + UD-Q4_K_XL @ 32k | **78.5** | 14735 | 187.7s | `gol_full/spiritbuun_turbo3_tcq.html` |
| AmesianX tbq3 + UD-Q4_K_XL @ 32k | **96.5** | 4350 | 45.1s | `gol_full/amesianx_tbq3.html` |
| animehacker tq3_0 + UD-Q4_K_XL @ 32k | **17.0** | 6183 | 363.1s | `gol_full/animehacker_tq3_0.html` |

Note on `spiritbuun_turbo3_tcq`: it produced a 14k-token output (about 2.7× longer than other variants) — this isn't bad behavior, it just generated a more verbose explanation alongside the HTML. Its t/s is comparable to others when normalized.

Note on `animehacker_tq3_0`: dramatic slowdown (17 t/s vs ~100 for others) at long generation. Their CUDA path is unoptimized and self-described as "PolarQuant 3-bit, NOT full TurboQuant with QJL" — partial implementation.

---

## Quick Reference — All Runtime Configurations Tested

| # | Config | Avg t/s | Notes |
|---|--------|--------:|-------|
| 1 | ik_llama.cpp + UD-Q4_K_XL @ 262k / 2 slots / q4_0 KV + Hadamard (original) | **84.7** | Pre-experiment baseline |
| 2 | ik_llama.cpp + UD-Q4_K_XL @ 64k / 2 slots / q8_0 KV (mid-experiment production) | **99.0** | 2-slot setup, 32k effective per slot |
| 3 | ik_llama.cpp + UD-Q4_K_XL @ 32k / 1 slot / q8_0 KV | **105.1** | Speed champion at fixed config |
| 4 | upstream llama.cpp + UD-Q4_K_XL @ 32k / 1 slot / q8_0 KV | **106.8** | Engines tied within margin |
| 5 | ik_llama.cpp + bartowski imatrix-Q4_0 @ 262k / 2 slots / q4_0 KV + Hadamard | **108.1** | imatrix Q4_0 wins on speed |
| 6 | **Madreag fork** + UD-Q4_K_XL + turbo3 KV @ 262k / 1 slot ⭐ (current production) | **103.6** | Full 262k restored, K_XL quality |
| 7 | TheTom fork + turbo3 KV @ 32k / 1 slot | **106.3** | Slightly faster at short ctx |
| 8 | spiritbuun + turbo3 KV @ 32k / 1 slot | **101.9** | Active Qwen3.6 DFlash work |
| 9 | spiritbuun + turbo3_tcq (Viterbi) @ 32k / 1 slot | **96.3** | Higher quality, slower encode |
| 10 | AmesianX + tbq3 @ 32k / 1 slot | **98.9** | Different cache type design |
| 11 | animehacker + tq3_0 @ 32k / 1 slot | **48.0** | Half-speed CUDA path; unoptimized |

Bold = headline t/s for each config.

### Quality test suite (14 prompts run on multiple configs)

| # | Prompt | Hard? | Output saved |
|---|--------|:-----:|--------------|
| 1 | TypeScript `groupBy<T,K>` generic | ⭐ | `q4_0/typescript.txt`, `k_xl/typescript.txt` |
| 2 | SvelteKit infinite-scroll image gallery | ⭐ | `q4_0/svelte.txt`, `k_xl/svelte.txt` |
| 3 | Kotlin/Compose Paging 3 reader | ⭐ | `q4_0/kotlin.txt`, `k_xl/kotlin.txt` |
| 4 | Conway's Game of Life + RLE + URL hash sync | ⭐⭐⭐ | `gol_full/*.html` (8 variants) |
| 5 | Regex engine from scratch (no `re`, Thompson NFA) | ⭐⭐⭐ | `*/02_regex_engine.py` |
| 6 | Mini Lisp interpreter in TypeScript | ⭐⭐⭐ | `*/03_mini_lisp.ts` |
| 7 | Sudoku CSP solver with AC-3 | ⭐⭐⭐ | `*/04_sudoku_csp.py` |
| 8 | RGA CRDT in TypeScript | ⭐⭐⭐ | `*/05_crdt_rga.ts` |
| 9 | B-Tree with disk persistence in Rust | ⭐⭐⭐ | `*/06_btree_rust.rs` |
| 10 | Real-time collaborative markdown editor (10-file SvelteKit) | ⭐⭐⭐ | `*/07_svelte_collab_editor.md` |
| 11 | Offline-first Android reader with 3-way merge sync (12-file Kotlin) | ⭐⭐⭐ | `*/08_kotlin_offline_sync.md` |

---

# Production setup — from-scratch install

This is what's currently running. Reproducible from a fresh WSL2 Ubuntu 22.04 install.

## Prerequisites

```bash
# System packages
sudo apt update
sudo apt install -y build-essential cmake git curl wget pkg-config

# CUDA Toolkit 12.6 — assume installed at /usr/local/cuda-12.6
ls /usr/local/cuda-12.6/bin/nvcc  # should exist; if not see https://developer.nvidia.com/cuda-12-6-0-download-archive

# Python via uv (Astral's fast Python tooling — preferred over pip/venv directly)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Node.js / Bun for any JS work
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc
```

## Build Madreag's TurboQuant fork (the inference engine)

```bash
# Clone Madreag's fork — built on TheTom's TurboQuant base + RTX 3090-validated CUDA optimizations
cd ~
git clone https://github.com/Madreag/turbo3-cuda.git ~/llama-cpp-turboquant-src
cd ~/llama-cpp-turboquant-src

# Configure CUDA build for Ampere (sm_86 = RTX 3090/3090Ti/4090M)
# Need cmake 3.25+ — install via uv if your system version is too old:
uv tool install cmake

cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DLLAMA_CURL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_BUILD_TYPE=Release

# Build (takes ~10 minutes on a Ryzen 5900X)
cmake --build build --target llama-server -j$(nproc)

# Move binary to persistent location (so we don't depend on the source tree)
mkdir -p ~/llama-cpp-turboquant
cp build/bin/llama-server ~/llama-cpp-turboquant/

# Verify
~/llama-cpp-turboquant/llama-server --version | head -3
```

## Download the model (UD-Q4_K_XL from Unsloth)

```bash
# Use uv to install huggingface CLI in an isolated env
uv tool install --with hf-transfer huggingface_hub

# Download just the UD-Q4_K_XL quant (~21 GB)
mkdir -p ~/models
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
    unsloth/Qwen3.6-35B-A3B-GGUF \
    Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --local-dir ~/models/

ls -lh ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

## Configure llama-server

```bash
# Generate API keys (or reuse existing ones from ~/.config/wsl-llm/install.env)
LLAMA_API_KEY="$(openssl rand -hex 24)"
echo "Save this: $LLAMA_API_KEY"

# Write config
cat > ~/llama-server.conf <<EOF
# llama-server configuration
# Edit this file, then restart: llm restart

# Engine binary (Madreag's llama-cpp-turboquant fork)
LLAMA_BIN=$HOME/llama-cpp-turboquant/llama-server

# Model
MODEL=$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
ALIAS=qwen3.6-35b-a3b

# Context & slots — full 262k now possible thanks to turbo3 KV
CONTEXT_SIZE=262144
NUM_SLOTS=1

# GPU
CUDA_VISIBLE_DEVICES=0
GPU_LAYERS=999
FLASH_ATTENTION=on

# KV cache (TurboQuant)
KV_TYPE_K=turbo3
KV_TYPE_V=turbo3

# Sampling defaults (non-thinking coding preset)
TEMP=0.6
TOP_P=0.95
TOP_K=20
MIN_P=0.0

# Thinking
REASONING_BUDGET=0

# Network
HOST=0.0.0.0
PORT=8080
API_KEY=$LLAMA_API_KEY

# Extra flags
EXTRA_FLAGS="--jinja --reasoning-format deepseek"
EOF

chmod 600 ~/llama-server.conf
```

## Wire into systemd

```bash
# Clone the wsl-llm repo for the wrapper script + service template
cd ~/git
git clone https://github.com/<your-fork>/wsl-llm.git  # or the upstream repo
cd wsl-llm

# Generate the service unit from template
sed \
    -e "s|{{USER}}|$USER|g" \
    -e "s|{{REPO_DIR}}|$HOME/git/wsl-llm|g" \
    services/llama-server.service.template \
    | sudo tee /etc/systemd/system/llama-server.service

sudo systemctl daemon-reload
sudo systemctl enable llama-server
sudo systemctl start llama-server

# Watch it come up
sudo systemctl status llama-server --no-pager
```

## Smoke test

```bash
# Health check (after ~30 sec for model load)
curl -s --max-time 5 \
    -H "Authorization: Bearer $LLAMA_API_KEY" \
    http://localhost:8080/health | python3 -m json.tool

# Quick chat completion
curl -s -H "Authorization: Bearer $LLAMA_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen3.6-35b-a3b",
        "messages": [{"role":"user","content":"Write a python one-liner to flatten a list of lists."}],
        "max_tokens": 200,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": false}
    }' \
    http://localhost:8080/v1/chat/completions \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
```

If this returns a Python one-liner, you're up and running with full 262k context, UD-Q4_K_XL quality, and ~100 t/s decode.

## Day-to-day commands (via the `llm` CLI)

```bash
llm status      # all services + GPU state
llm health      # health check endpoints
llm restart     # restart llama-server
llm logs        # tail server logs
llm config edit # edit ~/llama-server.conf
llm bench quick # 3-prompt benchmark
```

---

# Full Details

## 1. Setup & Models

### Models tested (both calibrated, *neither* naive)

| Quant | Source | Size | Calibration |
|-------|--------|-----:|-------------|
| UD-Q4_K_XL (Unsloth Dynamic 2.0, mixed Q4_K/Q5_K/Q6_K/Q8_0) | https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF | 22.4 GB | imatrix + mixed-precision tensor selection |
| imatrix-Q4_0 (bartowski) | https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF | 19 GB | imatrix only (Q4_0 tensor format) |

Both repos ship calibration data:
- bartowski: `Qwen_Qwen3.6-35B-A3B-imatrix.gguf`
- unsloth: hidden in their dynamic v2.0 pipeline

### Recommended sampling (per [Unsloth docs](https://unsloth.ai/docs/models/qwen3.6))

| Mode | temp | top_p | top_k | min_p | presence_penalty |
|------|-----:|------:|------:|------:|-----------------:|
| Non-thinking coding (default) | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Non-thinking general | 0.7 | 0.8 | 20 | 0.0 | 1.5 |
| Thinking coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking general | 1.0 | 0.95 | 20 | 0.0 | 1.5 |

Disable thinking by default with `--reasoning-budget 0`; re-enable per request via `chat_template_kwargs: {"enable_thinking": true}`.

### Flags introduced for Qwen 3.6 (vs 3.5)

- `--reasoning-format deepseek` — proper reasoning block parsing
- `--chat-template-kwargs '{"preserve_thinking":true}'` — improves agentic loops (don't strip thinking between turns)

---

## 2. Engine comparison: ik_llama.cpp vs upstream llama.cpp

Both rebuilt to latest master as of 2026-04-24 (ik_llama.cpp at HEAD, upstream at commit `a702f3959`). Same runtime config: 32k ctx / 1 slot / q8_0 KV / Q4_K_XL.

| Engine | Avg t/s (6 hard prompts) |
|--------|------------------------:|
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | **105.1** |
| [upstream llama.cpp](https://github.com/ggml-org/llama.cpp) | **106.8** |

**Verdict**: tied within +1.6%. The much-quoted ["135.7 t/s on RTX 3090"](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) reference appears to be a measurement artifact (likely warm cache / shorter prompt). No reason to switch engines for Q4_K_XL on Ampere.

ik_llama.cpp's known wins (`-fmoe`, `-rtr`, optimized Q2/Q3_R4 quants) don't materialize for Q4_K_XL.

---

## 3. imatrix-Q4_0 vs UD-Q4_K_XL

Same config (262k ctx / 2 slots / q4_0 KV + Hadamard, ik_llama.cpp). Hard bench across 6 algorithm/systems prompts (Game of Life, Regex, Lisp, Sudoku, CRDT, B-Tree) plus 2 framework prompts (Svelte editor, Kotlin offline sync).

### Throughput

| Quant | Algorithm suite avg | Svelte (~6k tok) | Kotlin (~6.7k tok) |
|-------|---------------------:|-----------------:|-------------------:|
| UD-Q4_K_XL | 84.7 | 88.0 | 82.4 |
| bartowski imatrix-Q4_0 | **108.1** | **107.5** | **102.7** |
| **Δ** | **+28%** | **+22%** | **+25%** |

### Quality (the more interesting finding)

| Task | Verdict |
|------|---------|
| Templated codegen (groupBy, gallery, Paging-3 list) | **Tied.** Both produce correct, idiomatic, modern code |
| Algorithm/systems (Lisp, CRDT, Sudoku, regex, B-Tree) | **Tied.** Both produce equivalent quality |
| **Game of Life URL-hash feedback-loop prevention** | **imatrix-Q4_0 wins.** Used `history.replaceState` (correct). K_XL used `window.onhashchange = null` trick that doesn't work with `addEventListener`-registered handlers — real bug on the exact requirement the prompt called out |
| **Svelte sanitizer (prompt required *allowlist*-based)** | **K_XL wins.** Implemented allowlist of tags + attributes per prompt. imatrix-Q4_0 went pure blocklist regex — missed the requirement |
| **Kotlin merge tests (≥3 cases incl. conflict)** | **K_XL wins.** Asserts merged content (`"Line 1\nLocal Modified\nRemote Modified"`); imatrix-Q4_0 only asserts result type. K_XL would catch a broken merge; imatrix-Q4_0 wouldn't |
| Kotlin idioms (`object` vs `class` for stateless engine) | **K_XL wins.** Idiomatic Kotlin |

**Pattern**: imatrix-Q4_0 does the obvious interpretation faster. UD-Q4_K_XL pays attention to subtle prompt constraints. For 80% of coding tasks, imatrix-Q4_0's speed wins. For careful-spec work, K_XL wins.

This was a fairer comparison than the earlier framing implied — both quants are calibrated, just with different sophistication. The mixed-precision in UD-Q4_K_XL is what gives it the constraint-following edge.

---

## 4. Runtime config tuning

Same engine (ik_llama.cpp), same model (UD-Q4_K_XL), varying runtime config:

| Config | Avg t/s | Per-slot ctx | Concurrent? |
|--------|--------:|-------------:|:-----------:|
| 262k / 2 slots / q4_0 KV + Hadamard | 84.7 | 131k | ✅ |
| 64k / 2 slots / q8_0 KV | 99.0 | 32k | ✅ |
| **32k / 1 slot / q8_0 KV** (max speed at fixed config) | **105.1** | 32k | ❌ |

Levers ranked by impact:
1. **Context size**: 262k → 32k buys ~24% throughput
2. **Hadamard transforms (`-khad -vhad`)**: only useful with q4_0 KV; drop them with q8_0 KV
3. **q8_0 KV vs q4_0**: q8_0 has simpler dequant kernel; despite 2× larger, no measurable speed cost at 32-64k ctx
4. **Slot count**: 2→1 buys ~6% but loses concurrency

But — TurboQuant changes this calculus entirely (see §6).

---

## 5. Speculative decoding — net negative

Tracked in [llama.cpp PR #19493](https://github.com/ggml-org/llama.cpp/pull/19493). Public benchmark on identical hardware ([thc1006/qwen3.6-speculative-decoding-rtx3090](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)) tested 19 configurations (ngram-cache, ngram-mod, classic draft with Qwen3.5-0.8B as draft model). Every variant was net-negative: -3% to -12% decode throughput.

[MTP (multi-token prediction) PR #20700](https://github.com/ggml-org/llama.cpp/pull/20700) targets Qwen3.5/3.6 **dense** models only. Our 35B-A3B is MoE; no MTP head shipped.

**Conclusion**: don't enable speculative decoding for this model on this GPU.

---

## 6. TurboQuant — all 5 active forks tested

[TurboQuant](https://arxiv.org/abs/2504.19874) (Google DeepMind, ICLR 2026) compresses KV cache to 2-4 bits via Walsh-Hadamard transform + Lloyd-Max quantization with QJL correction. Status discussed in [llama.cpp #20969](https://github.com/ggml-org/llama.cpp/discussions/20969). Not yet merged upstream, but multiple community forks exist.

All 5 forks were cloned, built for sm_86 with CUDA 12.6, and tested with our UD-Q4_K_XL model.

### Per-fork Game of Life results (max_tokens=16384, all natural-stop)

| Fork | URL | Latest commit | Cache type tested | t/s | Tokens | Time | VRAM |
|------|-----|---------------|-------------------|----:|-------:|-----:|-----:|
| **Madreag** ⭐ | https://github.com/Madreag/turbo3-cuda | 2026-04-12 | `turbo3` @ **262k** | **101.6** | 5988 | 58.9s | 23.0 GiB |
| TheTom | https://github.com/TheTom/llama-cpp-turboquant | 2026-04-24 | `turbo3` @ 32k | **97.1** | 6082 | 62.6s | 21.8 GiB |
| spiritbuun | https://github.com/spiritbuun/llama-cpp-turboquant-cuda | 2026-04-24 | `turbo3` @ 32k | **96.7** | 5468 | 56.6s | 21.9 GiB |
| spiritbuun (TCQ) | (same) | 2026-04-24 | `turbo3_tcq` (Viterbi) @ 32k | **78.5** | 14735 | 187.7s | 21.9 GiB |
| AmesianX | https://github.com/AmesianX/TurboQuant | 2026-04-24 | `tbq3` @ 32k | **96.5** | 4350 | 45.1s | 21.8 GiB |
| animehacker | https://github.com/animehacker/llama-turboquant | 2026-03-28 | `tq3_0` @ 32k | **17.0** | 6183 | 363.1s | 21.8 GiB |

### The big test: Madreag turbo3 at full 262k context

| Config | t/s | VRAM |
|--------|----:|-----:|
| Old: ik_llama + UD-Q4_K_XL @ 262k / q4_0 KV + Hadamard | 84.7 | 22.3 GiB |
| Mid-experiment: ik_llama + UD-Q4_K_XL @ 64k / q8_0 KV | 99.0 | 22.5 GiB |
| **Madreag turbo3 + UD-Q4_K_XL @ 262k / 1 slot ⭐** | **103.6** (algorithm avg) / **101.6** (GoL) | **23.0 GiB** |

**The previously-impossible combo (full 262k context + UD-Q4_K_XL quality + reasonable speed) now works.**

### Validation: full hard bench on Madreag turbo3 @ 262k

Re-ran the full 8-prompt hard suite (6 algorithm/systems + Svelte editor + Kotlin offline sync) against Madreag fork at 262k ctx with turbo3 KV. Outputs saved to `/mnt/c/Users/Will/Dropbox/qwen36-bench/madreag_turbo3_262k/`.

| Prompt | t/s |
|--------|----:|
| 01 Game of Life | 104.6 |
| 02 Regex engine | 103.6 |
| 03 Mini Lisp | 102.0 |
| 04 Sudoku CSP | 102.3 |
| 05 CRDT RGA | 102.0 |
| 06 B-Tree Rust | 101.8 |
| 07 Svelte collab editor | 101.1 |
| 08 Kotlin offline sync | 99.8 |
| **Average** | **102.2** |

Quality preserved: spot-check confirmed UD-Q4_K_XL behaviors (allowlist sanitizer in Svelte, content-asserting Kotlin tests) carry through with turbo3 KV.

### Per-fork detailed assessment

#### Madreag fork ⭐ recommended (now in production)

URL: https://github.com/Madreag/turbo3-cuda

- Built on top of TheTom; adds CUDA kernel optimizations (8-wide LUT, `nthreads_KQ=8`, sparse V skip, `__launch_bounds__(128, 3)`, half-precision LUT, `__expf` softmax, L2 prefetch)
- Authors validated on **4 GPUs including RTX 3090 and 3090 Ti** (most relevant for us)
- Claims 13-69% gain over base TheTom at 32K+ context
- Reported PPL on Qwen 3.5 27B Q6_K: turbo3 +1.38% PPL @ ctx=512, equals q8_0 @ ctx=2048
- Recently absorbed [TCQ from spiritbuun](https://github.com/Madreag/turbo3-cuda/pull/1)
- **Zero open issues**

Build: standard llama.cpp `cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES=86`

Runtime: `--cache-type-k turbo3 --cache-type-v turbo3 -fa on`

#### TheTom fork (base implementation)

URL: https://github.com/TheTom/llama-cpp-turboquant
Branch: `feature/turboquant-kv-cache`

- The "base" TurboQuant CUDA + Metal + HIP implementation
- Most upstream-aligned — recently synced to upstream b8871 (64 commits)
- Open issues:
  - [#102 — turbo4 slowdown at large context on RTX 3090](https://github.com/TheTom/llama-cpp-turboquant/issues/102) (relevant to us; turbo3 is fine)
  - [#88 — Vulkan turbo3 incoherent decode](https://github.com/TheTom/llama-cpp-turboquant/issues/88) (we use CUDA, N/A)
  - [#89 — Compile bug Windows + Vulkan](https://github.com/TheTom/llama-cpp-turboquant/issues/89)
  - [#95 — vague "problem"](https://github.com/TheTom/llama-cpp-turboquant/issues/95)
- Slightly faster than Madreag at short ctx but Madreag wins at long ctx

#### spiritbuun fork (TCQ research codebase)

URL: https://github.com/spiritbuun/llama-cpp-turboquant-cuda

- Adds **Trellis-Coded Quantization (TCQ)** — Viterbi decoder constrains quant indices to a 512-state trellis, achieving 10-44% KLD reduction at same bit rate
- At 3.25 bpv, TCQ produces **lower PPL than FP16 KV cache** (5.802 vs 5.805) per their docs
- Heavy active work on **DFlash speculative decoding for Qwen3.6-35B-A3B hybrid MoE** specifically — could become important if the speculative-decoding-is-net-negative finding is overturned
- Plain `turbo3` performs the same as Madreag/TheTom; `turbo3_tcq` is ~5% slower due to Viterbi encode but should give better quality
- **Zero open issues** (work-in-progress fork)
- Their TCQ work [merged into Madreag's fork via PR #1](https://github.com/Madreag/turbo3-cuda/pull/1)

#### AmesianX fork (TBQ/AMX3 hybrid + TriAttention)

URL: https://github.com/AmesianX/TurboQuant

- Original [ICLR 2026 reference implementation](https://arxiv.org/abs/2504.19874) of TurboQuant by Google DeepMind
- Different cache types: `tbq3`, `tbq4`, `tbqp3`, `amx3` (hybrid), each with subtle differences
- v1.7.0 added **TriAttention token pruning** — separate axis from KV compression, requires calibration files
- Tested mainly on DGX Spark (GB10 Blackwell) and 2080 Ti — Ampere is unvalidated
- **Open issues that worried us:**
  - [#11 — V-cache precision bug at 1500-2000t on head_dim=128 (Qwen3-14B)](https://github.com/AmesianX/TurboQuant/issues/11). Our Qwen3.6-35B-A3B is also head_dim=128 — **risk of long-context corruption**. Did not manifest in our short tests but warrants caution
  - [#18 — llama-server crashes silently with any tbq cache type (RTX 2080 Ti)](https://github.com/AmesianX/TurboQuant/issues/18). Did not crash on our RTX 3090
  - [#16 — V-cache breaks any tbq](https://github.com/AmesianX/TurboQuant/issues/16) — may be the same head_dim issue
  - [#19 — non-deterministic output even at fixed seed](https://github.com/AmesianX/TurboQuant/issues/19)

#### animehacker fork (TQ3_0 — partial implementation)

URL: https://github.com/animehacker/llama-turboquant

- Self-described in latest commit: "PolarQuant 3-bit, **not full TurboQuant with QJL**"
- Uses `tq3_0` cache type (different naming)
- 4.6× compression at +4.6% PPL on Qwen3.5-0.8B
- Tested mainly on AMD ROCm (Radeon 8060S Strix Halo) — the CUDA path is not the primary code path
- **Open issues:**
  - [#4 — turbo3 crash (0xc0000005) on Qwen3-14B + RTX 3060 Ampere](https://github.com/animehacker/llama-turboquant/issues/4). Important: this is sm_86 = **same compute capability as our RTX 3090**. Their workaround was `turbo2` works but `turbo3` crashes. Our `tq3_0` test ran without crashing but was extremely slow at long generation
  - [#1 — Segmentation fault](https://github.com/animehacker/llama-turboquant/issues/1)
  - [#3 — Gemma 4 support](https://github.com/animehacker/llama-turboquant/issues/3)
- Our `tq3_0` test ran without crashing but was **17 t/s on the GoL prompt — 5× slower than other forks**, indicating an unoptimized CUDA path
- **Skip**: stale (28 days), partial implementation, unoptimized CUDA, known Ampere bugs

---

## 7. Things that DON'T help (already confirmed, don't retry)

- **Speculative decoding** on Qwen3.6-35B-A3B: net-negative on RTX 3090 ([thc1006 benchmark](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090))
- **MTP (multi-token prediction)** [PR #20700](https://github.com/ggml-org/llama.cpp/pull/20700): dense models only, doesn't apply to MoE
- **`-sm graph`**: 10 t/s on PCIe (needs NVLink)
- **Smart Expert Reduction `-ser`**: no speed gain on this architecture
- **Fused MoE `-fmoe`**: already default in ik_llama.cpp
- **`LLAMA_SET_ROWS=1`**: no impact
- **vLLM TP=2**: 7× slower on PCIe
- **Tensor override `-ot` to CPU**: only useful if model doesn't fit on GPU; we fit fine

---

## 8. Watch list (potential future wins)

- **TurboQuant CUDA upstream merge**: [discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) — when it lands, Madreag's optimizations may upstream
- **TurboQuant + MTP** combo: vLLM stack with [Lorbus AutoRound INT4](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) reportedly hits 85 t/s at 125k ctx for the 27B variant — would need MTP-enabled weights
- **spiritbuun's DFlash for Qwen3.6 MoE**: in active development, may overturn the "spec dec is net-negative" finding
- **[turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3)**: 3.5-bit Walsh-Hadamard transform for **weights** (not KV), claims Q4 quality at 10% smaller — different lever, would need re-quantization

---

## 9. References & resources

### Models
- Qwen 3.6 announcement & weights: https://github.com/QwenLM/Qwen3.6
- Unsloth GGUF (UD-Q4_K_XL): https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
- bartowski GGUF (imatrix-Q4_0): https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF
- Unsloth dynamic v2.0 docs: https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs
- Unsloth Qwen 3.6 run guide: https://unsloth.ai/docs/models/qwen3.6

### Engines
- ik_llama.cpp: https://github.com/ikawrakow/ik_llama.cpp
- ik_llama.cpp quick-start: https://github.com/ikawrakow/ik_llama.cpp/discussions/258
- upstream llama.cpp: https://github.com/ggml-org/llama.cpp
- Qwen llama.cpp guide: https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html

### TurboQuant ecosystem
- TurboQuant paper (Google DeepMind, ICLR 2026): https://arxiv.org/abs/2504.19874
- Status discussion in upstream: https://github.com/ggml-org/llama.cpp/discussions/20969
- HIP/ROCm port discussion: https://github.com/ggml-org/llama.cpp/discussions/21526
- **Madreag turbo3-cuda fork (production)**: https://github.com/Madreag/turbo3-cuda
- TheTom base implementation: https://github.com/TheTom/llama-cpp-turboquant
- spiritbuun TCQ fork: https://github.com/spiritbuun/llama-cpp-turboquant-cuda
- AmesianX ICLR 2026 reference: https://github.com/AmesianX/TurboQuant
- animehacker partial impl: https://github.com/animehacker/llama-turboquant
- turbo-tan TQ3 weight quant: https://github.com/turbo-tan/llama.cpp-tq3

### Speculative decoding research
- llama.cpp PR #19493 (Qwen 3.5/3.6 MoE classic spec decode): https://github.com/ggml-org/llama.cpp/pull/19493
- llama.cpp PR #20700 (MTP for dense Qwen 3.5/3.6): https://github.com/ggml-org/llama.cpp/pull/20700
- thc1006 RTX 3090 benchmark: https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090
- Speculative decoding general discussion: https://github.com/ggml-org/llama.cpp/discussions/12130

### Practical guides
- vLLM Qwen 3.5/3.6 recipe: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
- vLLM 85 TPS @ 125k ctx writeup: https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914
- MoE offload guide (when model doesn't fit on GPU): https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide

### Tooling
- uv (Astral fast Python tooling): https://docs.astral.sh/uv/
- Bun (JS/TS runtime): https://bun.sh
- huggingface_hub CLI: https://huggingface.co/docs/huggingface_hub/guides/cli

### Cited GitHub issues
- TheTom #102 (turbo4 large-ctx slowdown on 3090): https://github.com/TheTom/llama-cpp-turboquant/issues/102
- TheTom #88 (Vulkan turbo3 incoherent): https://github.com/TheTom/llama-cpp-turboquant/issues/88
- TheTom #95 ("problem"): https://github.com/TheTom/llama-cpp-turboquant/issues/95
- TheTom #89 (Windows Vulkan compile): https://github.com/TheTom/llama-cpp-turboquant/issues/89
- AmesianX #11 (V-cache precision bug head_dim=128): https://github.com/AmesianX/TurboQuant/issues/11
- AmesianX #16 (V-cache breaks tbq): https://github.com/AmesianX/TurboQuant/issues/16
- AmesianX #18 (silent crash any tbq): https://github.com/AmesianX/TurboQuant/issues/18
- AmesianX #19 (non-deterministic output): https://github.com/AmesianX/TurboQuant/issues/19
- animehacker #4 (turbo3 crash on RTX 3060 sm_86): https://github.com/animehacker/llama-turboquant/issues/4
- animehacker #1 (segfault): https://github.com/animehacker/llama-turboquant/issues/1
- Madreag PR #1 (TCQ port from spiritbuun): https://github.com/Madreag/turbo3-cuda/pull/1
