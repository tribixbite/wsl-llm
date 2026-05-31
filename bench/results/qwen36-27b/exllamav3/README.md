# ExLlamaV3 — first benchmark on Qwen3.6-27B (2026-05-31)

Finally got ExLlamaV3 working after multiple failed attempts on May 1-3. The blocker was the **torch + flash-attn + xformers + CUDA dependency matrix** — every version permutation conflicted.

## Working install recipe (exl3 0.0.30, single RTX 3090)

```bash
# Fresh venv
uv venv ~/exl3_env_v2 --python 3.10

# Pin torch 2.6.0+cu126 (NOT cu13 — wheels won't exist for flash-attn)
uv pip install --python ~/exl3_env_v2/bin/python \
    --index-url https://download.pytorch.org/whl/cu126 "torch==2.6.0"

# Install flash-attn from prebuilt wheel matching torch 2.6 + cu126
uv pip install --python ~/exl3_env_v2/bin/python \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
    ninja setuptools wheel

# Install exllamav3 WITHOUT deps (xformers requires torch>=2.12 which breaks us)
uv pip install --python ~/exl3_env_v2/bin/python --no-deps \
    --no-build-isolation "exllamav3==0.0.30"

# Install only the runtime deps we actually need
uv pip install --python ~/exl3_env_v2/bin/python \
    "tokenizers" "numpy<2.2" "rich" "pyyaml" "safetensors" \
    "marisa-trie" "kbnf" "formatron" "pydantic" "pillow"

# IMPORTANT: do NOT install xformers — it pins torch>=2.12 which breaks the flash-attn ABI
```

At first import, ExL3 JIT-compiles its CUDA extension. Needs `ninja` and `nvcc` (CUDA 12.6) on PATH:

```bash
PATH=~/exl3_env_v2/bin:/usr/local/cuda-12.6/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.6 \
python -c "import exllamav3"   # ~30s first run for JIT compile
```

## Baseline result (no speculative decoding)

| Workload | Tokens | Wall | Decode TPS |
|----------|-------:|-----:|----------:|
| Prose (LSM tree explanation) | 798 | 30.22s | **26.40** |
| Code (TS binary search tree) | 798 | 29.86s | **26.73** |

**~26.5 t/s** — same neighborhood as Madreag llama.cpp, ik_llama.cpp, and vLLM no-spec baselines on this hardware. Confirms ExL3 doesn't break the **27B Dense bandwidth ceiling** of ~67 t/s on its own — the 13.5 GB/token weight read is the wall.

## Why we ran this

To establish that ExL3 install works and matches expected baseline, before attempting speculative decoding via:

- **`turboderp/Qwen3.6-27B-DFlash-exl3`** (May 6) — DFlash drafter quants
  - Branches: 2.50, 3.00, 3.50, 4.00, 5.00, 6.00 bpw
  - Mean accepted tokens at 4.00 bpw: 4.46 (vs MTP ~2.4)
  - Claim per community bench: **140-177 t/s on agentic code**

**Caveat (not yet validated)**: DFlash is a same-arch-as-target drafter (~27B params), not a small head. At 4.00 bpw ≈ 14 GB for the drafter alone. Combined with our 16 GB target = **30 GB**, which exceeds the 24 GB single-3090 budget. The 140-177 t/s claim likely requires multi-GPU or tighter quants. Needs investigation.

## Files

| File | What |
|------|------|
| `bench.py` | The actual generate() bench used to produce the above numbers |
| `requirements.txt` | Pinned dep set that imports cleanly |
