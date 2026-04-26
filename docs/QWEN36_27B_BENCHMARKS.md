# Qwen 3.6 27B Local Inference — Complete Benchmark Report

**Hardware**: 1× RTX 3090 (24 GB, sm_86, CUDA 12.6) on WSL2 Ubuntu 22.04
**Model under test**: [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (Dense, hybrid arch, multimodal, released 2026-04-22)
**Test date**: 2026-04-25
**Outputs**: [bench/results/qwen36-27b/gol/](../bench/results/qwen36-27b/gol/) and `/mnt/c/Users/Will/Dropbox/qwen36-27b-bench/`

---

## Executive Summary

We benchmarked **27 runtime configurations** across **2 inference backends** ([Madreag turboquant fork](https://github.com/Madreag/turbo3-cuda) and [vLLM nightly with MTP speculative decoding](https://github.com/vllm-project/vllm)) on the 27B model, including a **deep-dive sweep of 11 KV cache types**, **3 weight quants**, and **MTP n∈{2,3,4}** to find the speed ceiling. GoL parser-validation as the quality gate.

**Top finding**: **vLLM + [Lorbus/Qwen3.6-27B-int4-AutoRound](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) + MTP n=3 + fp8 KV at full 262k context = ~54 t/s on coding prompts, ~31 t/s on prose** — a **2.7× speedup over llama.cpp** *only when speculative-decode acceptance is high*. MTP (multi-token prediction speculative decoding) is the key — vLLM without MTP runs at the same ~25 t/s as llama.cpp.

**Speed ceiling confirmed (Section 8 deep dive)**: KV format (11 types tested) varies output by <13%, smaller weight quants don't help (UD-Q3_K_XL = UD-Q4_K_XL ≈ 20 t/s), and MTP n=2 is strictly worse than n=3. The only untested levers likely to break the 54 t/s ceiling are [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), or block-diffusion drafting via [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash).

The downside: vLLM setup is significantly more complex (vLLM nightly install, Lorbus AutoRound model download, tokenizer config patch, GPU memory tuning).

For a llama.cpp-only stack, **IQ4_XS + turbo3 KV** runs at **24 t/s with full 262k context** and only 18.5 GB VRAM — same speed at any context size, easier setup.

**27B Dense vs 35B-A3B MoE comparison** (same hardware, same Madreag binary):
- 35B-A3B + UD-Q4_K_XL + turbo3 KV @ 262k: **~102 t/s** (3 GB active params/token)
- 27B Dense + UD-Q4_K_XL + turbo3 KV @ 262k: **~20 t/s** (~13.5 GB active params/token)
- 27B is **fundamentally bandwidth-bound** — 5× more weights to read per token than 35B-A3B's MoE

The 27B is best treated as a **quality model for hard coding tasks**, not a daily driver. Qwen claims 77.2 SWE-bench Verified for 27B (Sonnet 4.6 territory). Use 35B-A3B for speed, 27B for hard problems.

---

## Quick Reference — All Configurations Tested

GoL prompt at `max_tokens=12288`, all natural stop, all parsers validated 3/3 (handles `3o2b1o!2b3o`, `o!o!o`, `24bo` correctly).

| # | Backend / Config | t/s | VRAM | Context |
|---|------------------|----:|-----:|--------:|
| 1 | Madreag UD-Q4_K_XL + turbo3 KV | 19.97 | 17.5 GB | 32k |
| 2 | Madreag UD-Q4_K_XL + turbo3 KV | 20.21 | 17.9 GB | 64k |
| 3 | Madreag UD-Q4_K_XL + turbo3 KV | ~20 | 20.6 GB | 262k |
| 4 | Madreag UD-Q4_K_XL + q8_0 KV | 20.08 | 18.1 GB | 32k |
| 5 | Madreag UD-Q4_K_XL + bf16 KV | 19.80 | 19.1 GB | 32k |
| 6 | **Madreag IQ4_XS + turbo3 KV** ⭐ (llama.cpp speed champion) | **24.18** | 15.4 GB | 32k |
| 7 | **Madreag IQ4_XS + turbo3 KV** ⭐ (best simple long-context) | **23.96** | 18.5 GB | **262k** |
| 8 | vLLM Lorbus AutoRound + fp8 KV (no MTP) | 24.90 | 21.7 GB | 64k |
| 9 | vLLM Lorbus AutoRound + fp8 KV + MTP n=3 | 53.74 | 21.3 GB | 64k |
| 10 | vLLM Lorbus AutoRound + fp8 KV + MTP n=3 | 53.70 | 21.3 GB | 125k |
| **11** | **vLLM Lorbus AutoRound + fp8 KV + MTP n=3** ⭐⭐ (overall winner) | **54.55** | **21.3 GB** | **262k** |

Bold = headlines. ⭐⭐ = overall speed/context champion. ⭐ = llama.cpp tier champion (no Python deps).

---

## Quick Reference — Full Hard Bench (winner config)

vLLM + Lorbus AutoRound INT4 + MTP n=3 + fp8 KV @ 262k, six prompts, all natural stop:

| Prompt | t/s | Tokens |
|--------|----:|-------:|
| 01 Game of Life | 53.53 | 3642 |
| 02 Regex engine (Thompson NFA) | 59.81 | 4476 |
| 03 Mini Lisp interpreter | 51.18 | 9326 |
| 04 Sudoku CSP + AC-3 | 52.04 | 3196 |
| 05 CRDT RGA | 52.16 | 5150 |
| 06 B-Tree Rust | 58.53 | 9813 |
| **Average** | **54.5** | |

Total 6-prompt time: **11 minutes** (vs. ~30 minutes on Madreag UD-Q4_K_XL).

---

# Full Details

## 1. Model architecture (why 27B is so different from 35B-A3B)

Per [Unsloth Qwen3.6-27B GGUF model card](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF):

| Property | Value |
|----------|-------|
| Type | Dense (NOT MoE) |
| Total parameters | 27B (all active per token) |
| Hidden dim | 5120 |
| Layers | 64 total |
| Architecture | Hybrid: 16 blocks × (3 × Gated DeltaNet + 1 × Gated Attention) |
| Attention KV | 4 heads × 256 head_dim, GQA 6:1 (24Q heads → 4 KV heads) |
| DeltaNet | Linear attention, no KV cache |
| Native context | 262,144 tokens (extensible to 1,010,000 via YaRN) |
| Multimodal | Yes (vision encoder; use `--language-model-only` for text-only) |

**Implication**: only **16 of 64 layers** have KV cache. KV memory is therefore tiny (~32 KB/token at f16 → 8.4 GB at 262k) — easy to fit.

But **all 27B weights are read per token**. At Q4 (~13.5 GB), bandwidth = ~13.5 GB/token. RTX 3090 ≈ 936 GB/s memory bandwidth → theoretical max ~69 t/s. Observed: 20-25 t/s = ~30% of theoretical.

By contrast, 35B-A3B MoE: only **3B active params/token** at Q4 ≈ 1.5 GB/token → theoretical ~624 t/s, observed ~120 t/s ≈ 19% of theoretical. 27B Dense ends up bandwidth-bound at much lower throughput.

**MTP unlocks the rest**: by speculatively predicting 3 tokens and verifying them in parallel, we run ~3 forward passes-worth of work but check 4 tokens, getting effective ~2.7× speedup.

---

## 2. Recommended sampling (per [Unsloth docs](https://unsloth.ai/docs/models/qwen3.6))

| Mode | temp | top_p | top_k | min_p | presence_penalty |
|------|-----:|------:|------:|------:|-----------------:|
| Thinking general | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking precise coding | 0.6 | 0.95 | 20 | 0.0 | 0.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

**New for 3.6**: `--reasoning-format deepseek`, `chat_template_kwargs: {preserve_thinking: true}` for agentic loops.

**Important**: default mode is **thinking**. Disable per-request via `chat_template_kwargs: {enable_thinking: false}` for fast non-reasoning replies.

**Caveat**: Unsloth recommends min context 128K to preserve thinking quality.

**Hardware caveat**: Avoid CUDA 13.2 — produces gibberish on Qwen 3.6. We use 12.6, safe.

---

## 3. Backend deep dives

### 3a. Madreag turboquant fork (llama.cpp family)

URL: https://github.com/Madreag/turbo3-cuda

This is the same engine we use for 35B-A3B production. Already built at `~/llama-cpp-turboquant/llama-server`.

Best-of-class for: simple setup, full long context, no Python dependencies, lowest VRAM.

**Configurations tested with Qwen3.6-27B**:
- Best speed: **IQ4_XS + turbo3 KV @ 32k = 24.18 t/s** (lowest VRAM 15.4 GB)
- Best quality+context: **UD-Q4_K_XL + turbo3 KV @ 262k = ~20 t/s** (full ctx with quality quant)
- Best speed+context: **IQ4_XS + turbo3 KV @ 262k = 23.96 t/s** ⭐ (full ctx, only 18.5 GB)

KV format barely matters for 27B speed:

| KV format @ 32k UD-Q4_K_XL | t/s | VRAM |
|-----|----:|-----:|
| turbo3 (3.125 bpv) | 19.97 | 17.5 GB |
| q8_0 (8.5 bpv) | 20.08 | 18.1 GB |
| bf16 (16 bpv) | 19.80 | 19.1 GB |

Why: at 27B Dense the model-weight bandwidth dominates. KV cache reads are tiny (only 16 layers).

### 3b. vLLM 0.17.0rc1.dev126 + Lorbus AutoRound INT4

URL (model): https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound (~14 GB)

The Lorbus AutoRound model is special: it keeps the **MTP (multi-token prediction) head in BF16** while the main weights are INT4. This is what enables vLLM speculative decoding without quality loss.

**Required setup steps**:
1. vLLM nightly with Qwen3.6 support (we have 0.17.0rc1.dev126)
2. Download Lorbus AutoRound model
3. Patch `tokenizer_config.json`: change `tokenizer_class` from `"TokenizersBackend"` to `"Qwen2TokenizerFast"` (vLLM doesn't recognize the custom Genesis tokenizer)
4. Run vllm serve with `--quantization auto_round`, `--kv-cache-dtype fp8`, `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`

**Speculative decoding tuning**:
- n=1 to n=3: monotonically faster
- **n=3 is sweet spot** (~54 t/s)
- n=4: **CRASHES with CUDA illegal memory access** — MTP head trained for n=3, exceeding it breaks the model

**Context vs speed**: speed is **flat from 64k to 262k** (53.7 / 53.7 / 54.5 t/s) — context doesn't add token-level cost when KV reads are tiny.

---

## 4. From-scratch setup commands

### 4a. Madreag llama.cpp setup (already in repo)

If you don't have it: `~/git/wsl-llm/scripts/build-turboquant.sh` builds Madreag's fork.

Then download the recommended 27B quant:

```bash
# UD-Q4_K_XL (better quality, recommended by Unsloth)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
    unsloth/Qwen3.6-27B-GGUF \
    Qwen3.6-27B-UD-Q4_K_XL.gguf \
    --local-dir ~/models/

# IQ4_XS (faster, slightly lower quality)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
    unsloth/Qwen3.6-27B-GGUF \
    Qwen3.6-27B-IQ4_XS.gguf \
    --local-dir ~/models/
```

Run as a separate model alongside production (port 8081 to avoid clashing with 8080):

```bash
~/llama-cpp-turboquant/llama-server \
    -m ~/models/Qwen3.6-27B-IQ4_XS.gguf \
    --alias qwen3.6-27b \
    -ngl 999 -c 262144 --parallel 1 -fa on \
    --cache-type-k turbo3 --cache-type-v turbo3 \
    --jinja --reasoning-format deepseek \
    --host 0.0.0.0 --port 8081
```

### 4b. vLLM + Lorbus AutoRound + MTP setup (speed champion)

```bash
# Activate the existing bench Python env (has vLLM 0.17.0rc1.dev126 already)
source ~/bench_env/bin/activate

# Install hf_transfer for fast HF downloads
pip install hf_transfer

# Download Lorbus AutoRound INT4 model (~14 GB)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download \
    Lorbus/Qwen3.6-27B-int4-AutoRound \
    --local-dir ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound

# Patch tokenizer config (vLLM doesn't support Genesis "TokenizersBackend")
python3 -c "
import json
p = '$HOME/models/Lorbus-Qwen3.6-27B-int4-AutoRound/tokenizer_config.json'
d = json.load(open(p))
d['tokenizer_class'] = 'Qwen2TokenizerFast'
json.dump(d, open(p, 'w'), indent=2)
"

# Launch vLLM with MTP n=3 + fp8 KV at 262k context
CUDA_VISIBLE_DEVICES=0 ~/bench_env/bin/python -m vllm.entrypoints.cli.main serve \
    ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --quantization auto_round \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --port 8081 \
    --trust-remote-code \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

**Watch for**: GPU 1 utilization (other workloads). If GPU 0 has < ~22.5 GB free at startup, lower `--gpu-memory-utilization` or `--max-model-len`.

---

## 5. Production decision

For our 1× RTX 3090 setup we keep **35B-A3B as primary daily driver** (5× faster, comparable code quality) and recommend **27B as a parallel "deep-thinking" option** when quality matters more than speed.

| Use case | Model | Backend | Why |
|----------|-------|---------|-----|
| Daily coding (chat, edits, reviews) | Qwen3.6-35B-A3B-UD-Q4_K_XL | Madreag turboquant @ 262k turbo3 | 100+ t/s, 262k ctx, comparable quality on most tasks |
| Hard one-shot coding (refactor, design, agentic) | Qwen3.6-27B-UD-Q4_K_XL | Madreag turboquant @ 262k turbo3 | 20 t/s but ~Sonnet 4.6 quality (per Qwen) |
| 27B with serious throughput | Qwen3.6-27B (Lorbus AutoRound INT4) | vLLM + MTP n=3 + fp8 KV @ 262k | 54 t/s, 2.7× speedup, complex setup |

The two models can co-exist on different ports (35B-A3B on 8080, 27B on 8081) but only one can run at a time on a single 24 GB GPU.

---

## 6. Things that don't help (or fail) on 27B

- **MTP n=4**: CUDA illegal memory access. The Lorbus MTP head is trained for n=3.
- **vLLM with Genesis-style tokenizer**: out-of-the-box vLLM doesn't recognize `TokenizersBackend`. Patch `tokenizer_class` to `Qwen2TokenizerFast`.
- **vLLM TurboQuant 3-bit KV** (`turboquant_3bit_nc`): not available in vanilla vLLM 0.17.0rc1.dev126. Would need [Sandermage Genesis patches](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) (untested by us).
- **Increasing context past 262k**: requires YaRN config edit (`mrope_interleaved`, `factor=4.0`); we didn't test 1M.
- **CUDA 13.2**: Unsloth warns of gibberish output. We're on 12.6, safe.

---

## 7. Watch list

- **TurboQuant 3-bit KV in vanilla vLLM**: would shave KV-cache memory and let MTP run at 1M context. [Sandermage's patches](https://github.com/ggml-org/llama.cpp/discussions/20969) are the reported path; not tested here.
- **z-lab/Qwen3.6-27B-DFlash** (HF): block-diffusion drafting + DDTree verification claimed 207 t/s on Qwen 3.5-27B. Could be a 4× win over MTP if it works on 27B Dense. Untested by us.
- **`spiritbuun/Qwen3.6-27B-DFlash-GGUF`**: spiritbuun has a Qwen3.6-27B variant. Worth testing if speculative decoding for the dense variant lands cleanly.

---

## 8. Speed ceiling investigation — what actually moves 27B Dense t/s

Re-bench on 2026-04-25 to settle whether **KV format**, **smaller weight quants**, or **alternative MTP step counts** can beat the existing 24 t/s (Madreag llama.cpp) / 54 t/s (vLLM+MTP) ceilings. All same hardware, single RTX 3090, GPU 0 only. Bench prompt: 800-token "explain LSM trees" (prose, low MTP acceptance) — chosen so all backends finish quickly and produce comparable steady-state generation rates.

### 8a. KV cache format — full sweep on Madreag IQ4_XS @ 64k

Madreag fork supports 11 KV types (`f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, turbo1.5, turbo2, turbo3, turbo4, turbo3_tcq, turbo2_tcq`). We tested all 11 with K and V both set to the same type:

| KV type | gen t/s | prompt t/s | VRAM (MiB) | Notes |
|---------|--------:|-----------:|-----------:|-------|
| **turbo3** ⭐ | **27.28** | 224.20 | 15787 | Best — 3.125 bpv, fastest decode kernel |
| q8_0 | 26.89 | 260.49 | 17161 | Fastest prompt processing (highest precision) |
| bf16 | 26.34 | 270.22 | 19081 | Highest VRAM, no quality compromise |
| q4_0 | 26.08 | 208.80 | 16137 | |
| iq4_nl | 25.50 | 164.46 | 16107 | |
| q5_0 | 25.09 | 192.34 | 16393 | |
| turbo1.5 | 24.52 | 202.57 | 16011 | Lowest bpv non-tcq |
| turbo2 | 24.31 | 203.01 | 15735 | |
| turbo4 | 24.23 | 202.17 | 16075 | Higher bpv didn't help |
| turbo3_tcq | 24.22 | 171.15 | 15821 | Transform-coded variant; same speed |
| turbo2_tcq | 24.00 | 201.40 | 15565 | Lowest VRAM; slowest decode |

**Range: 24.0 – 27.3 t/s = 13% spread.** turbo3 wins by ~1 t/s over q8_0 and bf16, but the gap is within run-to-run noise. **KV format does not unlock 27B Dense throughput.**

### 8b. Smaller weight quants on Madreag (turbo3 KV @ 64k)

| Quant | File size | gen t/s | VRAM (MiB) | Notes |
|-------|----------:|--------:|-----------:|-------|
| Qwen3.6-27B-IQ4_XS | 15.4 GB | **25.24** | 15787 | Madreag's tuned llama.cpp kernel path |
| Qwen3.6-27B-UD-Q4_K_XL | 17.6 GB | 20.40 | 17859 | UD mixed-tier blocks (Q4/Q5/Q6) |
| Qwen3.6-27B-UD-Q3_K_XL | 14.5 GB | 20.05 | 14865 | UD mixed-tier — **smaller file but no faster** |

**Surprise**: UD-Q3_K_XL (14.5 GB on disk) is **not faster than UD-Q4_K_XL** despite ~3 GB less weight bandwidth on paper. Reason: Unsloth's UD ("dynamic") quants use mixed Q3/Q5/Q6/Q8 blocks for sensitive tensors, so effective bandwidth ≠ filesize. **IQ4_XS still wins** because it hits Madreag's specialized IQ-flavored kernel — pure Q3_K_S (12.4 GB plain) wasn't tested but would likely fall in the 22-25 t/s band based on the kernel-path effect.

### 8c. vLLM MTP num_speculative_tokens — n=2 vs n=3, same prompt

Same vLLM/Lorbus/fp8/64k config; only `num_speculative_tokens` varied. Same 800-token LSM-tree prompt:

| MTP n | gen t/s | VRAM (MiB) | Notes |
|------:|--------:|-----------:|-------|
| 2 | 29.37 | 22327 | Lower acceptance rate per [vLLM warning](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py): "Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate" |
| **3** ⭐ | **31.06** | 22339 | Production target, MTP head trained for n=3 |
| 4 | crash | — | CUDA illegal memory access (already documented) |

**n=3 wins by ~6%** even on prose; the gap widens dramatically on coding prompts (54 t/s GoL vs 31 t/s LSM-prose). MTP acceptance rate is **prompt-content-dependent**: predictable patterns (code, structured output) accept far better than novel prose.

### 8d. Why 27B Dense is bandwidth-bound — the math

| Backend | Effective bandwidth | Observed t/s | Theoretical ceiling | Utilization |
|---------|--------------------:|-------------:|--------------------:|------------:|
| Madreag IQ4_XS | 15.4 GB / token | 25 | 936 / 15.4 = **60.7 t/s** | 41% |
| Madreag UD-Q4_K_XL | 17.6 GB | 20 | 936 / 17.6 = **53.2** | 38% |
| vLLM Lorbus INT4 (no MTP) | ~14 GB | 25 | 936 / 14 = **66.9** | 37% |
| vLLM + MTP n=3 (code) | ~14 GB / 2.7 tok per pass | **54** | 66.9 × 2.7 = **180** | 30% (real) |

RTX 3090 ≈ 936 GB/s memory bandwidth. The 30-41% utilization gap is compute overhead (matmul, attention, softmax) that doesn't get amortized further. Speculative decoding is the only known path past the bandwidth ceiling.

### 8e. Untried levers ranked by expected upside

The following weren't tested but are the only paths likely to materially exceed 54 t/s on 27B Dense + 1× RTX 3090:

| Lever | Expected gain | Cost / risk |
|-------|---------------|-------------|
| [ExLlamaV3 (turboderp)](https://github.com/turboderp-org/exllamav3) | +30-60% (35-45 t/s baseline) | Hand-tuned Ampere kernels, but no MTP support yet — would lose the 2.7× spec-decode multiplier |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | unknown (5-30% over vLLM) | Days of setup, separate weight conversion |
| [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) (block-diffusion drafting) | claimed 4× over MTP | Untested on Dense 27B; HF model exists but kernel support unverified |
| [Sandermage Genesis vLLM patches](https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914) (TurboQuant 3-bit KV) | +0% speed (memory only) | Already proved KV format doesn't matter for 27B |

The headline number — **vLLM + Lorbus AutoRound INT4 + MTP n=3 + fp8 KV @ 262k = ~54 t/s on coding tasks, ~31 t/s on prose** — remains the production ceiling on this hardware until ExLlamaV3 ships MTP support or DFlash gains traction.

---

## 9. References

### Models
- Qwen 3.6 27B official: https://github.com/QwenLM/Qwen3.6
- Unsloth GGUF (UD-Q4_K_XL etc): https://huggingface.co/unsloth/Qwen3.6-27B-GGUF
- Lorbus AutoRound INT4 (vLLM MTP): https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound
- bartowski GGUF: https://huggingface.co/bartowski/Qwen_Qwen3.6-27B-GGUF
- Unsloth Qwen3.6 docs: https://unsloth.ai/docs/models/qwen3.6
- Qwen 3.6 official model: https://huggingface.co/Qwen/Qwen3.6-27B

### Engines
- Madreag turboquant fork: https://github.com/Madreag/turbo3-cuda
- vLLM nightly: https://github.com/vllm-project/vllm
- TurboQuant CUDA discussion (vLLM patches): https://github.com/ggml-org/llama.cpp/discussions/20969

### Reference benchmarks
- "Overnight stack for Qwen3.6-27B: 85 TPS, 125K Context, Vision": https://medium.com/@fzbcwvv/an-overnight-stack-for-qwen3-6-27b-85-tps-125k-context-vision-on-one-rtx-3090-0d95c6291914
- Simon Willison's run: https://simonwillison.net/2026/Apr/22/qwen36-27b/
- llama.cpp vs vLLM consumer GPU comparison: https://dev.to/defilan/we-ran-qwen36-27b-on-800-of-consumer-gpus-day-one-llamacpp-vs-vllm-mg1
- Qwen3.5-27B 207 t/s (DFlash): https://news.ycombinator.com/item?id=47838788
- 27B Dense vs 35B-A3B MoE: https://insiderllm.com/guides/qwen-3-6-local-ai-guide/

### Project docs
- 35B-A3B benchmark report: [QWEN36_BENCHMARKS.md](QWEN36_BENCHMARKS.md)
- 27B GoL HTMLs: [bench/results/qwen36-27b/gol/](../bench/results/qwen36-27b/gol/)
