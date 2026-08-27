# PM — Legion RTX 5080 (16 GB) Qwen3.8-27B bring-up & engine benchmark

**Machine (NEW — differs from repo CLAUDE.md, which documents the 2×3090 desktop):**
- GPU: RTX 5080 Laptop, 16 GB GDDR7, Blackwell **sm_120**
- CPU: Intel Core Ultra 9 275HX (24c) · RAM 62 GB · Swap 16 GB
- Driver 610.57.01, CUDA UMD 13.3; toolkits `/usr/local/cuda-12.6`, `/usr/local/cuda-12.8`
- WSL2 Ubuntu 22.04 (kernel 5.15). Native Windows 11 available as an alternative.
- Disk: 91 GB free on `/`, 309 GB free on `/mnt/c`
- ⚠️ `nvidia-smi` reports power cap **95 W / 95 W** (RTX 5080 Laptop can be up to ~175 W) — investigate power mode.
- Tooling present: git, cmake, gcc, nvcc(12.8), python3.10, uv, hf, bun. **Missing: docker, aria2c.**

**Goal:** best accuracy + tok/s + agentic coding score for `unsloth/Qwen3.8-27B-GGUF` on this box.
Benchmark llama.cpp vs vLLM vs other engines found by research. Consider native Windows vs WSL2.

## Todos

### Phase 0 — recon (done)
- [x] Confirm actual hardware (it is NOT the 2×3090 machine in CLAUDE.md)
- [x] Inventory tooling & CUDA toolkits
- [x] Enumerate the unsloth GGUF repo file list + sizes
- [x] Start download of `Qwen3.8-27B-UD-Q3_K_XL.gguf` (12.24 GiB) + `mtp-Qwen3.8-27B-Q4_0.gguf` (1.28 GiB)
- [x] **Download complete** — both files present and whole (13,146,393,504 B + 1,369,590,656 B, no `.incomplete`)

### Phase 1 — research (subagents, in flight)
- [ ] A: Qwen3.8-27B model facts — arch, thinking mode, sampling params, published agentic/coding scores, quant sizing, engine support matrix
- [ ] B: Blackwell sm_120 engine landscape — llama.cpp / vLLM / SGLang / TensorRT-LLM / ExLlamaV3 / MLC / ik_llama; WSL2 vs native Windows; sysmem-fallback trap; laptop TGP & bandwidth ceiling; KV-cache strategy for 16 GB
- [x] C: Agentic-coding benchmark harnesses runnable locally; reuse of existing `bench/` runners; quantization-damage measurement (KL-divergence); fair engine-vs-engine methodology — **done, see "Benchmark-research findings" below + Phase 3 plan**
- [ ] Reconcile the three reports; resolve conflicts; pick the candidate stack list

### Phase 2 — environment
- [ ] Fix GPU power cap / Legion power mode if it is limiting us
- [ ] Verify CUDA sysmem-fallback is OFF (silent VRAM→RAM spill destroys throughput on WDDM)
- [ ] Build llama.cpp for sm_120 (`CMAKE_CUDA_ARCHITECTURES=120`, CUDA 12.8)
- [ ] Set up vLLM (venv, Blackwell-capable torch/wheel)
- [ ] Set up whatever additional engines research recommends
- [ ] Record a memory-bandwidth-derived theoretical decode ceiling as the benchmark target

### Phase 3 — benchmarks (plan finalised by research subagent C, 2026-08-27)

**Tier 0 — gates (~15 min, run before anything else; a failure here invalidates everything downstream)**
- [ ] `test-chat-auto-parser <gguf>` — confirm llama.cpp derives a tool-call parser from the embedded template
- [ ] `GET /props` → assert `chat_template_caps.supports_tools` / `supports_tool_calls`
- [ ] Greedy-diff sanity: 10 fixed prompts, temp 0, Q3_K_XL vs Q8_0 — catches a broken quant in minutes
- [ ] Record `enforced.power.limit` — it drifts (65 W ↔ 90 W observed); every run must log it

**Tier 1 — speed (~30 min/engine)**
- [ ] `llama-benchy` depth sweep `--depth 0 4096 16384 65536 --exact-tg --no-cache` (works on every OpenAI endpoint)
- [ ] Reuse `bench/stream_bench.py` for the house decode_TPS/TTFT number (comparable to the 3090 results)
- [ ] `llama-bench -d 0,4096,16384,65536` for the engine-internal prefill/decode split (llama.cpp only)
- [ ] MTP / speculative on vs off: `--spec-type draft-mtp -md MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (costs 1.37 GiB of KV budget)
- [ ] KV-cache quant sweep (f16 / q8_0 / q4_0) → max usable context

**Tier 2 — quality (~2-4 h)**
- [ ] `bench/aider_lite.py` — 34 python exercises, directly comparable to the existing 27B/35B/coder-30B numbers
- [ ] Official aider polyglot, host-native via `AIDER_DOCKER=1`, `--languages python,go,rust`
- [ ] Tool-calling: `scripts/tool_bench.py` + `DEBUG_EXTERNAL=1 tests.sh unit/test_tool_call.py`
- [ ] KL-divergence vs **Q8_0** baseline (BF16 = 54.7 GB, won't fit) — **`--chunks 200` is mandatory**:
      logits file = `n_chunk × (n_ctx/2−1) × (n_vocab+4) × 2`; at vocab 248320 that is 23.6 GiB
      @200 chunks but **66.5 GiB for the full corpus vs 72 GiB free disk**

**Tier 3 — overnight**
- [ ] mini-swe-agent on `swe-bench-verified-mini` (50 inst) via `MSWEA_DOCKER_EXECUTABLE=podman`; grade with `sb-cli`
- [ ] Quant ladder: UD-Q3_K_XL vs UD-IQ4_XS vs UD-Q4_K_S on the tier-2 suite
- [ ] LiveCodeBench v6 date-sliced (`--start_date 2025-01-01`) if a runner is stood up
- [ ] WSL2 vs native Windows head-to-head (if research says the delta is material)

**Fair-comparison invariants** (must be identical across engines or the numbers are meaningless):
sampling (temp/top_p/top_k/min_p), seed, prompt set, context length, thinking on/off,
KV dtype, flash-attn, batch/slots, warmup count, speculative on/off, and `--ignore-eos`
(or `--exact-tg`) so output length is fixed. Interleave runs ABAB — do not run engine A's
whole block then engine B's, because the power cap drifts.

### Phase 4 — deliverables
- [ ] `docs/QWEN38_27B_LEGION_BENCHMARKS.md` — full report in house style
- [ ] Update `CLAUDE.md` to cover BOTH machines (desktop 2×3090 and this Legion)
- [ ] Serving scripts/config for the winning stack
- [ ] Capture hard-won lessons (Blackwell build flags, WSL2 traps) per global CLAUDE.md

## ⚠️ THE BIG ONE — `--parallel` default silently destroys performance on this box

`llama-server` defaults to **`--parallel 4`**. On this hybrid arch every slot gets its own
DeltaNet recurrent state + compute buffers, so the same `-c 32768 -ctk q8_0` config costs:

| slots | peak VRAM | decode |
|---|---:|---:|
| 4 (default) | **15,941 MiB** of 16,303 | **0.04 t/s** ← evicted |
| 1 (`--parallel 1`) | 13,384 MiB | 27.0 t/s |

At 15.9 GiB we cross the VRAM ceiling and **WDDM silently evicts the model to system RAM
instead of raising OOM**. Measured VRAM trace: `13376 → 15941 → 2910 MiB` while
`/health` still returned `{"status":"ok"}`. Decode collapsed to 0.04 t/s (a ~700× cliff),
prompt eval to 0.28 t/s, and the server died ~100 s later.

**Rules for this machine:**
1. Always pass `--parallel 1` unless concurrency is actually needed.
2. Keep peak VRAM ≤ ~14.5 GiB (≥1.5 GiB slack). WSL2 ignores NVIDIA's
   "Prefer No Sysmem Fallback" setting ([WSL#11050](https://github.com/microsoft/WSL/issues/11050), closed stale),
   so there is **no OOM guardrail** — you get a 100–700× slowdown instead of an error.
3. Never trust "it loaded". Validate every config with a timed generation;
   `bench/legion/find_max_context.sh` does this automatically (`SPILL_THRESHOLD_TPS`).

## Measured: context fit (all with `--parallel 1`, `-fa on`, CUDA graphs ON)

| ctx | KV | peak VRAM | decode | verdict |
|---:|---|---:|---:|---|
| 8192 | q8_0 | 12,458 MiB | 27.18 | OK |
| 16384 | q8_0 | 12,760 MiB | 26.69 | OK |
| 24576 | q8_0 | 13,072 MiB | 26.10 | OK |
| 32768 | q8_0 | 13,384 MiB | 27.02 | OK |
| 32768 | q4_0 | 12,816 MiB | 26.54 | OK |
| 49152 | q4_0 | 13,176 MiB | 27.13 | OK |
| 65536 | q4_0 | 13,536 MiB | 27.33 | OK |

**Context is not the constraint.** Decode is flat 8k→64k (27.18 → 27.33 t/s) because only
16 of 64 layers hold a growing KV cache. 64k costs ~13.5 GiB, leaving ~2.4 GiB slack.

## Measured: other levers
- **CUDA graphs ON = +20%** (30.74 vs 25.55 t/s, llama-bench). **No Xid 8 hang observed**,
  contrary to llama.cpp#27330 (which is reported on RTX 5090 Laptop). Keep graphs ON here.
- llama-bench @ small ctx: **pp512 = 949.9 t/s, tg128 = 30.7 t/s**, peak 12,600 MiB.
- Peak power *draw* hit **110.3 W against a 95 W cap** at only **56 °C** — Dynamic Boost
  overshoots transiently; we are power-limited, never thermally limited.
- **Verified independently**: the GGUF contains **24 tensors in 2-bit classes totalling
  2,002,780,160 params = 7.33%** of the model (IQ2_S×15, IQ2_XS×4, Q2_K×3, IQ2_XXS×2).
  This confirms the community complaint about Unsloth's Dynamic-V3 rebuild. Revision
  `408fcc18` (12.52 GiB) reportedly has zero 2-bit tensors — worth a quality head-to-head.
- Windows interop works from WSL (`powershell.exe`, `nvidia-smi.exe`). Windows-side
  `nvidia-smi.exe` confirms `power.max_limit = 175 W`. Power plan is **Balanced**.
  Lenovo `LENOVO_GAMEZONE_DATA` WMI returns **Access denied** unelevated → the Fn+Q
  Performance switch must be done by the user.

## Verified model architecture (read from the GGUF + upstream `config.json`, not from memory)

`Qwen/Qwen3.8-27B` → `model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`.
**GGUF declares `general.architecture = qwen35`** — the arch mainline llama.cpp already
supports as `LLM_ARCH_QWEN35`. So this is a Qwen3.5-family hybrid, not a new arch.

| Property | Value |
|---|---|
| Layers | 64 (+1 MTP) — **16 full-attention, 48 linear-attention** (`full_attention_interval=4`) |
| Attention | 24 Q heads, 4 KV heads (GQA 6:1), **head_dim 256**, `attn_output_gate=true` |
| Linear attn | GatedDeltaNet: 16 key heads / 48 value heads, head dim 128, conv kernel 4, state 128 |
| Hidden / FFN | 5120 / 17408 |
| Vocab | **248320** (untied lm_head → ~2.5B params in embeddings alone) |
| Context | 262144 native; mRoPE interleaved, `mrope_section=[11,11,10]`, θ=1e7, `partial_rotary_factor=0.25` |
| MTP | `mtp_num_hidden_layers=1` — 1-token draft head, shipped as a separate 1.28 GiB GGUF |
| Vision | 27-layer tower, hidden 1152 → out 5120 (out of scope for coding) |
| Model-card sampling defaults (in GGUF kv) | `temp=1.0, top_p=0.95, top_k=20` |

**KV cache math** (derived): only the 16 full-attention layers hold a growing cache.
`16 layers × 4 kv heads × 256 dim × 2 (K+V) × 2 B = 64 KiB/token` at f16.

| ctx | f16 | q8_0 | q4_0 |
|---:|---:|---:|---:|
| 32k | 2.0 GiB | 1.0 GiB | 0.5 GiB |
| 64k | 4.0 GiB | 2.0 GiB | 1.0 GiB |
| 128k | 8.0 GiB | 4.0 GiB | 2.0 GiB |
| 262k | 16.4 GiB | 8.2 GiB | 4.1 GiB |

The 48 linear-attention layers use a *constant* recurrent state (~150 MiB/sequence), independent
of context length — this is why long context is affordable on a 16 GB card.

**Budget**: 15.92 GiB total − 12.24 GiB weights − ~0.9 GiB compute/CUDA ctx ≈ **2.8 GiB for KV**
→ roughly **44k @ f16, 88k @ q8_0, 176k @ q4_0**. No display is attached to the GPU in WSL2
(`Display Active: Disabled`), so the full 16 GB is usable.
Loading the MTP head costs ~1.28 GiB of that budget — speculative decoding vs context is a real tradeoff to measure.

## Verified GPU facts
- `compute_cap 12.0` (sm_120); llama.cpp configure auto-promotes `120` → **`120a`** (Blackwell-specific kernels).
- **`power.max_limit = 175 W`, `power.default_limit = 80 W`, current ceiling `90.17 W`.** We are running at
  ~half the GPU's available power envelope. `nvidia-smi -pl` needs root and WSL2 likely cannot set it —
  must be changed Windows-side (Legion power mode / Vantage / Fn+Q). **Ask the user.**
- Max mem clock 14001 MHz, max SM clock 3090 MHz, PCIe gen5 x16.

## Benchmark-research findings (subagent C, 2026-08-27) — verified on this box

**1. `podman` 3.4.4 is installed and its Docker-compatible socket is enabled.** "No Docker" is a much
softer constraint than assumed. A real SWE-bench eval image was pulled and executed successfully.
- `export DOCKER_HOST=unix:///run/user/$(id -u)/podman/podman.sock` → `docker.from_env()` harnesses work
- `export MSWEA_DOCKER_EXECUTABLE=podman` → mini-swe-agent works
- Harbor exposes `podman` and `singularity` as first-class environment types
- ⚠️ podman 3.4.4 is the Ubuntu 22.04 default (~2021). Docker-API compat is partial; newer harnesses may need podman 4.x/5.x.

**2. The GGUF's own chat template uses the Qwen3-Coder XML tool format, NOT Hermes JSON.** Read
directly out of `Qwen3.8-27B-UD-Q3_K_XL.gguf` (`tokenizer.chat_template`, 9993 chars):
```
<tool_call><function=NAME><parameter=KEY>value</parameter></function></tool_call>
```
- vLLM must use `--tool-call-parser qwen3_xml` (or `qwen3_coder`) — **`hermes` will silently fail**
- SGLang must use `--tool-call-parser qwen3_coder`, **not** `qwen`
- llama.cpp must derive a "Constructed" PEG parser — the hardest of its three shapes

**3. ⚠️ MULTI-TURN TOOL-CALL LANDMINE (verified, template lines 148-151).** The template calls
`raise_exception` if `tool_calls[].function.arguments` arrives as a **JSON string** — which is
exactly what the OpenAI API spec mandates and what every agent framework replays on turn 2+:
```jinja
{%- elif tool_call.arguments is string %}
  {%- if tool_call.arguments|trim %}
    {{- raise_exception('Tool call arguments ... were passed as a JSON string. Parse them into an object ...') }}
```
Single-turn tool calls will pass; agentic loops will 500 on the second turn. Must be tested explicitly
and, if it bites, patched via `--chat-template-file`.

**4. llama.cpp deleted all per-model tool-call parsers in March 2026 (PR #18675).** `common_chat_format`
is now only `CONTENT_ONLY / PEG_SIMPLE / PEG_NATIVE / PEG_GEMMA4 / PEG_MINIMAX_M3` — parsers are
*derived* from the Jinja template by differential analysis. `docs/function-calling.md` is stale;
`docs/development/parsing.md` and `docs/autoparser.md` are current. There is **no flag to force a
parser**; the only lever is `--chat-template-file`.
- Validator: `cmake --build build --target test-chat-auto-parser && ./build/bin/test-chat-auto-parser <gguf> --input-message=all --output=both`
  (the widely-cited name `debug-template-parser` does **not** exist in master as of today)

**5. ⚠️ GPU power cap drifts.** `enforced.power.limit` was **90 W** during Phase 0 recon and **65 W**
a few hours later, on AC, idle (default 80 W, max 175 W). Any t/s number not accompanied by a
logged power limit is meaningless, and engine A-vs-B blocks must be **interleaved**, not sequential.

**6. `-fa` now requires a value** (`-fa on`). Bare `-fa` no longer parses — nearly every pre-2026
guide is wrong on this. Also new and useful here: `llama-fit-params` (auto-fits ngl/ctx to free VRAM)
and `--spec-type draft-mtp` (drives the already-downloaded 1.37 GiB MTP head).

**7. Stale/dead benchmarks to skip:** BigCodeBench (archived), CodeContests (archived 2023),
SWE-Lancer (archived → `openai/preparedness`), RepoBench, CRUXEval, SWE-Gym, R2E-Gym.
Renames: `All-Hands-AI/OpenHands`→`OpenHands/OpenHands`, `princeton-nlp/SWE-bench`→`SWE-bench/SWE-bench`,
`laude-institute/terminal-bench`→`harbor-framework/terminal-bench-1` (Terminal-Bench 2.x now runs under **Harbor**).
Slowing: LiveCodeBench (2025-07-16), EvalPlus (2025-10-02), aider (2026-05-22), BFCL (2026-03-23).

**8. Q3 prior is unfavourable.** Unsloth's own Qwen3.6-27B table: 3-bit costs **3.2× the mean KLD**
of 4-bit (0.0734 vs 0.0227) and 2.4× the P99.9 tail, for **8 % disk saved**. Budget maths says
UD-IQ4_XS (13.27 GiB) still leaves ~1.8 GiB for KV. Worth benching Q3_K_XL vs IQ4_XS head-to-head
before committing the daily driver.

## Open questions / decisions
- Which quant to make the daily driver: `UD-Q3_K_XL` (12.24 GiB, requested) vs `UD-IQ4_XS` (13.27 GiB) vs `UD-Q4_K_S` (14.30 GiB)? Larger = better quality but squeezes KV cache. Decide after quality bench.
- Is native Windows worth the effort? vLLM/SGLang effectively require Linux/WSL2.
- Vision (`mmproj`) is out of scope for coding; not downloaded.

## Notes
- Model is dense 27B (BF16 ≈ 50.9 GiB across 2 shards).
- Repo ships an MTP head → self-speculative decoding is available; likely the biggest throughput lever on a bandwidth-bound laptop GPU.
