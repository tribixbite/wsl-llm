# Windows agent handoff — Qwen3.8-27B on 2× RTX 3090

Everything here is either **impossible from WSL2** or **expected to be materially
faster natively**. Numbers below were measured on this box under WSL2 unless marked
otherwise. Read `docs/QWEN38_PRODUCTION_SPEC.md` for the current serving setup.

## Hardware / current baseline (WSL2, for comparison)

| | |
|---|---|
| GPUs | 2× RTX 3090 24 GB, Ampere sm_86, PCIe (**no NVLink**), GPU0 x4 / GPU1 x8 |
| Power cap | **200 W** ← the big problem, see task 1 |
| Best measured decode | **105 t/s avg / 116 code** (vLLM W4A16 + MTP, 1 card) |
| Best measured quality | **79.7% pass@2** aider polyglot (n=74, temp 1.0) |
| Driver | 591.86, CUDA 13.1 runtime |

---

## Task 1 — RAISE THE POWER CAP (highest value, do this first)

**Cannot be done from WSL2.** GPU-PV puts the driver host-side; `sudo nvidia-smi -pl`
inside WSL returns `Insufficient Permissions` even as root (verified 2026-09-02).

Published measurement on a 3090, 14-min sustained load (syv-ai/qwen38-27b-rtx3090#62):

| cap | decode | SM clock |
|---|---:|---:|
| 200 W | 57.5 t/s | 781 MHz |
| **250 W** | **85.6 t/s** | 978 MHz |
| 280 W | no gain — 90 °C in ~2 min, then throttles |

Run `windows/gpu-init.bat` (already updated to 250 W) **as Administrator**.

⚠️ This box has a history of **hard resets under dual-GPU load at 350 W × 2 (~700 W)**.
500 W total should be inside budget, but the first time both cards are busy at 250 W,
watch for a reset. If it resets, drop to 225 W.

**Then re-run every benchmark below** — all our numbers are 200 W numbers and are
expected to move ~+30–49%.

## Task 2 — Multi-GPU tensor parallel (impossible under WSL2)

Verified structurally on this box: no NVIDIA kernel module (`lsmod` empty,
`/dev/nvidia*` absent), `libcuda.so.1` is a **175 KB shim** to the Windows KMD, and
`nvidia-smi topo -p2p r` reports **CNS (Chipset Not Supported)**. So TP=2 cannot work
here regardless of engine.

On native Windows/Linux, test:
- **vLLM TP=2.** Note vLLM skips its NVLink pre-filter when `world_size <= 2`, so a
  2-card PCIe rig *does* get the fast custom all-reduce. Expected +16–27%.
  If output is garbage (`!!!!!!`) at plausible TPS, add `--disable-custom-all-reduce`
  (custom all-reduce over patched BAR1 can return wrong data on Ampere).
- **The P2P driver patch** (smcleod.net) — supports Ampere despite the "4090-only"
  folklore; needs IOMMU passthrough + large BAR1. 10–30% reported on a 3090.
- **Do NOT bother with `-sm tensor` in llama.cpp**: the maintainer's own 2×4090 bench
  has tensor **42% slower** than layer split.
- **ExLlamaV3 TP is hopeless for this arch** — author confirmed 128 synchronizations
  per prefill chunk, ~4 GB traffic per 2048-token chunk. Skip.

## Task 3 — Engine bake-off at 250 W

Re-measure all of these with the same harness (`bench/stream_bench.py`,
`bench/aider_multi.py`) so results are comparable to the WSL2 table.

| Engine | WSL2 @200 W | Notes for Windows |
|---|---:|---|
| vLLM W4A16 + MTP (syv-ai) | **105 t/s** | the current champion; `MAX_SEQS=1` mandatory |
| llama.cpp Q6_K_XL dual-GPU + MTP | 53.5 | layer split, not parallel compute |
| llama.cpp Q3_K_XL 1-GPU + MTP | 47.7 | |
| llama.cpp Q5_K_XL 1-GPU + MTP | 36.1 | |
| NInfer C1 (mtp3, int8 KV) | 44.8 | **ships .bat launchers — first-class Windows target** |
| NInfer C8 | untested | author claims **165 t/s**; our C1 is its worst case |
| ExLlamaV3 EXL3 5.0bpw | untested | quant already staged at `~/models/qwen38-exl3-5.0` |
| SGLang | skip on 1 card | 43 t/s @8k, speculation auto-disabled; only wins at TP=2 (141–153) |

**NInfer specifically**: our 44.8 t/s vs the author's published 71 t/s C1 is most
likely the power cap. Windows + 250 W is the fair test, and it has native `.bat`
launchers (`scripts/run-qwen38-c1.bat`, `-c8.bat`, `-vision.bat`).

## Task 4 — Vision configurations

Three working paths; benchmark quality *and* speed with an image in context.

1. **vLLM `VISION=1 VISION_OFFLOAD=1`** — tower is only 0.858 GiB, CPU-offloaded by
   default. Gotchas: keep `--enable-prefix-caching` **off** for multi-turn image work
   (syv-ai#50); set `VLLM_USE_V2_MODEL_RUNNER=1` when combining MTP with images or
   acceptance drops up to −24.8% (vLLM#54498). Stock vLLM ≤0.28.0 **cannot** offload
   the tower — this is a fork feature.
2. **NInfer `--vision`** — verified working here; 34.5 t/s with an image in context.
3. **llama.cpp + mmproj** — projectors already staged at
   `~/models/qwen38/mmproj-F16.gguf` (927 MB) / `mmproj-BF16.gguf`. Zero downloads.

**Sampling finding to carry over:** temp 1.0 is correct for reasoning/coding, but at
temp 1.0 the model misread `TEST-7429` as `TEST-7428`; at temp 0.3 it read it
correctly twice. **Use ~0.2–0.3 for OCR / exact transcription.**

## Task 5 — DFlash2 (blocked on our vLLM version)

`SPEC=dflash2` claims ~130 t/s vs MTP's ~116. On vLLM **0.28.0** it fails with
`AttributeError: 'QKVParallelLinear' object has no attribute 'weight'` — the DFlash2
code reads `.weight`, but W4A16 layers expose `qweight`/`scales`. The patches are
written against **0.27.1**. Two earlier blockers are already solved and worth knowing:
`SPEC=dflash2` in `.env` is **silently ignored** on the venv path (`.env` is
docker-compose only — export it), and `UVA is not available` under WSL2 GPU-PV is
fixed by `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. On Windows/native, try vLLM 0.27.1 + the
full 19-patch set.

## Task 6 — Storage / OS notes (already settled, do not redo)

- **Keep models on ext4, never `/mnt/c`.** Cold read of a 13.5 GiB GGUF: **12.3 s
  ext4 vs 102.8 s on /mnt/c**; warm re-read **1.05 s vs 104 s** (9p never caches,
  `msize=65536` caps throughput).
- **WSL2 storage is not a bottleneck** — the ext4 VHDX hit 4,938 MB/s with 4 parallel
  readers, *beating* native NTFS unbuffered (3,595 MB/s). No storage argument for
  dual-booting.
- Windows mmap is fine on current llama.cpp (`PrefetchVirtualMemory`); the old
  15-minute-load bug was fixed by PR #801. `--no-mmap` is not a general Windows
  recommendation.

## Task 7 — Detector worth adding

`nvidia-smi dmon` during decode: **SM ~100% at only 100–200 W means WDDM is paging
weights to host RAM**. That silent 5–700× slowdown is WSL2's signature failure and it
never surfaces as an error — `/health` keeps answering. **Always validate a config
change with a timed generation, never `/health`.**

## Benchmarks to run (same harness both sides)

```bash
# throughput
python3 bench/stream_bench.py --url http://127.0.0.1:PORT --model qwen3.8-27b --label NAME

# quality: full 225-exercise aider polyglot, 2 attempts, leaderboard metric
python3 bench/aider_multi.py --url http://127.0.0.1:PORT --model qwen3.8-27b \
  --langs python,javascript,java,cpp,go,rust --n 50 \
  --effort medium --tries 2 --temp 1.0 --out results.json
# NInfer needs --effort-style top_level (it 400s on chat_template_kwargs)
```
