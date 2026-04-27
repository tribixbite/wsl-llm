# EAGLE-3 Drafter Training Plan for Qwen3.6-27B

## Why

Our production stack (vLLM + Genesis + MTP n=3 + Lorbus AutoRound INT4) hits **67 t/s code / 46 t/s prose** decode_TPS — at the bandwidth ceiling for 27B Dense + RTX 3090.

**EAGLE-3 typically delivers 3-4× over baseline vs MTP's 2×.** For our workload that translates to expected **80-100 t/s code / 60-75 t/s prose** if we can train a Qwen3.6-27B-specific EAGLE-3 head. **No such head is published yet** (the [Qwen3.6 collection on HF](https://huggingface.co/collections/Qwen/qwen36-69e0ce993efc132aabacb11d) only contains the base + FP8 weights; no EAGLE/spec heads).

## Toolchain: SpecForge

[SpecForge](https://github.com/sgl-project/SpecForge) (LMSYS, requires Python 3.11+) is the canonical EAGLE-3 trainer. Output checkpoints plug into SGLang via `--speculative-algorithm EAGLE3 --speculative-draft-model-path` and into vLLM via `speculative_config:{method:eagle3,...}`.

Setup:
```bash
git clone https://github.com/sgl-project/SpecForge.git ~/SpecForge
uv venv ~/specforge_env --python 3.11
uv pip install --python ~/specforge_env/bin/python setuptools wheel
uv pip install --python ~/specforge_env/bin/python --no-build-isolation -e ~/SpecForge
```

## Hardware constraint analysis

EAGLE-3 training has two phases:
1. **Hidden-state generation** (`prepare_hidden_states.py`) — runs the **frozen target** over a calibration corpus to record per-layer hidden states. SpecForge uses SGLang backend by default ("to minimize precision mismatch in training and serving" per `build_target_model` in script).
2. **Train drafter** (`train_eagle3.py`) — trains a single-layer transformer drafter (~0.7-1.1B params for 27B target) against the cached hidden states. Pure compute on the drafter; target weights aren't needed.

VRAM by phase:

| Phase | Target dtype | VRAM needed | Fits on 1× RTX 3090 (24 GB)? |
|-------|--------------|-------------|------------------------------|
| Hidden states | BF16 (Qwen/Qwen3.6-27B) | ~54 GB | ❌ no — needs TP=2 |
| Hidden states | INT4 (Lorbus AutoRound) | ~18 GB + KV | ✅ **likely yes via SGLang `--quantization auto-round`** |
| Hidden states | FP8 (Qwen/Qwen3.6-27B-FP8) | ~28 GB | ❌ no — but with offload maybe |
| Train drafter | (target frozen + cached states) | ~10 GB | ✅ comfortably |

**Key risk:** AutoRound INT4 hidden states have **slightly degraded precision** vs BF16. The drafter trained on these may have lower acceptance rate than ideal. Per the agent's research: *"Safer path: do hidden-state generation once against a BF16 target and keep INT4 only for serving."* Still worth trying since it's the only single-3090 path.

## Path A — INT4 hidden states (single 3090, ~16h total)

```bash
# 1. Prepare calibration data (CPU, ~10 min)
~/specforge_env/bin/python ~/SpecForge/scripts/prepare_data.py \
    --dataset sharegpt --num-samples 30000

# 2. Generate hidden states with SGLang + auto_round backend
#    (Lorbus AutoRound INT4 = 18 GB on disk; should fit on 24 GB GPU)
#    ~6-10h depending on samples and TP
torchrun --nproc_per_node=1 ~/SpecForge/scripts/prepare_hidden_states.py \
    --target-model-path ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound \
    --enable-aux-hidden-states \
    --data-path ~/SpecForge/cache/dataset/sharegpt_train.jsonl \
    --output-path ~/SpecForge/cache/hidden_states/qwen36-27b-int4 \
    --chat-template qwen \
    --max-length 2048 \
    --tp-size 1 \
    --batch-size 4 \
    --num-samples 30000 \
    --sglang-mem-fraction-static 0.85 \
    --trust-remote-code

# 3. Train drafter (single 3090, ~10h)
torchrun --nproc_per_node=1 ~/SpecForge/scripts/train_eagle3.py \
    --target-model-path ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound \
    --train-hidden-states-path ~/SpecForge/cache/hidden_states/qwen36-27b-int4 \
    --output-dir ~/SpecForge/outputs/qwen36-27b-eagle3 \
    --batch-size 4 --learning-rate 3e-4 --ttt-length 7 \
    --num-epochs 1 --bf16

# 4. Serve via vLLM 0.17 + Genesis + EAGLE-3
PATH=$HOME/bench_env/bin:$PATH \
CUDA_VISIBLE_DEVICES=0 \
~/bench_env/bin/python -m vllm.entrypoints.cli.main serve \
    ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound \
    --served-model-name qwen3.6-27b \
    --quantization auto_round --dtype float16 \
    --tensor-parallel-size 1 --max-model-len 32000 \
    --gpu-memory-utilization 0.93 --max-num-seqs 1 \
    --kv-cache-dtype fp8_e5m2 \
    --enable-prefix-caching --enable-chunked-prefill \
    --speculative-config '{"method":"eagle3","model":"~/SpecForge/outputs/qwen36-27b-eagle3","num_speculative_tokens":7}' \
    --port 8081 --trust-remote-code
```

Or via SGLang:
```bash
~/sglang_env/bin/python -m sglang.launch_server \
    --model-path ~/models/Lorbus-Qwen3.6-27B-int4-AutoRound \
    --quantization auto-round --dtype bfloat16 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path ~/SpecForge/outputs/qwen36-27b-eagle3 \
    --speculative-num-steps 7 --speculative-num-draft-tokens 64 \
    --speculative-eagle-topk 4 \
    --port 8082 --trust-remote-code
```

## Path B — BF16 hidden states (TP=2, ~12h total)

Requires temporarily releasing GPU 1 from the user's other project for ~6-8 hours.

```bash
# Identical to Path A but with --tp-size 2 and BF16 target
torchrun --nproc_per_node=2 ~/SpecForge/scripts/prepare_hidden_states.py \
    --target-model-path Qwen/Qwen3.6-27B \   # NOT the INT4 quant
    --tp-size 2 --batch-size 2 \
    ... # rest same
```

## Path C — Cloud rental (~$15-30, 6-12h)

Rent an 80GB H100 from RunPod or similar for hidden state generation phase only. Train drafter locally afterward. Cleanest, isolates risk. Cost ~$2-3/h × 6-10h = $12-30 total.

```bash
# On rented H100:
torchrun --nproc_per_node=1 ~/SpecForge/scripts/prepare_hidden_states.py \
    --target-model-path Qwen/Qwen3.6-27B \  # BF16, 54 GB fits on H100
    --tp-size 1 --batch-size 8 \
    ... # rest same

# Then rsync hidden states home
rsync -avz h100:~/SpecForge/cache/hidden_states ~/SpecForge/cache/

# Train locally as in Path A step 3
```

## Realistic expected outcome

EAGLE-3 typical acceptance rate is 0.7-0.8 vs MTP's 0.5-0.7. For our workload:

| Configuration | Decode TPS (code) | Decode TPS (prose) |
|---------------|------------------:|-------------------:|
| Current vLLM+Genesis+MTP n=3 | 67 | 46 |
| **Estimated EAGLE-3 (BF16 hidden states)** | **85-100** | **65-80** |
| **Estimated EAGLE-3 (INT4 hidden states)** | **75-90** | **55-70** |
| Theoretical bandwidth-bound max with perfect spec | ~150 | ~90 |

INT4 hidden states are expected to give ~10-15% lower acceptance rate than BF16, which translates to ~10-15% lower TPS. Still likely above the **80 t/s target** on code workloads.

## Decision tree

1. **Try Path A first** — only needs single 3090, downside is unsupported INT4 path. ~16h overnight job, can be aborted if it fails early.
2. **If A fails** → ask user for GPU 1 access for ~6-8h (Path B).
3. **If GPU 1 not available** → rent cloud H100 (Path C, ~$20).

## Status (2026-04-27)

- ✅ SpecForge env installed at `~/specforge_env` (Python 3.11 + specforge editable)
- ✅ SpecForge scripts available at `~/SpecForge/scripts/`
- ⏸ Hidden state generation NOT YET STARTED — pending decision on path
- ⏸ Trainer config NOT YET DRAFTED

## Open questions for the user

1. Is GPU 1 available for ~6-8h overnight (Path B)?
2. Or budget for cloud rental (Path C, ~$20)?
3. Or proceed with the unsupported INT4 path (Path A) and accept lower acceptance rate?
