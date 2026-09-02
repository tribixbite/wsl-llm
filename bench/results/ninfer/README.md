# NInfer (Don-Chad/ninfer-3090 v0.6.1) on RTX 3090 — 2026-09-02

Build recipe: `docs/NINFER_BUILD_UBUNTU2204.md` (CUDA 12.9 toolkit, gcc-13, 2 API backports).

Serve: `--max-context 65536 --kv-capacity 65536 --max-concurrency 1
--prefill-chunk 1024 --kv-dtype int8 --spec mtp --draft-tokens 3 --lm-head-draft`,
GPU 1, **200 W power cap**, 20.9 GB, weights load in 20.8 s.

## Throughput (C1, single stream)

| | prose | code | json | avg |
|---|---:|---:|---:|---:|
| NInfer C1 | 32.6 | 51.7 | 50.0 | **44.8** |
| vLLM W4A16 + MTP (GPU 0) | 74.5 | 116.4 | 124.1 | **105.0** |

**NInfer is ~2.3× slower than vLLM for single-stream on this box**, and below the
author's published C1 71 t/s. Two caveats before concluding the fork is slow:
our card is **capped at 200 W** (author ran 250 W; the community measures 220 W→30 t/s
vs 250 W→35–40 on llama.cpp, so the cap is worth a lot), and GPU 0 was serving
concurrently. NInfer's published advantage is **concurrency** (C8 165 t/s), which we
did not test — C1 is its weakest configuration.

TTFT is good: ~310 ms.

## Quality — aider polyglot, temp 1.0, reasoning_effort=medium, 2 attempts

| Language | n | pass@1 | pass@2 |
|---|---:|---:|---:|
| JavaScript | 40 | 67.5% | **95.0%** |
| Python | 34 | 29.4% | 61.8% |
| **Overall** | **74** | 50.0% | **79.7%** |

⚠️ **Not directly comparable to our earlier vLLM 73.0%** — that run used temp 0.6,
this one temp 1.0. Engine and sampling both changed. A vLLM re-run at temp 1.0 is
in `aider_vllm_temp1.json` to isolate the variable.

## API quirk

NInfer rejects `chat_template_kwargs.reasoning_effort` with
`chat_template_option_not_supported` (HTTP 400) and wants the **OpenAI top-level
`reasoning_effort`** field instead. vLLM/llama.cpp want the kwargs form.
`bench/aider_multi.py --effort-style top_level|kwargs` handles both.
