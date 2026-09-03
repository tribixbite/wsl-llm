# Qwen3.8-27B production serving spec (desktop `matilda`, 2× RTX 3090)

**Status:** GPU 0 runs the fastest known single-card stack; GPU 1 is the experiment lane.
All client traffic goes through the monitor proxy so the web dashboard sees every call.

| | |
|---|---|
| **OpenAI base URL (use this)** | `http://192.168.1.32:8090/v1` |
| **Model name** | `qwen3.8-27b` |
| **API key** | *none required* — the proxy injects the upstream key |
| **Context (`max_model_len`)** | **65536** (64k) |
| **Vision** | ✅ **enabled** — send OpenAI `image_url` parts |
| **`max_tokens` to send** | **≥1000**, recommend **4000–8000** (see "thinking" below) |
| **Web dashboard** | `http://192.168.1.32:8090/` |
| **Direct upstream (bypasses dashboard)** | `http://192.168.1.32:18020/v1`, key `c67e38c5fc66f203348ff28ad05e41d552eb56a96e5b9f10` |

## ⚠️ `max_tokens` must be generous — thinking is ON by default

Qwen 3.8 is thinking-first. Reasoning is emitted **before** the answer and consumes the
same `max_tokens` budget. A measured example: a *haiku* request used **337 completion
tokens, 319 of them reasoning**. With `max_tokens: 10` you get an **empty** `content`
and `finish_reason: length`.

- Chat/agent use: `max_tokens` 4000–8000
- Reasoning appears in a separate `reasoning` field (vLLM `--reasoning-parser qwen3`)
- To reduce/disable thinking: `chat_template_kwargs: {"reasoning_effort": "medium"|"low"|"none"}`
  (`xhigh` is the model default and is ~2–4× slower — measured, and it cap-truncates on
  hard problems; use `medium`)

## Sampling parameters (official Qwen/Unsloth + r/LocalLLaMA consensus)

vLLM applies these automatically from the model's `generation_config.json` — confirmed
in the server log:
`Default vLLM sampling parameters have been overridden by the model's generation_config.json: {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}`

| Mode | temp | top_p | top_k | min_p | presence_penalty |
|---|---:|---:|---:|---:|---:|
| **Thinking (default, use this)** | **1.0** | 0.95 | 20 | 0.0 | 0.0 |
| Instruct / non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

**Do not use temp 0 / greedy.** Community finding: *"Temp of 0 greatly cripples the
intelligence of Qwen 3.8."* Also note this stack sets `draft_sample_method=probabilistic`
for MTP, which is **incompatible with greedy sampling**. Temperature has no measurable
speed cost (77.5 t/s @ temp 0 vs 77.9 @ temp 1.0).

## Measured performance (this box)

| Config | prose | code | json | avg |
|---|---:|---:|---:|---:|
| **vLLM W4A16 + MTP n=4, `MAX_SEQS=1`** | 74.5 | **116.4** | 124.1 | **105.0** |
| llama.cpp Q6_K_XL dual-GPU + MTP | 41.1 | 59.3 | 60.0 | 53.5 |
| llama.cpp Q3_K_XL 1-GPU + MTP + vision | 36.9 | 52.7 | 53.5 | 47.7 |

MTP speculative-decode acceptance measured live: **58.4%** (matches the ~59–61% others
report for this stack). Quality: **73.0% pass@2** on the aider polyglot subset
(n=74, python + javascript).

Context: the r/LocalLLaMA consensus band for a single 3090 is 45–75 t/s on 4-bit GGUF
with MTP; nothing credibly beats ~133 t/s single-stream on this card.

## GPU 0 — the production stack (full run command)

Repo: [`syv-ai/qwen38-27b-rtx3090`](https://github.com/syv-ai/qwen38-27b-rtx3090) at
`~/qwen38-vllm`, venv install (not Docker — Docker has no GPU access in WSL2).

```bash
cd ~/qwen38-vllm
VISION=1 VISION_OFFLOAD=1 \
MODEL=$HOME/qwen38-vllm/models2/Qwen3.8-27B-W4A16-AutoRound \
MAX_SEQS=1 \
bash single-user/start_qwen.sh
```

`VISION=1` adds the vision tower (only **0.858 GiB**, kept in pinned host RAM by
`VISION_OFFLOAD=1`). Measured cost: **105.0 → 95.3 t/s (−9%)** for full image support
— the best speed/vision trade available on one card. Verified reading `TEST-7429`
from a test image through the proxy in 3.1 s.

which expands to:

```bash
vllm serve /home/matilda/qwen38-vllm/models2/Qwen3.8-27B-W4A16-AutoRound \
  --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 18020 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --api-server-count 1 \
  --language-model-only \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype float16 \
  --async-scheduling \
  --max-num-batched-tokens 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

### ⚠️ `MAX_SEQS=1` is mandatory
With the default 8 slots at `gpu-memory-utilization 0.93` (~22.3 of 24 GB) the model
loads, LISTENs, logs `Application startup complete` — then **WDDM silently evicts the
weights to system RAM** (WSL2 has no OOM guardrail). `/health` keeps answering while
every real request times out. Same trap as llama.cpp `--parallel 4`.
**Validate any change with a timed generation, never `/health`.**

## Monitor proxy (all calls flow through it)

```bash
cd ~/git/wsl-llm
python3 scripts/llm-monitor.py --port 8090 \
  --upstream http://127.0.0.1:18020 \
  --api-key "$(cat /tmp/vllm_key.txt)" \
  --params "temperature=1.0,top_p=0.95,top_k=20,min_p=0.0"
```

Dashboard at `http://192.168.1.32:8090/` shows: backend + model, context, live MTP
acceptance rate, KV-cache use, per-request prompt/completion/**reasoning** token counts,
decode t/s, TTFT, and any images clients send. Requests are proxied verbatim including
SSE streaming, so client behaviour is unchanged.

## Client examples

```bash
curl http://192.168.1.32:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.8-27b",
       "messages":[{"role":"user","content":"hi"}],
       "max_tokens":4000,
       "temperature":1.0,"top_p":0.95,"top_k":20}'
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://192.168.1.32:8090/v1", api_key="not-needed")
r = c.chat.completions.create(model="qwen3.8-27b",
        messages=[{"role":"user","content":"hi"}],
        max_tokens=4000, temperature=1.0, top_p=0.95)
```

Any OpenAI-compatible client works (Open WebUI, Continue, aider, LiteLLM) — set base URL
to `http://192.168.1.32:8090/v1`, model `qwen3.8-27b`, any non-empty API key string.

## GPU 1 — experiment lane

Currently the vision endpoint (`qwen38-vision.service`, llama.cpp + mmproj, port 8085).
Stop it to free the card for experiments: `sudo systemctl stop qwen38-vision`.
