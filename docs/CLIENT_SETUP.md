# Connecting Open WebUI / AnythingLLM / any OpenAI client to Qwen3.8-27B

Values below are read from the running server (`/v1/models`, `/props`) and from the model's
own metadata — not from documentation. Benchmarks behind the recommendations are in
`QWEN38_27B_LEGION_BENCHMARKS.md`.

---

## 1. Connection

| setting | value |
|---|---|
| **Base URL** | `http://127.0.0.1:8080/v1` |
| **Model id** | `qwen3.8-27b` |
| **API key** | none required — but most UIs demand a non-empty string, so put `sk-local` |
| Capabilities reported | `completion`, `multimodal` |

⚠️ **The server binds to `127.0.0.1` by default**, so a container or another machine cannot
reach it. If Open WebUI / AnythingLLM runs in Docker or on another host:

```bash
./scripts/serve-qwen38.sh --bind 0.0.0.0 --api-key "sk-pick-something"
```
```powershell
.\start-qwen38.ps1 -Bind 0.0.0.0 -ApiKey "sk-pick-something"
```

Then from a container use `http://host.docker.internal:8080/v1` (Docker Desktop) or the host's
LAN IP. **Set an API key whenever you bind beyond localhost** — otherwise anyone on the network
has an open LLM endpoint.

Without `--alias` the model id is the full `.gguf` path (`C:\llm\models\Qwen3.8-27B-…gguf`),
which several UIs mangle. The launchers set `--alias qwen3.8-27b` for this reason.

---

## 2. Context window and token limits

**128k context is achievable and usable** — measured, not extrapolated. The trade is speculative
decoding: the MTP draft head costs 1.83 GiB, which is worth about 80k tokens of q4_0 KV.

### With MTP + vision (`--mode both`), q4_0 KV

| ctx | peak VRAM | slack | verdict |
|---:|---:|---:|---|
| 32k | 14,806 MiB | 1.5 GiB | ⭐ default — comfortable |
| 48k | 15,254 MiB | 1.0 GiB | fine |
| 64k | 15,645 MiB | 0.66 GiB | works, but near the eviction cliff |

### Without MTP (`--mode long` / `vision`), q4_0 KV

| ctx | peak VRAM | slack |
|---:|---:|---:|
| 64k | 14,829 MiB | 1.4 GiB |
| 96k | 15,565 MiB | 0.7 GiB |
| **128k** | **15,731 MiB** | 0.56 GiB |

### Decode actually holds up at depth

Measured on a 128k server by filling the context, not just allocating it:

| depth | 20k | 41k | 61k | 85k | 100k | 115k |
|---|---:|---:|---:|---:|---:|---:|
| decode t/s | 38.8 | 34.6 | 31.0 | 27.2 | 25.6 | **24.3** |

Output stayed coherent throughout. **llama.cpp#27623's reported ~25× collapse past 80k did not
reproduce here**, nor did #27756's instant-EOS — decode just declines ~30% from peak to 115k.

| | value |
|---|---|
| Model's architectural max | 262,144 (YaRN to 1M; not reachable in 16 GB) |
| KV cost | ~39 KiB/token at q8_0, ~22.5 KiB/token at q4_0 |

⚠️ Past 16k the KV must be **q4_0**, and q4_0 on **K** is the quality-sensitive half
(llama.cpp#21591 — q4_0 on K alone can reproduce a quality collapse, while V-only costs ~1/500).
The launchers switch automatically, but this repo has **not** quality-tested q4_0 K on this model.
If you need long context *and* maximum fidelity, that is the thing to verify first.

**`n_ctx` is fixed when the server loads.** A client-side "context length" box does not change
it — if the UI sends more than the server's `n_ctx` the request truncates or errors. Set the UI's
context to match the server, never above. Check it with
`curl -s localhost:8080/props | jq .default_generation_settings.n_ctx`.

### max_tokens (output budget)

It shares the window with your prompt: `prompt + max_tokens ≤ n_ctx`.

| server ctx | practical `max_tokens` | notes |
|---:|---:|---|
| 32k (`both`, default) | **8192**–16384 | median generation is 620 non-thinking / ~1100 thinking |
| 48k–64k | 16384 | |
| 128k (`long`) | **32768+** | plenty of room; the limit becomes patience, not memory |

Thinking traces can reach 10k+ tokens on hard problems, so do not set `max_tokens` below ~8192
when thinking is on — truncating mid-reasoning returns an empty or broken answer.

---

## 3. Sampling — use Qwen's official values

Two distinct presets. Using the thinking preset in non-thinking mode (or vice versa) measurably
degrades output.

| parameter | **thinking** | **non-thinking** |
|---|---:|---:|
| temperature | **1.0** | **0.7** |
| top_p | **0.95** | **0.80** |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 0.0 | **1.5** |
| repeat_penalty | 1.0 | 1.0 |

**Do not use temperature 0 / greedy** — Qwen explicitly warns it degrades this model, and it can
cause repetition loops.

---

## 4. Thinking control

The launchers already pass `--reasoning-effort medium`, which is the recommended default:
thinking is worth roughly **2× on coding** (58.8% vs 38.2% pass@2), while `xhigh` costs ~220 s
per task for little gain.

Per-request override, if your client can send extra body fields:

```json
{ "chat_template_kwargs": { "reasoning_effort": "medium" } }
```

Valid values: `xhigh` (model default) · `medium` · `low`. To disable thinking entirely, restart
with `--no-thinking` (`-NoThinking` on Windows) rather than fighting it per-request — llama.cpp
now deprecates toggling it through `chat_template_kwargs.enable_thinking`.

Reasoning appears in a separate `reasoning_content` field. Open WebUI renders it as a collapsible
"thinking" block automatically.

---

## 5. ⚠️ Concurrency — the setting that will bite you

**The server has exactly one slot (`total_slots: 1`)**, because `--parallel 1` is mandatory on
this GPU: the default of 4 slots pushes VRAM past the ceiling and the driver silently evicts the
model to system RAM, collapsing decode ~700× (39.8 → 0.04 t/s).

Open WebUI issues **background requests alongside your chat** — title generation, tag generation,
follow-up suggestions, and RAG query rewriting. With one slot these queue behind your reply, so
the UI feels stalled for seconds after every message.

**In Open WebUI → Admin → Settings → Interface, turn off:**
- Title Generation
- Tags Generation
- Follow-Up Generation
- Autocomplete / "Retrieval Query Generation"

Or point those specific features at a small separate model. Raising `--parallel` is the wrong
fix here — it costs roughly 850 MB of VRAM per extra slot at 32k and walks straight into the
eviction cliff.

---

## 6. Vision

Reported as `multimodal`; images go through standard `image_url` content parts, so any
OpenAI-compatible client works unchanged. In Open WebUI simply enable image upload on the model.

| | |
|---|---:|
| cost of a 640×360 image | **~250 prompt tokens** |
| decode with an image in context | **48.1 t/s** |
| first encode of a *new* image | ~3.1 s (projector runs on CPU) |
| repeat queries on the same image | ~0.19 s (llama.cpp caches it) |

Requires `--mode both` or `--mode vision`. In `fast` mode there is no projector loaded and image
requests fail.

---

## 7. Tool / function calling

`/props` reports the template supports it:

```
supports_tools                : true
supports_tool_calls           : true
supports_parallel_tool_calls  : true
supports_object_arguments     : true
```

The server runs with `--jinja`, which is required for tool calls to be parsed. Note this repo has
**not** benchmarked tool-calling accuracy — the 100% well-formed diff rate in the polyglot run is
adjacent evidence, not the same thing.

---

## 8. Throughput to expect

| workload | t/s |
|---|---:|
| text decode, code (MTP) | **~88** |
| text decode, prose | ~58 |
| with an image in context | ~48 |
| prefill | ~1120–1350 |

These assume the GPU is at **175 W** (Lenovo Performance mode). At the ~95 W Balance cap expect
roughly −32% decode and −42% prefill — measured 88 t/s vs 58–67 t/s on the identical config.

⚠️ **Performance mode does not survive a reboot.** After a restart the EC drops back to ~95 W
even though `nvidia-smi` still reports `Current Power Limit: 175 W`. The reliable tell is under
load: `clocks_event_reasons.active = 0x4` (SW Power Cap) with the SM clock at ~1357 MHz instead
of ~2482 MHz. Press **Fn+Q → Performance** after every boot, or you silently lose a third of your
throughput.

```bash
nvidia-smi --query-gpu=clocks.sm,power.draw,clocks_event_reasons.active --format=csv -l 1
```

---

## 9. AnythingLLM specifics

Use **Generic OpenAI** as the LLM provider (not "LocalAI"/"Ollama", which assume different paths):

| field | value |
|---|---|
| Base URL | `http://127.0.0.1:8080/v1` |
| API Key | `sk-local` |
| Chat Model Name | `qwen3.8-27b` |
| **Token context window** | `16384` (or 32768 in `fast` mode) |
| Max Tokens | `4096` non-thinking / `8192` thinking |

AnythingLLM's context-window field is used for *its own* prompt budgeting — set it to the real
`n_ctx` or it will over-stuff the prompt and the server will truncate.

For embeddings use a separate model; this server exposes a chat model only, and pointing an
embedder at it will fail or waste the single slot.
