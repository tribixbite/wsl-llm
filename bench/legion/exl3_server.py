"""Minimal OpenAI-compatible server over ExLlamaV3.

Exists so the ExLlamaV3 engine can be scored with the *same* harness as
llama.cpp (bench/aider_lite.py), instead of comparing a speed number from one
tool against a quality number from another. TabbyAPI would also work but pulls
in a large dependency set that conflicts with the pinned torch/exllamav3 build.

Implements only what the harness uses: POST /v1/chat/completions (non-streaming)
and GET /health. The model's own chat_template.jinja is applied so prompts match
what llama.cpp --jinja produces.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from exllamav3 import (Cache, CacheLayer_quant, ComboSampler, Config, Generator,
                       Job, Model, Tokenizer)
from jinja2 import Environment

STATE = {}
LOCK = threading.Lock()   # single GPU, single cache -> serialize requests


def build(model_dir: str, cache_tokens: int, kv_bits: int | None):
    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    # max_batch_size defaults to 16; on this hybrid arch every slot carries its
    # own DeltaNet recurrent state, which alone can exhaust a 16 GB card.
    kw = {"max_num_tokens": cache_tokens, "max_batch_size": 1}
    if kv_bits:
        cache = Cache(model, layer_type=CacheLayer_quant, k_bits=kv_bits, v_bits=kv_bits, **kw)
    else:
        cache = Cache(model, **kw)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer, max_batch_size=1)

    tpl_path = Path(model_dir) / "chat_template.jinja"
    tpl_src = tpl_path.read_text() if tpl_path.exists() else None
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(RuntimeError(m))
    template = env.from_string(tpl_src) if tpl_src else None
    return model, tokenizer, generator, template


def render(template, tokenizer, messages, kwargs):
    if template is None:  # crude fallback
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    return template.render(messages=messages, add_generation_prompt=True,
                           bos_token="", eos_token="", **kwargs)


def generate(prompt: str, max_tokens: int, temperature: float, top_p: float, top_k: int):
    tok, gen = STATE["tokenizer"], STATE["generator"]
    ids = tok.encode(prompt, encode_special_tokens=True)
    # ComboSampler defaults: top_p=1.0 (disabled), top_k=0 (disabled).
    sampler = ComboSampler(temperature=max(temperature, 1e-4),
                           top_p=top_p if top_p else 1.0,
                           top_k=top_k if top_k else 0)
    job = Job(input_ids=ids, max_new_tokens=max_tokens, sampler=sampler,
              stop_conditions=list(STATE["stop_ids"]))
    gen.enqueue(job)
    out = []
    while gen.num_remaining_jobs():
        for res in gen.iterate():
            if res.get("text"):
                out.append(res["text"])
    return "".join(out), int(ids.shape[-1])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages", [])
        tpl_kwargs = req.get("chat_template_kwargs") or {}
        prompt = render(STATE["template"], STATE["tokenizer"], msgs, tpl_kwargs)
        t0 = time.time()
        try:
            with LOCK:
                text, n_prompt = generate(
                    prompt,
                    int(req.get("max_tokens") or 512),
                    float(req.get("temperature", 0.6)),
                    float(req.get("top_p", 0.95)),
                    int(req.get("top_k", 20)),
                )
        except Exception as exc:  # surface errors to the harness rather than hanging
            self._send(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            return
        n_out = len(STATE["tokenizer"].encode(text)[0]) if text else 0
        self._send(200, {
            "id": "chatcmpl-exl3", "object": "chat.completion",
            "created": int(t0), "model": req.get("model", "exl3"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                      "total_tokens": n_prompt + n_out},
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--cache-tokens", type=int, default=32768)
    ap.add_argument("--kv-bits", type=int, default=0)
    args = ap.parse_args()

    model, tok, gen, tpl = build(args.model_dir, args.cache_tokens, args.kv_bits or None)
    stop = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        try:
            ids = tok.encode(s, encode_special_tokens=True)
            if ids.shape[-1] == 1:
                stop.add(int(ids[0, 0]))
        except Exception:
            pass
    STATE.update(model=model, tokenizer=tok, generator=gen, template=tpl, stop_ids=stop)

    free, total = torch.cuda.mem_get_info()
    print(f"loaded; VRAM {(total-free)/2**20:.0f} MiB of {total/2**20:.0f} MiB; "
          f"stop_ids={sorted(stop)}; serving on :{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
