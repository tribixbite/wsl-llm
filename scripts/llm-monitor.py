#!/usr/bin/env python3
"""Web dashboard for llama-server: live stats plus the images clients actually send.

Why a proxy: `/slots` reports token counts and busy state but NOT prompt content,
so there is no way to see an image from the server's own endpoints. Sitting
between client and server is the only way to capture payloads.

    # observe traffic you redirect through it (no server restart needed)
    ./llm-monitor.py --port 8090 --upstream http://127.0.0.1:8080
    #   -> point your client at http://<host>:8090/v1

    # transparent: move llama-server to 8081, run this on 8080, and every
    #   existing client is captured with no client-side change
    ./llm-monitor.py --port 8080 --upstream http://127.0.0.1:8081

Dashboard on http://<host>:<port>/ . Requests are proxied verbatim, including
streaming, so behaviour is unchanged for the client.
"""

import argparse
import base64
import json
import os
import re
import threading
import time
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://127.0.0.1:8080"
API_KEY = ""                       # injected upstream auth (vLLM etc.)
BACKEND = None                     # "llama.cpp" | "vllm" | None (auto-detected once)
SERVER_PARAMS = {}                 # shown on the dashboard (vLLM has no /props)
PORT = 8090
LAN_HOST = ""
MODEL_NAME = "unknown"
CTX_LEN = None
MAX_TOKENS_HINT = 4000
REQUESTS = deque(maxlen=500)      # newest last
STORE = os.path.expanduser("~/.local/share/llm-monitor/requests.jsonl")
STORE_MAX_BYTES = 64 * 1024 * 1024   # rotate past this
IMG_DIR = os.path.expanduser("~/.local/share/llm-monitor/images")
IMG_KEEP = 300                       # most-recent images kept on disk
MIME_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
            "image/webp": "webp", "image/gif": "gif"}
IMAGES = {}                        # id -> (mime, bytes)
LOCK = threading.Lock()
DATA_URI = re.compile(r"^data:([^;]+);base64,(.*)$", re.S)


def _img_save(key, mime, raw):
    """Write the image so the gallery survives a restart; prune oldest."""
    try:
        os.makedirs(IMG_DIR, exist_ok=True)
        ext = MIME_EXT.get(mime, "bin")
        with open(os.path.join(IMG_DIR, f"{key}.{ext}"), "wb") as f:
            f.write(raw)
        files = sorted(os.listdir(IMG_DIR))
        for old in files[:-IMG_KEEP]:
            try:
                os.remove(os.path.join(IMG_DIR, old))
            except OSError:
                pass
    except Exception:
        pass


def _img_load(key):
    """Serve an image from memory, else from disk (post-restart)."""
    with LOCK:
        item = IMAGES.get(key)
    if item:
        return item
    try:
        for fn in os.listdir(IMG_DIR):
            if fn.rsplit(".", 1)[0] == key:
                ext = fn.rsplit(".", 1)[-1]
                mime = next((m for m, e in MIME_EXT.items() if e == ext),
                            "application/octet-stream")
                with open(os.path.join(IMG_DIR, fn), "rb") as f:
                    return mime, f.read()
    except Exception:
        pass
    return None


def _store_load():
    """Re-read the tail of the JSONL log so history survives restarts/reboots."""
    try:
        with open(STORE) as f:
            lines = f.readlines()[-REQUESTS.maxlen:]
    except FileNotFoundError:
        return
    for ln in lines:
        try:
            REQUESTS.append(json.loads(ln))
        except Exception:
            pass
    print(f"restored {len(REQUESTS)} requests from {STORE}", flush=True)


def _store_append(rec):
    """Append one record. Images are referenced by id, never inlined."""
    try:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        if os.path.exists(STORE) and os.path.getsize(STORE) > STORE_MAX_BYTES:
            os.replace(STORE, STORE + ".1")
        with open(STORE, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _record(rec):
    with LOCK:
        REQUESTS.append(rec)
    _store_append(rec)


def _auth_headers(extra=None):
    """Headers for an upstream call: caller's auth if present, else our --api-key."""
    h = dict(extra or {})
    if API_KEY and not any(k.lower() == "authorization" for k in h):
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def _get_json(path, timeout=4):
    req = urllib.request.Request(f"{UPSTREAM}{path}", headers=_auth_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _detect_backend():
    """llama.cpp exposes /props; vLLM does not. Detect once, cache."""
    global BACKEND
    if BACKEND:
        return BACKEND
    try:
        _get_json("/props", timeout=3)
        BACKEND = "llama.cpp"
        return BACKEND
    except Exception:
        pass
    try:
        _get_json("/v1/models", timeout=3)
        BACKEND = "vllm"
    except Exception:
        BACKEND = None
    return BACKEND


def _vllm_metrics():
    """Scrape the few Prometheus counters worth showing (incl. spec-decode)."""
    out = {}
    try:
        req = urllib.request.Request(f"{UPSTREAM}/metrics", headers=_auth_headers())
        with urllib.request.urlopen(req, timeout=4) as r:
            body = r.read().decode("utf8", "replace")
    except Exception:
        return out
    want = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:gpu_cache_usage_perc": "kv_cache_used",
        "vllm:spec_decode_num_accepted_tokens_total": "spec_accepted",
        "vllm:spec_decode_num_draft_tokens_total": "spec_drafted",
    }
    for line in body.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, val = line.rpartition(" ")
        base = name.split("{")[0]
        if base in want:
            try:
                out[want[base]] = out.get(want[base], 0.0) + float(val)
            except ValueError:
                pass
    if out.get("spec_drafted"):
        out["spec_accept_rate"] = round(out["spec_accepted"] / out["spec_drafted"], 3)
    return out


def _guess_lan_ip():
    """Best-effort LAN address to advertise to other machines."""
    import socket
    try:
        s_ = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s_.connect(("8.8.8.8", 80))
        ip = s_.getsockname()[0]
        s_.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _server_cmdline():
    """The exact command line currently serving the upstream model."""
    try:
        import subprocess
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:
        return None
    port = UPSTREAM.rsplit(":", 1)[-1].strip("/")
    best = None
    for line in out.splitlines():
        low = line.lower()
        if "llm-monitor" in low or " ps -eo" in low:
            continue
        if not any(t in low for t in ("vllm", "llama-server", "ninfer", "sglang")):
            continue
        if port and port in line:
            return line.strip().split(None, 1)[-1]
        best = best or line.strip().split(None, 1)[-1]
    return best


def _connect_info():
    """Everything a client (or an agent) needs to talk to this endpoint."""
    be = _detect_backend()
    host = LAN_HOST or "127.0.0.1"
    base = f"http://{host}:{PORT}/v1"
    model, ctx = MODEL_NAME, CTX_LEN
    if be == "vllm":
        try:
            m = _get_json("/v1/models")["data"][0]
            model = m.get("id") or model
            ctx = m.get("max_model_len") or ctx
        except Exception:
            pass
    elif be == "llama.cpp":
        try:
            pr = _get_json("/props")
            ctx = (pr.get("default_generation_settings") or {}).get("n_ctx") or ctx
            model = os.path.basename(pr.get("model_path") or "") or model
        except Exception:
            pass
    samp = SERVER_PARAMS or {"temperature": "1.0", "top_p": "0.95", "top_k": "20"}
    max_tok = MAX_TOKENS_HINT
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": max_tok,
        **{k: (float(v) if "." in str(v) else int(v)) if str(v).replace(".", "", 1).isdigit()
           else v for k, v in samp.items()},
    }
    curl = ("curl " + base + "/chat/completions \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '" + json.dumps(body) + "'")
    py = ("from openai import OpenAI\n"
          f"c = OpenAI(base_url=\"{base}\", api_key=\"not-needed\")\n"
          f"r = c.chat.completions.create(model=\"{model}\",\n"
          "        messages=[{\"role\":\"user\",\"content\":\"hello\"}],\n"
          f"        max_tokens={max_tok}, temperature={samp.get('temperature','1.0')},"
          f" top_p={samp.get('top_p','0.95')})")
    return {
        "openai_base_url": base,
        "openai_base_url_local": f"http://127.0.0.1:{PORT}/v1",
        "upstream_direct": UPSTREAM + "/v1",
        "model": model,
        "api_key_required": False,
        "api_key": "not-needed",
        "context_length": ctx,
        "recommended_max_tokens": max_tok,
        "min_max_tokens": 1000,
        "max_tokens_note": ("Thinking is emitted before the answer and consumes the same "
                            "budget; small max_tokens returns empty content."),
        "sampling": samp,
        "backend": be,
        "server_command": _server_cmdline(),
        "endpoints": {
            "chat_completions": base + "/chat/completions",
            "completions": base + "/completions",
            "models": base + "/models",
            "dashboard": f"http://{host}:{PORT}/",
            "state_json": f"http://{host}:{PORT}/api/state",
            "connect_json": f"http://{host}:{PORT}/api/connect",
            "api_index": f"http://{host}:{PORT}/api",
        },
        "clients": {
            "anythingllm": {
                "LLM Provider": "Generic OpenAI",
                "Base URL": base,
                "API Key": "not-needed",
                "Chat Model Name": model,
                "Token context window": ctx,
                "Max Tokens": max_tok,
            },
            "open_webui": {"OPENAI_API_BASE_URL": base, "OPENAI_API_KEY": "not-needed"},
            "aider": f"OPENAI_API_BASE={base} OPENAI_API_KEY=x aider --model openai/{model}",
            "continue_dev": {"provider": "openai", "apiBase": base, "model": model,
                             "contextLength": ctx},
            "litellm": {"model_name": model,
                        "litellm_params": {"model": f"openai/{model}",
                                           "api_base": base, "api_key": "not-needed"}},
        },
        "examples": {"curl": curl, "python_openai_sdk": py},
    }


def _harvest(messages):
    """Pull plain text and any inline images out of an OpenAI messages array."""
    texts, imgs = [], []
    for m in messages or []:
        c = m.get("content")
        if isinstance(c, str):
            texts.append(f"{m.get('role','?')}: {c}")
            continue
        for part in c or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(f"{m.get('role','?')}: {part.get('text','')}")
            elif part.get("type") == "image_url" or "image_url" in part:
                url = (part.get("image_url") or {}).get("url", "")
                mo = DATA_URI.match(url)
                if mo:
                    try:
                        raw = base64.b64decode(mo.group(2))
                    except Exception:
                        continue
                    key = f"{int(time.time()*1000)}_{len(IMAGES)}"
                    with LOCK:
                        IMAGES[key] = (mo.group(1), raw)
                        for old in list(IMAGES)[:-60]:
                            IMAGES.pop(old, None)
                    _img_save(key, mo.group(1), raw)
                    imgs.append({"id": key, "mime": mo.group(1), "bytes": len(raw)})
                else:
                    imgs.append({"id": None, "mime": "remote-url", "url": url[:200]})
    return "\n".join(texts), imgs


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------- dashboard ----------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if path == "/api/state":
            return self._send(200, json.dumps(self._state()))
        if path in ("/api/connect", "/.well-known/llm-endpoint.json", "/connect.json"):
            return self._send(200, json.dumps(_connect_info(), indent=2))
        if path in ("/api", "/api/"):
            # discovery index: an agent hitting the root API learns every route
            info = _connect_info()
            return self._send(200, json.dumps({
                "service": "llm-monitor",
                "description": "OpenAI-compatible proxy + dashboard in front of a local LLM server",
                "openai_base_url": info["openai_base_url"],
                "model": info["model"],
                "context_length": info["context_length"],
                "recommended_max_tokens": info["recommended_max_tokens"],
                "routes": {
                    "GET /api": "this index",
                    "GET /api/connect": "full connection + client config (curl, python, AnythingLLM, ...)",
                    "GET /api/state": "live stats, recent requests with prompt/response",
                    "GET /img/<id>": "an image seen in a prompt",
                    "POST /v1/chat/completions": "proxied to upstream, captured",
                    "GET /v1/models": "proxied to upstream",
                }}, indent=2))
        if path.startswith("/img/"):
            key = path[5:]
            item = _img_load(key)
            if not item:
                return self._send(404, b"missing", "text/plain")
            mime, raw = item
            return self._send(200, raw, mime)
        # anything else: proxy through (e.g. /health, /props, /slots, /v1/models)
        return self._proxy_get(path)

    def _state(self):
        be = _detect_backend()
        out = {"upstream": UPSTREAM, "ok": False, "backend": be or "unknown"}

        if be == "llama.cpp":
            try:
                p = _get_json("/props")
                g = p.get("default_generation_settings", {})
                out.update(ok=True, n_ctx=g.get("n_ctx"), slots=p.get("total_slots"),
                           params={k: g.get("params", {}).get(k) for k in
                                   ("temperature", "top_p", "top_k", "min_p",
                                    "presence_penalty")})
            except Exception as e:
                out["error"] = str(e)
            try:
                s_ = _get_json("/slots")
                s_ = (s_ if isinstance(s_, list) else [s_])[0]
                out["slot"] = {"busy": s_.get("is_processing"), "task": s_.get("id_task"),
                               "prompt_tokens": s_.get("n_prompt_tokens"),
                               "processed": s_.get("n_prompt_tokens_processed"),
                               "speculative": s_.get("speculative")}
            except Exception:
                out["slot"] = None

        elif be == "vllm":
            try:
                m = _get_json("/v1/models")["data"][0]
                out.update(ok=True, n_ctx=m.get("max_model_len"), model_id=m.get("id"))
            except Exception as e:
                out["error"] = str(e)
            mx = _vllm_metrics()
            out["metrics"] = mx
            out["slots"] = None
            if mx:
                busy = (mx.get("running", 0) or 0) > 0
                out["slot"] = {"busy": busy,
                               "task": f"{int(mx.get('running',0))} running"
                                       f" / {int(mx.get('waiting',0))} queued",
                               "prompt_tokens": None, "processed": None,
                               "speculative": mx.get("spec_accept_rate")}
            else:
                out["slot"] = None
            # vLLM applies the model's generation_config.json as server defaults
            out["params"] = SERVER_PARAMS
        else:
            out["error"] = "upstream not reachable"
            out["slot"] = None

        with LOCK:
            out["requests"] = list(REQUESTS)[-40:][::-1]
        return out

    def _proxy_get(self, path):
        try:
            hdrs = _auth_headers({k: v for k, v in self.headers.items()
                                  if k.lower() in ("authorization",)})
            req = urllib.request.Request(f"{UPSTREAM}{path}", headers=hdrs)
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                ctype = r.headers.get("Content-Type", "application/json")
            return self._send(200, body, ctype)
        except Exception as e:
            return self._send(502, json.dumps({"error": str(e)}))

    # ---------------- proxy ----------------
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        rec = {"t": time.strftime("%H:%M:%S"), "path": self.path, "images": [],
               "prompt_chars": 0, "text": "", "usage": None, "decode_tps": None,
               "ttft_ms": None, "stream": False, "model": None,
               "reply": "", "reasoning": "", "reply_chars": 0}
        try:
            body = json.loads(raw or b"{}")
            rec["model"] = body.get("model")
            rec["stream"] = bool(body.get("stream"))
            text, imgs = _harvest(body.get("messages"))
            rec["text"] = text[-8000:]
            rec["prompt_chars"] = len(text)
            rec["images"] = imgs
        except Exception:
            body = None

        fwd = {k: v for k, v in self.headers.items()
               if k.lower() not in ("host", "content-length", "accept-encoding")}
        req = urllib.request.Request(f"{UPSTREAM}{self.path}", data=raw,
                                     headers=_auth_headers(fwd), method="POST")
        t0 = time.time()
        try:
            up = urllib.request.urlopen(req, timeout=3600)
        except Exception as e:
            rec["usage"] = {"error": str(e)}
            _record(rec)
            return self._send(502, json.dumps({"error": str(e)}))

        ctype = up.headers.get("Content-Type", "application/json")
        if rec["stream"]:
            # stream through untouched, but sniff timing + usage as it passes
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            ttft = None
            ntok = 0
            usage = None
            reply_parts, reason_parts = [], []
            try:
                for line in up:
                    self.wfile.write(line)
                    self.wfile.flush()
                    s = line.decode("utf8", "replace").strip()
                    if not s.startswith("data:"):
                        continue
                    payload = s[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        o = json.loads(payload)
                    except Exception:
                        continue
                    ch = o.get("choices") or []
                    if ch:
                        d_ = ch[0].get("delta") or {}
                        # Qwen3.8 streams thinking in `reasoning`/`reasoning_content`;
                        # counting only `content` badly undercounts decode t/s.
                        c_ = d_.get("content")
                        r_ = d_.get("reasoning") or d_.get("reasoning_content")
                        if c_ or r_:
                            if ttft is None:
                                ttft = time.time() - t0
                            ntok += 1
                        if c_:
                            reply_parts.append(c_)
                        if r_:
                            reason_parts.append(r_)
                    if o.get("usage"):
                        usage = o["usage"]
            except Exception:
                pass
            wall = time.time() - t0
            tok = (usage or {}).get("completion_tokens") or ntok
            rec["usage"] = usage or {"completion_tokens": ntok}
            rec["ttft_ms"] = round((ttft or 0) * 1000)
            rec["decode_tps"] = round(tok / max(wall - (ttft or 0), 1e-6), 1)
            full = "".join(reply_parts)
            rec["reply_chars"] = len(full)
            rec["reply"] = full[:8000]
            rec["reasoning"] = "".join(reason_parts)[:8000]
            _record(rec)
            return

        data = up.read()
        wall = time.time() - t0
        try:
            o = json.loads(data)
            rec["usage"] = o.get("usage")
            u_ = o.get("usage") or {}
            ct = u_.get("completion_tokens")
            if ct:
                rec["decode_tps"] = round(ct / max(wall, 1e-6), 1)
            rt = (u_.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if rt:
                rec["reasoning_tokens"] = rt
            ch = (o.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            full = msg.get("content") or ""
            rec["reply_chars"] = len(full)
            rec["reply"] = full[:8000]
            rec["reasoning"] = (msg.get("reasoning")
                                or msg.get("reasoning_content") or "")[:8000]
            rec["finish"] = ch.get("finish_reason")
        except Exception:
            pass
        _record(rec)
        self._send(200, data, ctype)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="dark light">
<title>LLM monitor</title><style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;
      --ok:#3fb950;--busy:#d29922;--bad:#f85149;--accent:#58a6ff}
*{box-sizing:border-box;-webkit-tap-highlight-color:rgba(88,166,255,.15)}
body{margin:0;background:var(--bg);color:var(--fg);
     font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
     padding-bottom:env(safe-area-inset-bottom)}
header{padding:10px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;
       align-items:center;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:14px;margin:0 6px 0 0;font-weight:600;white-space:nowrap}
.pill{padding:3px 9px;border-radius:99px;font-size:11.5px;border:1px solid var(--line);white-space:nowrap}
.ok{color:var(--ok);border-color:var(--ok)} .busy{color:var(--busy);border-color:var(--busy)}
.bad{color:var(--bad);border-color:var(--bad)}
main{padding:12px;display:grid;gap:12px;max-width:1400px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.card h2{font-size:11.5px;margin:0 0 10px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}
/* big TPS readout */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
.stat{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.stat .n{font-size:24px;font-weight:700;line-height:1.1}
.stat .l{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;margin-top:2px}
.stat.hi .n{color:var(--accent)}
.imgs{display:flex;gap:8px;flex-wrap:wrap}
.imgs a{display:block;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.imgs img{display:block;height:88px;width:auto}
/* request list: cards on mobile, table-ish on desktop */
.req{border:1px solid var(--line);border-radius:8px;margin-bottom:8px;overflow:hidden}
.req>summary{list-style:none;cursor:pointer;padding:10px 12px;display:flex;gap:10px;
             flex-wrap:wrap;align-items:center;min-height:44px}
.req>summary::-webkit-details-marker{display:none}
.req>summary:active{background:#1c2129}
.req .when{color:var(--dim);font-size:12px}
.req .tps{color:var(--accent);font-weight:700}
.req .meta{color:var(--dim);font-size:12px;margin-left:auto;text-align:right}
.body{padding:0 12px 12px}
.lbl{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em;
     margin:10px 0 4px;display:flex;gap:8px;align-items:center}
.blk{white-space:pre-wrap;word-break:break-word;background:#0d1117;border:1px solid var(--line);
     border-radius:6px;padding:9px;font-size:12.5px;max-height:320px;overflow:auto;
     -webkit-overflow-scrolling:touch}
.blk.reason{color:var(--dim)}
.none{color:var(--dim);font-style:italic}
.btn{background:#21262d;border:1px solid var(--line);color:var(--fg);border-radius:6px;
     padding:6px 10px;font:inherit;font-size:12px;cursor:pointer;min-height:34px}
.btn:active{background:#2d333b}
@media(max-width:640px){
  main{padding:8px;gap:8px} .card{padding:10px} h1{font-size:13px}
  .stat .n{font-size:20px} .imgs img{height:66px}
  .req .meta{margin-left:0;width:100%;text-align:left}
  .blk{max-height:240px;font-size:12px}
}
</style></head><body>
<header>
  <h1>LLM monitor</h1>
  <span id="up" class="pill">…</span>
  <span id="be" class="pill"></span>
  <span id="slot" class="pill"></span>
  <span id="ctx" class="pill"></span>
  <span id="spec" class="pill"></span>
</header>
<main>
  <div class="card"><h2>throughput</h2><div id="stats" class="stats"></div></div>
  <details class="card" id="conncard"><summary style="cursor:pointer;list-style:none;
      font-size:11.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em">
      connect / client settings &nbsp;<span style="color:var(--accent)">tap to expand</span></summary>
    <div id="conn"></div></details>
  <div class="card"><h2>server</h2><div id="params" class="stats"></div></div>
  <div class="card"><h2>images seen in prompts</h2><div id="gallery" class="imgs"></div></div>
  <div class="card"><h2>requests <span id="cnt" style="color:var(--dim)"></span></h2>
    <div id="rows"></div></div>
</main>
<script>
const $=id=>document.getElementById(id);
const esc=t=>(t||'').replace(/[<&>]/g,c=>({'<':'&lt;','&':'&amp;','>':'&gt;'}[c]));
const open=new Set();                    // keep expanded rows open across refreshes
let paused=false;

function statCard(n,l,hi){return `<div class="stat${hi?' hi':''}"><div class="n">${n}</div><div class="l">${l}</div></div>`;}

async function tick(){
  if(paused) return;
  let d; try{ d=await (await fetch('/api/state')).json(); }catch(e){ return; }
  $('up').textContent=d.ok?'upstream up':'upstream DOWN';
  $('up').className='pill '+(d.ok?'ok':'bad');
  $('be').textContent=(d.backend||'')+(d.model_id?(' · '+d.model_id):'');
  const s=d.slot;
  $('slot').textContent=s?(s.busy?('BUSY '+(s.task||'')):'idle'):'';
  $('slot').className='pill '+(s&&s.busy?'busy':'ok');
  $('ctx').textContent=d.n_ctx?('ctx '+d.n_ctx.toLocaleString()):'';
  const mx=d.metrics||{};
  $('spec').textContent=(mx.spec_accept_rate!=null)?('MTP accept '+(mx.spec_accept_rate*100).toFixed(1)+'%'):'';

  const rs=d.requests||[];
  const tps=rs.map(r=>r.decode_tps).filter(v=>typeof v==='number'&&v>0);
  const last=tps.length?tps[0]:null;
  const avg=tps.length?(tps.reduce((a,b)=>a+b,0)/tps.length):null;
  const best=tps.length?Math.max(...tps):null;
  $('stats').innerHTML=
     statCard(last!=null?last.toFixed(1):'—','last decode t/s',true)
    +statCard(avg!=null?avg.toFixed(1):'—','avg t/s (recent)')
    +statCard(best!=null?best.toFixed(1):'—','best t/s')
    +statCard(rs.length,'requests logged');

  const p=d.params||{};
  $('params').innerHTML=Object.keys(p).length
    ? Object.entries(p).map(([k,v])=>statCard(v,k)).join('')
    : '<span class="none">unavailable</span>';

  const imgs=[]; rs.forEach(r=>(r.images||[]).forEach(i=>{if(i.id)imgs.push(i);}));
  $('gallery').innerHTML=imgs.length
    ? imgs.slice(0,24).map(i=>`<a href="/img/${i.id}" target="_blank"><img loading="lazy" src="/img/${i.id}"></a>`).join('')
    : '<span class="none">no images yet — send an image_url part</span>';

  $('cnt').textContent='('+rs.length+')';
  $('rows').innerHTML = rs.length ? rs.map((r,i)=>{
    const u=r.usage||{}, k=r.t+'|'+i;
    const rt=(u.completion_tokens_details&&u.completion_tokens_details.reasoning_tokens)||r.reasoning_tokens||0;
    return `<details class="req" data-k="${esc(k)}"${open.has(k)?' open':''}>
      <summary>
        <span class="when">${r.t}</span>
        <span class="tps">${r.decode_tps!=null?r.decode_tps+' t/s':''}</span>
        <span class="meta">${u.prompt_tokens||0} in · ${u.completion_tokens||0} out${rt?(' · '+rt+' think'):''}${r.ttft_ms?(' · ttft '+r.ttft_ms+'ms'):''}${(r.images||[]).length?(' · '+r.images.length+' img'):''}</span>
      </summary>
      <div class="body">
        ${(r.images||[]).filter(i=>i.id).length?`<div class="lbl">images</div><div class="imgs">${
          r.images.filter(i=>i.id).map(i=>`<a href="/img/${i.id}" target="_blank"><img loading="lazy" src="/img/${i.id}"></a>`).join('')}</div>`:''}
        <div class="lbl">prompt${r.prompt_chars?(' · '+r.prompt_chars+' chars'):''}
          <button class="btn" data-copy="p">copy</button></div>
        <div class="blk" data-p>${esc(r.text)||'<span class="none">—</span>'}</div>
        ${r.reasoning?`<div class="lbl">reasoning</div><div class="blk reason">${esc(r.reasoning)}</div>`:''}
        <div class="lbl">response${r.reply_chars?(' · '+r.reply_chars+' chars'):''}
          <button class="btn" data-copy="r">copy</button></div>
        <div class="blk" data-r>${esc(r.reply)||'<span class="none">—</span>'}</div>
      </div></details>`;
  }).join('') : '<span class="none">no requests captured yet</span>';
}

// keep open/closed state; pause polling while a row is open so it doesn't jump
document.addEventListener('toggle',e=>{
  const d=e.target; if(!d.classList||!d.classList.contains('req'))return;
  const k=d.dataset.k; d.open?open.add(k):open.delete(k);
  paused=[...document.querySelectorAll('details.req')].some(x=>x.open);
},true);
document.addEventListener('click',e=>{
  const b=e.target.closest('button[data-copy]'); if(!b)return;
  e.preventDefault();
  const box=b.closest('.body').querySelector(b.dataset.copy==='p'?'[data-p]':'[data-r]');
  navigator.clipboard&&navigator.clipboard.writeText(box.innerText);
  b.textContent='copied'; setTimeout(()=>b.textContent='copy',1200);
});
async function conn(){
  let d; try{ d=await (await fetch('/api/connect')).json(); }catch(e){ return; }
  const row=(k,v)=>`<div class="lbl" style="margin:8px 0 3px">${k}
      <button class="btn" data-val="${esc(String(v))}">copy</button></div>
      <div class="blk" style="max-height:none">${esc(String(v))}</div>`;
  const a=d.clients.anythingllm;
  $('conn').innerHTML =
     row('OpenAI Base URL (LAN)', d.openai_base_url)
    +row('Model name', d.model)
    +row('API key', d.api_key + '  (any non-empty string works)')
    +row('Context length', d.context_length)
    +row('max_tokens (recommended)', d.recommended_max_tokens + '   — minimum ' + d.min_max_tokens + '. ' + d.max_tokens_note)
    +row('Sampling', Object.entries(d.sampling).map(([k,v])=>k+'='+v).join('  '))
    +`<div class="lbl" style="margin:14px 0 3px">AnythingLLM / Generic-OpenAI client</div>
      <div class="blk" style="max-height:none">${esc(Object.entries(a).map(([k,v])=>k+': '+v).join('\n'))}</div>`
    +row('aider', d.clients.aider)
    +row('curl', d.examples.curl)
    +row('python (openai sdk)', d.examples.python_openai_sdk)
    +row('server command now running', d.server_command||'(not detected)')
    +`<div class="lbl" style="margin:14px 0 3px">machine-readable</div>
      <div class="blk" style="max-height:none">GET ${esc(d.endpoints.connect_json)}\nGET ${esc(d.endpoints.api_index)}</div>`;
}
document.addEventListener('click',e=>{
  const b=e.target.closest('button[data-val]'); if(!b)return;
  e.preventDefault(); navigator.clipboard&&navigator.clipboard.writeText(b.dataset.val);
  b.textContent='copied'; setTimeout(()=>b.textContent='copy',1200);
});
conn(); tick(); setInterval(tick,2000); setInterval(conn,30000);
</script></body></html>"""


def main():
    global UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--upstream", default="http://127.0.0.1:8080")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--api-key", default="",
                    help="upstream API key, injected when the client sends none")
    ap.add_argument("--lan-host", default="",
                    help="hostname/IP to advertise in /api/connect (default: autodetect)")
    ap.add_argument("--max-tokens-hint", type=int, default=4000,
                    help="max_tokens recommended to clients in /api/connect")
    ap.add_argument("--params", default="",
                    help='sampling defaults to display, e.g. "temperature=1.0,top_p=0.95"')
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    global API_KEY, SERVER_PARAMS, PORT, LAN_HOST, MAX_TOKENS_HINT
    PORT = args.port
    MAX_TOKENS_HINT = args.max_tokens_hint
    LAN_HOST = args.lan_host or _guess_lan_ip()
    API_KEY = args.api_key
    if args.params:
        for kv in args.params.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                SERVER_PARAMS[k.strip()] = v.strip()
    _store_load()
    print(f"monitor  http://{args.bind}:{args.port}/   ->  upstream {UPSTREAM}", flush=True)
    print(f"persisting to {STORE}", flush=True)
    print(f"connect info  http://{LAN_HOST}:{args.port}/api/connect", flush=True)
    print(f"point clients at http://<host>:{args.port}/v1 to capture their traffic", flush=True)
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
