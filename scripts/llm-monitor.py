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
REQUESTS = deque(maxlen=200)      # newest last
IMAGES = {}                        # id -> (mime, bytes)
LOCK = threading.Lock()
DATA_URI = re.compile(r"^data:([^;]+);base64,(.*)$", re.S)


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
                        # bound memory: keep the most recent 40 images
                        for old in list(IMAGES)[:-40]:
                            IMAGES.pop(old, None)
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
        if path.startswith("/img/"):
            key = path[5:]
            with LOCK:
                item = IMAGES.get(key)
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
               "ttft_ms": None, "stream": False, "model": None}
        try:
            body = json.loads(raw or b"{}")
            rec["model"] = body.get("model")
            rec["stream"] = bool(body.get("stream"))
            text, imgs = _harvest(body.get("messages"))
            rec["text"] = text[-1200:]
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
            with LOCK:
                REQUESTS.append(rec)
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
                        if d_.get("content") or d_.get("reasoning") \
                                or d_.get("reasoning_content"):
                            if ttft is None:
                                ttft = time.time() - t0
                            ntok += 1
                    if o.get("usage"):
                        usage = o["usage"]
            except Exception:
                pass
            wall = time.time() - t0
            tok = (usage or {}).get("completion_tokens") or ntok
            rec["usage"] = usage or {"completion_tokens": ntok}
            rec["ttft_ms"] = round((ttft or 0) * 1000)
            rec["decode_tps"] = round(tok / max(wall - (ttft or 0), 1e-6), 1)
            with LOCK:
                REQUESTS.append(rec)
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
        except Exception:
            pass
        with LOCK:
            REQUESTS.append(rec)
        self._send(200, data, ctype)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>llama-server monitor</title><style>
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;
      --ok:#3fb950;--busy:#d29922;--bad:#f85149;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:16px;
       align-items:center;flex-wrap:wrap;position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:15px;margin:0;font-weight:600}
.pill{padding:2px 9px;border-radius:99px;font-size:12px;border:1px solid var(--line)}
.ok{color:var(--ok);border-color:var(--ok)} .busy{color:var(--busy);border-color:var(--busy)}
.bad{color:var(--bad);border-color:var(--bad)}
main{padding:16px;display:grid;gap:16px;grid-template-columns:1fr;max-width:1400px;margin:0 auto}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card h2{font-size:12px;margin:0 0 10px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:right;color:var(--dim);font-weight:500;padding:4px 8px;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left}
td{padding:5px 8px;border-bottom:1px solid #21262d;text-align:right;vertical-align:top}
td.l{text-align:left}
.imgs{display:flex;gap:8px;flex-wrap:wrap}
.imgs a{display:block;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.imgs img{display:block;height:90px;width:auto}
.txt{color:var(--dim);white-space:pre-wrap;word-break:break-word;max-height:78px;overflow:auto;
     font-size:11.5px;margin-top:4px}
.kv{display:flex;gap:20px;flex-wrap:wrap;color:var(--dim);font-size:12.5px}
.kv b{color:var(--fg);font-weight:600}
.none{color:var(--dim);font-style:italic}
@media(max-width:700px){.imgs img{height:64px}}
</style></head><body>
<header>
  <h1>llama-server monitor</h1>
  <span id="up" class="pill">…</span>
  <span id="slot" class="pill">…</span>
  <span id="ctx" class="pill"></span>
  <span id="be" class="pill"></span>
  <span id="spec" class="pill"></span>
  <span style="margin-left:auto;color:var(--dim);font-size:12px">refresh 2s · read-only</span>
</header>
<main>
  <div class="card"><h2>server</h2><div id="params" class="kv"></div></div>
  <div class="card"><h2>images seen in prompts</h2><div id="gallery" class="imgs"></div></div>
  <div class="card"><h2>recent requests</h2>
    <table><thead><tr><th>time</th><th>model</th><th>prompt tok</th><th>gen tok</th>
      <th>reason tok</th><th>decode t/s</th><th>ttft ms</th><th>img</th></tr></thead>
      <tbody id="rows"></tbody></table>
    <div id="detail"></div>
  </div>
</main>
<script>
const $=id=>document.getElementById(id);
async function tick(){
  let d; try{ d=await (await fetch('/api/state')).json(); }catch(e){ return; }
  $('up').textContent = d.ok ? 'upstream up' : 'upstream DOWN';
  $('up').className = 'pill ' + (d.ok?'ok':'bad');
  const s=d.slot;
  $('slot').textContent = s ? (s.busy?('BUSY task '+s.task):'idle') : 'slots n/a';
  $('slot').className = 'pill ' + (s && s.busy ? 'busy':'ok');
  $('ctx').textContent = d.n_ctx ? ('ctx '+d.n_ctx.toLocaleString()+(d.slots?(' · slots '+d.slots):'')) : '';
  $('be').textContent = (d.backend||'') + (d.model_id?(' · '+d.model_id):'');
  const mx = d.metrics||{};
  $('spec').textContent = (mx.spec_accept_rate!=null)
      ? ('spec accept '+(mx.spec_accept_rate*100).toFixed(1)+'%')
      : (mx.kv_cache_used!=null ? ('kv '+(mx.kv_cache_used*100).toFixed(0)+'%') : '');
  const p=d.params||{};
  $('params').innerHTML = Object.keys(p).length
    ? Object.entries(p).map(([k,v])=>`${k} <b>${v}</b>`).join('')
    : '<span class="none">unavailable</span>';

  const imgs=[];
  (d.requests||[]).forEach(r=>(r.images||[]).forEach(i=>{ if(i.id) imgs.push(i); }));
  $('gallery').innerHTML = imgs.length
    ? imgs.slice(0,24).map(i=>`<a href="/img/${i.id}" target="_blank" title="${i.mime} · ${(i.bytes/1024).toFixed(0)} KB">
         <img src="/img/${i.id}"></a>`).join('')
    : '<span class="none">no images sent yet — send a request with an image_url part</span>';

  $('rows').innerHTML = (d.requests||[]).map(r=>{
    const u=r.usage||{};
    return `<tr><td class="l">${r.t}</td><td class="l">${r.model||''}</td>
      <td>${(u.prompt_tokens||'')}</td><td>${(u.completion_tokens||'')}</td>
      <td>${(u.completion_tokens_details&&u.completion_tokens_details.reasoning_tokens)||r.reasoning_tokens||''}</td>
      <td>${r.decode_tps??''}</td><td>${r.ttft_ms??''}</td>
      <td>${(r.images||[]).length||''}</td></tr>
      ${r.text?`<tr><td colspan="8" class="l"><div class="txt">${
        r.text.replace(/[<&]/g,c=>({'<':'&lt;','&':'&amp;'}[c]))}</div></td></tr>`:''}`;
  }).join('') || '<tr><td colspan="8" class="none">no requests captured yet</td></tr>';
}
tick(); setInterval(tick,2000);
</script></body></html>"""


def main():
    global UPSTREAM
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--upstream", default="http://127.0.0.1:8080")
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--api-key", default="",
                    help="upstream API key, injected when the client sends none")
    ap.add_argument("--params", default="",
                    help='sampling defaults to display, e.g. "temperature=1.0,top_p=0.95"')
    args = ap.parse_args()
    UPSTREAM = args.upstream.rstrip("/")
    global API_KEY, SERVER_PARAMS
    API_KEY = args.api_key
    if args.params:
        for kv in args.params.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                SERVER_PARAMS[k.strip()] = v.strip()
    print(f"monitor  http://{args.bind}:{args.port}/   ->  upstream {UPSTREAM}", flush=True)
    print(f"point clients at http://<host>:{args.port}/v1 to capture their traffic", flush=True)
    ThreadingHTTPServer((args.bind, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
