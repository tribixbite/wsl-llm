#!/usr/bin/env bash
# Find the largest context that stays RESIDENT in VRAM on this 16 GB laptop card.
#
# Why this exists: on WSL2/WDDM, exceeding VRAM does NOT produce an OOM error.
# Windows' video memory manager silently evicts the model to system RAM and the
# server keeps answering /health while decoding at ~0.04 t/s (a ~700x collapse),
# then usually dies. Observed VRAM trace at -c 32768 -ctk q8_0: 13376 -> 15941
# -> 2910 MiB, i.e. demoted.
#
# So we cannot trust "it loaded". Every config must be validated by an actual
# timed generation: if decode t/s craters, the config spilled and is unusable.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf}"
PORT="${PORT:-8080}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/bench/results/legion-qwen38}"
TSV="$OUT_DIR/context_fit.tsv"
# Decode below this means the model got evicted to system RAM.
SPILL_THRESHOLD_TPS="${SPILL_THRESHOLD_TPS:-8}"

mkdir -p "$OUT_DIR"
[[ -f "$TSV" ]] || printf 'ctx\tkv\tubatch\tpeak_vram_mib\tdecode_tps\tpwr_limit_w\tverdict\n' > "$TSV"

stop_server() {
  pkill -9 -f "$LLAMA_BIN" 2>/dev/null
  for _ in $(seq 1 30); do
    pgrep -f "$LLAMA_BIN" >/dev/null || break
    sleep 0.5
  done
  sleep 3
}
trap stop_server EXIT

# ctx kv ubatch
CASES=(
  "8192 q8_0 512"
  "16384 q8_0 512"
  "24576 q8_0 512"
  "32768 q8_0 512"
  "32768 q8_0 256"
  "32768 q4_0 256"
  "49152 q4_0 256"
  "65536 q4_0 256"
)

for case in "${CASES[@]}"; do
  read -r ctx kv ub <<< "$case"
  tag="c${ctx}_${kv}_ub${ub}"
  echo "================ $tag ================"
  stop_server

  log="$OUT_DIR/fit_${tag}.log"
  setsid "$LLAMA_BIN" -m "$MODEL" -ngl 99 -fa on \
      -c "$ctx" -ctk "$kv" -ctv "$kv" -ub "$ub" -b 2048 --parallel 1 \
      --host 127.0.0.1 --port "$PORT" --jinja --no-webui \
      > "$log" 2>&1 < /dev/null &

  up=0
  for _ in $(seq 1 120); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { up=1; break; }
    sleep 2
  done
  if [[ $up -ne 1 ]]; then
    echo "  LOAD FAILED"
    printf '%s\t%s\t%s\t\t\t\tLOAD_FAILED\n' "$ctx" "$kv" "$ub" >> "$TSV"
    continue
  fi

  pwr_csv="$OUT_DIR/fit_${tag}.power.csv"
  nvidia-smi --query-gpu=memory.used,enforced.power.limit --format=csv,noheader -l 1 > "$pwr_csv" 2>&1 &
  pwr_pid=$!

  # Timed generation. enable_thinking:false keeps this short and comparable.
  read -r tps < <(python3 - "$PORT" <<'PY'
import json,sys,time,urllib.request
port=sys.argv[1]
body={"model":"m","messages":[{"role":"user","content":"Write a Python function that reverses a linked list. Code only."}],
      "max_tokens":128,"temperature":0.7,"top_p":0.8,"top_k":20,"stream":True,
      "stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}
req=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
t0=time.time(); ttft=None; n=0; comp=None
try:
    with urllib.request.urlopen(req,timeout=600) as r:
        for raw in r:
            l=raw.decode("utf8","replace").strip()
            if not l.startswith("data:"): continue
            p=l[5:].strip()
            if p=="[DONE]": break
            try: o=json.loads(p)
            except Exception: continue
            ch=o.get("choices") or []
            if ch and ch[0].get("delta",{}).get("content"):
                if ttft is None: ttft=time.time()-t0
                n+=1
            if o.get("usage"): comp=o["usage"].get("completion_tokens")
    tok=comp or n
    dw=max(time.time()-t0-(ttft or 0),1e-6)
    print(f"{tok/dw:.2f}")
except Exception as e:
    print("0.00")
PY
)

  kill "$pwr_pid" 2>/dev/null; wait "$pwr_pid" 2>/dev/null
  read -r peak_vram pwr_lim < <(awk -F', ' '{gsub(/ W|MiB/,"");
      if($1+0>m)m=$1+0; l=$2+0} END{printf "%d %.1f", m, l}' "$pwr_csv")

  verdict=OK
  awk "BEGIN{exit !($tps < $SPILL_THRESHOLD_TPS)}" && verdict=SPILLED_TO_RAM
  echo "  ctx=$ctx kv=$kv ub=$ub peak_vram=${peak_vram}MiB decode=${tps}t/s limit=${pwr_lim}W -> $verdict"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ctx" "$kv" "$ub" "$peak_vram" "$tps" "$pwr_lim" "$verdict" >> "$TSV"
  sleep 20
done

stop_server
echo "=== results ==="
column -t -s $'\t' "$TSV"
