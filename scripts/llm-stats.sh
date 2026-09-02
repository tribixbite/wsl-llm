#!/usr/bin/env bash
# Read-only view of what llama-server has been doing. Sends NO inference and
# never consumes the single slot — safe to run while a job is in flight.
#
#   ./llm-stats.sh          one-shot summary
#   ./llm-stats.sh -w       watch, refreshing every 5 s
#   ./llm-stats.sh -n 40    show the last 40 completed requests
#
# Sources, cheapest first:
#   GET /slots   live slot state (busy? prompt size? cache hits?) — a plain GET
#   the log      per-request timings: prefill t/s, decode t/s, token counts
#   nvidia-smi   VRAM / clocks / power / throttle reasons
#
# The log is UTF-16LE (PowerShell's `*>>` writes it that way), so it must go
# through iconv before grep sees anything.
set -uo pipefail

PORT="${PORT:-8080}"
LOGDIR="${LOGDIR:-/mnt/c/llm/logs}"
N=20
WATCH=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--watch) WATCH=1; shift ;;
    -n) N="$2"; shift 2 ;;
    *) echo "usage: $0 [-w] [-n N]" >&2; exit 2 ;;
  esac
done

# Whichever log file was written most recently is the live server's.
live_log() { ls -t "$LOGDIR"/*.log 2>/dev/null | head -1; }
decode() { iconv -f UTF-16LE -t UTF-8 "$1" 2>/dev/null || tr -d '\000' < "$1"; }

report() {
  local LOG; LOG="$(live_log)"

  echo "=== endpoint ==="
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    curl -s -m 5 "http://127.0.0.1:$PORT/props" 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin); g=d['default_generation_settings']
print(f\"  up | n_ctx={g['n_ctx']} | slots={d['total_slots']}\")" 2>/dev/null
  else
    echo "  DOWN"; return
  fi

  echo "=== live slot (is it busy right now?) ==="
  curl -s -m 5 "http://127.0.0.1:$PORT/slots" 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('  (slots endpoint disabled)'); raise SystemExit
if isinstance(d,dict) and 'error' in d: print('  (disabled)'); raise SystemExit
for s in (d if isinstance(d,list) else [d]):
    busy = s.get('is_processing')
    print(f\"  slot {s.get('id')}: {'BUSY' if busy else 'idle'} | task={s.get('id_task')} \"
          f\"| prompt_tokens={s.get('n_prompt_tokens')} cached={s.get('n_prompt_tokens_cache')} \"
          f\"| speculative={s.get('speculative')}\")" 2>/dev/null

  echo "=== last $N completed requests (from $(basename "${LOG:-none}")) ==="
  [[ -n "$LOG" ]] || { echo "  no log found"; return; }
  decode "$LOG" | python3 -c "
import re,sys,statistics as st
# Two log formats exist depending on build/verbosity:
#  (a) classic:  prompt eval time = .. / N tokens ( .. , X tokens per second)
#                       eval time = .. / N tokens ( .. , X tokens per second)
#  (b) progressive: print_timing .. prompt processing, n_tokens = N, .. / X tokens per second
#                   print_timing .. n_gen = N, tg = X t/s
#                   release: .. stop processing: n_tokens = N
pre_a=re.compile(r'prompt eval time =\s*[\d.]+ ms /\s*(\d+) tokens.*?([\d.]+) tokens per second')
gen_a=re.compile(r'\|\s+eval time =\s*[\d.]+ ms /\s*(\d+) tokens.*?([\d.]+) tokens per second')
pre_b=re.compile(r'task (\d+) \| prompt processing, n_tokens =\s*(\d+).*?([\d.]+) tokens per second')
gen_b=re.compile(r'task (\d+) \| n_gen =\s*(\d+), tg =\s*([\d.]+) t/s')
rel_b=re.compile(r'task (\d+) \| stop processing: n_tokens =\s*(\d+)')
rows=[]; cur={}; tasks={}
for line in sys.stdin:
    m=pre_a.search(line)
    if m: cur={'p_tok':int(m.group(1)),'p_tps':float(m.group(2))}; continue
    m=gen_a.search(line)
    if m and cur:
        cur.update({'g_tok':int(m.group(1)),'g_tps':float(m.group(2))}); rows.append(cur); cur={}; continue
    m=pre_b.search(line)
    if m: tasks.setdefault(m.group(1),{}).update({'p_tok':int(m.group(2)),'p_tps':float(m.group(3))}); continue
    m=gen_b.search(line)
    if m: tasks.setdefault(m.group(1),{}).update({'g_tok':int(m.group(2)),'g_tps':float(m.group(3))}); continue
    m=rel_b.search(line)
    if m:
        t=tasks.pop(m.group(1),None)
        if t and 'g_tok' in t: t['total']=int(m.group(2)); rows.append(t)
rows=rows[-$N:]
if not rows: print('  (no completed requests logged yet)'); raise SystemExit
print(f\"  {'prompt':>8} {'prefill t/s':>12} {'gen':>7} {'decode t/s':>11}\")
for r in rows:
    print(f\"  {r.get('p_tok',0):>8,} {r.get('p_tps',0):>12.1f} {r.get('g_tok',0):>7,} {r.get('g_tps',0):>11.1f}\")
g=[r['g_tps'] for r in rows if r.get('g_tps')]
p=[r['p_tps'] for r in rows if r.get('p_tps')]
print(f\"  ---- {len(rows)} reqs | median decode {st.median(g) if g else 0:.1f} t/s\"
      f\" | median prefill {st.median(p) if p else 0:.0f} t/s\"
      f\" | {sum(r.get('g_tok',0) for r in rows):,} tokens generated\")" 2>/dev/null

  echo "=== gpu ==="
  nvidia-smi --query-gpu=memory.used,memory.total,clocks.sm,power.draw,enforced.power.limit,temperature.gpu,clocks_event_reasons.active \
      --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  echo "  (clocks_event_reasons 0x4 = SW power cap, 0x0 = unthrottled)"
}

if [[ $WATCH -eq 1 ]]; then
  while true; do clear; date '+%H:%M:%S'; report; sleep 5; done
else
  report
fi
