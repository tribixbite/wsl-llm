#!/usr/bin/env bash
# KL-divergence of each Q3_K_XL revision against a Q8_0 reference.
#
# Why this rather than more aider runs: pass@1/pass@2 on 34 exercises has a huge
# run-to-run spread on this model (one config scored 14.7-29.4% across three
# runs), so it cannot resolve a quantization difference. KLD is deterministic
# and compares the full next-token distribution, position by position.
#
# Reference is Q8_0, not BF16: BF16 is 50.9 GiB and will not fit. Q8_0's own
# KLD to FP16 is ~0.0014, i.e. a few percent of what we are measuring.
#
# Q8_0 is 27.05 GiB and does NOT fit in 16 GB, so the baseline pass runs
# partially on CPU (-ngl BASE_NGL). That only affects speed, not the logits.
#
# Disk: the logits file is n_chunk * (n_ctx/2-1) * (n_vocab+4) * 2 bytes.
# With this model's 248,320-token vocab that is ~11.8 GiB at --chunks 100 and
# ~23.6 GiB at 200, so --chunks is not optional here.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LC="${LC:-$HOME/llama.cpp}"
PPL="$LC/build/bin/llama-perplexity"
DATA="${DATA:-$LC/wikitext-2-raw/wiki.test.raw}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/bench/results/legion-qwen38/kld}"
WORK="${WORK:-$HOME/models/kld}"
CHUNKS="${CHUNKS:-100}"
CTX="${CTX:-512}"
BASE_NGL="${BASE_NGL:-30}"     # Q8_0 partial offload; the rest runs on CPU
BASE_MODEL="${BASE_MODEL:-$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf}"

mkdir -p "$OUT_DIR" "$WORK"
KLD_BASE="$WORK/base-q8.kld"

run_pass() {  # run_pass <tag> <model> <extra args...>
  local tag="$1" model="$2"; shift 2
  echo "=== $tag ==="
  "$PPL" -m "$model" -f "$DATA" -c "$CTX" --chunks "$CHUNKS" -fa on --seed 1337 \
      "$@" 2>&1 | tee "$OUT_DIR/${tag}.log" | tail -25
}

if [[ ! -s "$KLD_BASE" ]]; then
  echo ">>> building Q8_0 reference logits (partial CPU offload, this is the slow part)"
  run_pass base_q8 "$BASE_MODEL" -ngl "$BASE_NGL" --kl-divergence-base "$KLD_BASE"
  ls -la "$KLD_BASE"
else
  echo ">>> reusing existing reference logits: $KLD_BASE"
fi

for entry in \
  "v3|$HOME/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf" \
  "r408|$HOME/models/Qwen3.8-27B-GGUF-408fcc18/Qwen3.8-27B-UD-Q3_K_XL.gguf" ; do
  tag="${entry%%|*}"; model="${entry#*|}"
  [[ -f "$model" ]] || { echo "skip $tag (missing $model)"; continue; }
  run_pass "kld_$tag" "$model" -ngl 99 --kl-divergence-base "$KLD_BASE" --kl-divergence
done

echo "=== summary ==="
for f in "$OUT_DIR"/kld_*.log; do
  [[ -f "$f" ]] || continue
  echo "--- $(basename "$f" .log)"
  grep -iE "Mean KLD|Maximum KLD|99\.9%|99\.0%|Median KLD|Same top|RMS |Mean Delta|PPL" "$f" | head -12
done
