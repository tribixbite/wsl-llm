#!/usr/bin/env bash
# Resilient downloader for the Qwen3.8-27B GGUF assets.
#
# WSL2 + HuggingFace's Xet CAS backend is flaky here: transient DNS failures
# ("Temporary failure in name resolution") kill the transfer mid-file and the
# hf CLI does not retry across them. We disable Xet (falling back to plain
# HTTPS range requests, which resume cleanly) and wrap the whole thing in a
# retry loop.
set -uo pipefail

REPO="${REPO:-unsloth/Qwen3.8-27B-GGUF}"
DEST="${DEST:-$HOME/models/Qwen3.8-27B-GGUF}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-40}"

# Files to fetch; override by passing them as args.
if [[ $# -gt 0 ]]; then
  FILES=("$@")
else
  FILES=(
    "Qwen3.8-27B-UD-Q3_K_XL.gguf"     # 12.24 GiB — the requested daily-driver quant
    "MTP/mtp-Qwen3.8-27B-Q4_0.gguf"   #  1.28 GiB — MTP head for speculative decoding
  )
fi

# Force the classic HTTP downloader; Xet is the component that keeps dying.
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=60

mkdir -p "$DEST"

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== attempt ${attempt}/${MAX_ATTEMPTS} at $(date -Is) ==="
  # shellcheck disable=SC2046
  if hf download "$REPO" \
       $(printf -- '--include %q ' "${FILES[@]}") \
       --local-dir "$DEST"; then
    echo "=== download complete at $(date -Is) ==="
    du -sh "$DEST"
    exit 0
  fi
  echo "--- attempt ${attempt} failed; sleeping before retry ---"
  sleep 10
done

echo "!!! giving up after ${MAX_ATTEMPTS} attempts" >&2
exit 1
