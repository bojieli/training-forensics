#!/usr/bin/env bash
# LIMO independent check, end to end. Inputs are fetched at pinned revisions (pins.py) into $TF_CACHE/limo_check
# (about 23 GB: 7 embedding matrices + the embedding slice of 32 optimizer shards); every step logs to
# results/limo_check/out/<step>.txt and writes results/limo_check/out/<step>.json; summarize.py assembles summary.json.
# Needs: torch, numpy, requests, huggingface_hub, tokenizers, datasets. CPU only.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../../results/limo_check/out; mkdir -p "$OUT"
for s in fetch compare run_args optim_analyze tok deeper qcheck controls summarize; do
  echo "== $s"
  python3 -u "$s.py" 2>&1 | sed -e "s#$HOME#~#g" | tee "$OUT/$s.txt"
done
python3 ../published/report_limo.py
