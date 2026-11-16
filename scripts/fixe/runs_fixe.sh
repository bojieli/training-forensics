#!/bin/bash
# Fix-E: fresh controlled runs trained AFTER the reader v2 was frozen (git tag
# reader-v2-frozen). Non-round peak LR, weight decay, step count and warm-up;
# seeds below 100,000 (inside the seed search range); three of the reader's
# schedule families and one outside them (inverse_sqrt). Single epoch, Pythia-160m.
cd "$(dirname "$0")/../.."
F="python3 scripts/controlled/finetune.py --data-jsonl runs/alpaca.jsonl"
run() { n=$1; shift; [ -f runs/fixe/$n/truth.json ] && return; echo "== $n $*"; $F --out runs/fixe/$n "$@" > runs/fixe/$n.log 2>&1 || echo "FAIL $n"; }
mkdir -p runs/fixe
run f01_lin    --sched linear       --lr 3.3e-5 --wd 0.027 --warmup 13 --n-docs 3512 --seed 48127 --pool-seed 21
run f02_cos    --sched cosine       --lr 4.7e-5 --wd 0.019 --warmup 29 --n-docs 5144 --seed 7731  --pool-seed 22
run f03_cmin   --sched cosine_min   --lr 2.1e-5 --wd 0.057 --warmup 41 --n-docs 4456 --seed 90311 --pool-seed 23 --min-lr-ratio 0.1
run f04_isqrt  --sched inverse_sqrt --lr 3.9e-5 --wd 0.034 --warmup 23 --n-docs 3688 --seed 55021 --pool-seed 24
echo ALLDONE_FIXE
