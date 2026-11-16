#!/bin/bash
# Fix-C (tech review M1/M2/M3): fresh controlled runs trained AFTER the reader,
# swap test v2 and membership method were frozen (git tag fixc-frozen).
# Non-round peak LR and weight decay, one schedule outside the reader's five
# families (inverse_sqrt), and seeds drawn at random (two of them 32-bit).
cd "$(dirname "$0")/../.."
F="python3 scripts/controlled/finetune.py --data-jsonl runs/alpaca.jsonl"
run() { n=$1; shift; [ -f runs/fixc/$n/truth.json ] && return; echo "== $n $*"; $F --out runs/fixc/$n "$@" > runs/fixc/$n.log 2>&1 || echo "FAIL $n"; }
mkdir -p runs/fixc
run n01_clean_cos   --sched cosine       --lr 3.7e-5 --wd 0.07  --warmup 23 --seed 2918374651 --pool-seed 11
run n02_clean_isqrt --sched inverse_sqrt --lr 4.3e-5 --wd 0.03  --warmup 37 --seed 48213      --pool-seed 12
run n03_swap_lin    --sched linear       --lr 2.3e-5 --wd 0.013 --warmup 17 --seed 71523 --swap 40 --pool-seed 13
run n04_swap_cos    --sched cosine       --lr 6.1e-5            --warmup 9  --seed 3141592653 --swap 40 --pool-seed 14 --bs 16
run n05_swap_cmin   --sched cosine_min   --lr 1.7e-5 --wd 0.041 --warmup 31 --seed 60917 --swap 40 --pool-seed 15 --min-lr-ratio 0.1
run n06_mem_wiki2   --source wiki --sched linear --lr 5e-5 --warmup 20 --seed 2024 --pool-seed 16 --save-model
echo ALLDONE_FIXC
