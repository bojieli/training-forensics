#!/bin/bash
# New controlled runs for the gap study (swap generalisation, rare-token membership).
cd "$(dirname "$0")/../.."
F="python3 scripts/controlled/finetune.py --data-jsonl runs/alpaca.jsonl"
run() { n=$1; shift; [ -f runs/gaps/$n/truth.json ] && return; echo "== $n $*"; $F --out runs/gaps/$n "$@" > runs/gaps/$n.log 2>&1 || echo "FAIL $n"; }
run g01_swap_wd    --sched cosine     --lr 1e-4 --bs 16 --wd 0.1 --swap 40 --seed 9  --pool-seed 2
run g02_swap_bs4   --sched linear     --lr 2e-5 --bs 4 --warmup 10 --swap 40 --seed 10 --pool-seed 3
run g03_swap_cmin  --sched cosine_min --lr 3e-5 --warmup 15 --swap 40 --seed 11 --pool-seed 4 --min-lr-ratio 0.1
run g04_swap_const --sched constant   --lr 1e-4 --warmup 50 --swap 40 --seed 12 --pool-seed 7
run g05_mem_wiki   --source wiki --sched linear --lr 5e-5 --warmup 20 --seed 1002 --pool-seed 6 --save-model
# two epochs: the q7_2ep recipe (freeze body, train head, linear 1e-5, 2000 docs x 2) on Pythia-410m, plus a full fine-tune
run g06_2ep_q7like --model EleutherAI/pythia-410m --freeze-body --train-head --sched linear --lr 1e-5 --epochs 2 --n-docs 2000 --seed 99
run g07_2ep_cos    --sched cosine --lr 5e-5 --warmup 20 --epochs 2 --n-docs 4000 --seed 77 --pool-seed 8
echo ALLDONE2
