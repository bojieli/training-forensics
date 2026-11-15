#!/bin/bash
# Controlled runs for the blind test. Each line: name and finetune.py arguments.
cd "$(dirname "$0")/../.."
F="python3 scripts/controlled/finetune.py --data-jsonl runs/alpaca.jsonl"
run() { n=$1; shift; [ -f runs/ctl/$n/truth.json ] && return; echo "== $n $*"; $F --out runs/ctl/$n "$@" > runs/ctl/$n.log 2>&1 || echo "FAIL $n"; }
run r01_lin      --sched linear     --lr 5e-5 --warmup 20  --seed 42
run r02_cos      --sched cosine     --lr 2e-5             --seed 1234
run r03_const    --sched constant   --lr 1e-4 --warmup 50  --seed 7
run r04_cosmin   --sched cosine_min --lr 3e-5 --warmup 15  --seed 3407 --min-lr-ratio 0.1
run r05_wsd      --sched wsd        --lr 5e-5 --warmup 20  --seed 0 --min-lr-ratio 0.0
run r06_lin_bs4  --sched linear     --lr 1e-5 --bs 4       --seed 2024
run r07_cos_wd   --sched cosine     --lr 1e-4 --bs 16 --wd 0.1 --seed 11
run r08_lin_wd   --sched linear     --lr 2e-5 --wd 0.01   --seed 99
run r09_const_wd --sched constant   --lr 3e-5 --wd 0.1    --seed 5
run r10_cos_410m --model EleutherAI/pythia-410m --sched cosine --lr 5e-5 --warmup 100 --bs 4 --seed 31337
run r11_lin_410m --model EleutherAI/pythia-410m --sched linear --lr 3e-4 --warmup 10 --wd 0.05 --seed 2
run r12_cosmin_wd --sched cosine_min --lr 1e-4 --wd 0.1 --seed 123 --min-lr-ratio 0.1
run r13_wsd_bs4  --sched wsd        --lr 1e-4 --bs 4 --seed 77 --min-lr-ratio 0.0
run r14_lin_8k   --sched linear     --lr 7e-5 --n-docs 8000 --pool-seed 1 --seed 8
run r15_cos_bs32 --sched cosine     --lr 2e-4 --bs 32 --warmup 5 --seed 2025
# specials
run s01_2ep      --sched linear     --lr 5e-5 --epochs 2 --seed 42
run s02_bigbatch --sched cosine     --lr 1e-4 --bs 128 --n-docs 52000 --warmup 20 --seed 42
run s03_adaml2   --sched linear     --lr 5e-5 --optim adam_l2 --wd 0.01 --seed 42
run s04_lora     --sched linear     --lr 2e-4 --lora --seed 42
run s05_swap     --sched linear     --lr 5e-5 --swap 40 --seed 4242
run s06_bf16     --sched cosine     --lr 5e-5 --bf16 --seed 42
run s07_3ep      --sched cosine     --lr 3e-5 --epochs 3 --n-docs 2000 --seed 1
echo ALLDONE
