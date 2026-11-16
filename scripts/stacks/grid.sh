#!/bin/bash
# Training-stack runs (one GPU: RTX PRO 6000, 96 GB). Each line: name, launcher, finetune_stack.py args.
# Several ranks share the one GPU over gloo (NCCL refuses two ranks per GPU).
cd "$(dirname "$0")/../.."
source runs/env.sh 2>/dev/null
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH
F=scripts/stacks/finetune_stack.py
D=runs/stacks
mkdir -p $D
port=29700
run() {  # name nproc args...
  n=$1; np=$2; shift 2
  [ -f $D/$n/truth.json ] && return
  port=$((port + 1))
  echo "== $n $*"
  if [ "$np" = 0 ]; then python3 $F --out $D/$n "$@" > $D/$n.log 2>&1 || echo "FAIL $n"
  else torchrun --nproc_per_node $np --master_port $port $F --out $D/$n "$@" > $D/$n.log 2>&1 || echo "FAIL $n"; fi
}
L="--sched linear --lr 5e-5 --warmup 20 --wd 0.1"
C="--sched cosine --lr 1e-4 --warmup 20 --wd 0.1"
run t01_clip_off      0 $L --max-grad-norm 0   --seed 101
run t02_clip_on       0 $L --max-grad-norm 1.0 --seed 102
run t03_clip_tight    0 $C --max-grad-norm 0.01 --seed 103
run t04_ga4           0 --sched cosine --lr 1e-4 --warmup 20 --wd 0.05 --bs 2 --ga 4 --seed 104
run t05_ddp2          2 $L --same-gpu --bs 4 --seed 105
run t06_ddp4_ga2      4 $L --same-gpu --bs 1 --ga 2 --seed 106
run t07_fsdp_410m     1 $C --model EleutherAI/pythia-410m --bf16 --fsdp "full_shard auto_wrap" --fsdp-config scripts/stacks/fsdp_neox.json --n-docs 2000 --seed 107
run t08_ds_z2_dsopt   1 $L --bf16 --deepspeed scripts/stacks/ds_z2_dsopt.json --seed 108
run t09_ds_z2_torchopt 1 $L --bf16 --deepspeed scripts/stacks/ds_z2_torchopt.json --seed 109
run t10_ds_z3_offload 1 $L --bf16 --deepspeed scripts/stacks/ds_z3_offload.json --seed 110
run t11_muon_hidden   0 $C --muon hidden --seed 111
run t12_muon_all      0 $C --muon all --seed 112
run t13_pack          0 $L --pack --seed 113
run t14_purebf16_hi   0 --sched linear --lr 2e-4 --warmup 20 --wd 0.1 --pure-bf16 --seed 114
run t15_purebf16_lo   0 --sched linear --lr 2e-5 --warmup 20 --wd 0.1 --pure-bf16 --seed 115
echo ALLDONE
