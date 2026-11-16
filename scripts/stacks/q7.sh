#!/bin/bash
# Qwen2.5-7B full fine-tune, DeepSpeed ZeRO-3 with optimizer state on NVMe (one GPU, <40 GB).
cd "$(dirname "$0")/../.."
source runs/env.sh 2>/dev/null
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH
torchrun --nproc_per_node 1 --master_port 29790 scripts/stacks/finetune_stack.py --out runs/stacks/q7_ds_z3_nvme \
  --model Qwen/Qwen2.5-7B --load-bf16 --bf16 --deepspeed scripts/stacks/ds_z3_nvme.json --grad-ckpt \
  --bs 4 --ga 2 --n-docs 480 --sched linear --lr 1e-5 --warmup 5 --wd 0.1 --seed 116 > runs/stacks/q7_ds_z3_nvme.log 2>&1
rm -rf /tmp/ds_nvme/*
echo Q7DONE
