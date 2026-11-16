#!/bin/bash
# Re-score every controlled run whose weights are on this machine, on one code version.
cd "$(dirname "$0")/../.."
export TF_PROCS=${TF_PROCS:-16}
git rev-parse HEAD > results/gaps/rescore_commit.txt
git diff HEAD --stat -- trainforensics scripts/controlled >> results/gaps/rescore_commit.txt
for r in runs/ctl/r12_cosmin_wd runs/ctl/r05_wsd runs/ctl/r01_lin runs/ctl/r02_cos runs/ctl/r03_const runs/ctl/r04_cosmin runs/ctl/r06_lin_bs4 \
         runs/ctl/r07_cos_wd runs/ctl/r08_lin_wd runs/ctl/r09_const_wd runs/ctl/r10_cos_410m runs/ctl/r11_lin_410m \
         runs/ctl/r13_wsd_bs4 runs/ctl/r14_lin_8k runs/ctl/r15_cos_bs32 runs/ctl/s02_bigbatch \
         runs/ctl/s04_lora runs/ctl/s05_swap runs/ctl/s06_bf16 runs/gaps/g01_swap_wd runs/gaps/g02_swap_bs4 \
         runs/gaps/g03_swap_cmin runs/gaps/g04_swap_const runs/ctl/m01; do
  n=$(basename $r)
  echo "== $n $(date +%T)"
  od=results/controlled; case $r in runs/gaps/*|runs/ctl/m01) od=results/gaps/scored;; esac
  python3 scripts/controlled/score.py $r --cache --outdir $od > runs/gaps/score_$n.log 2>&1 || echo "FAIL $n"
done
echo RESCORE_DONE $(date +%T)
