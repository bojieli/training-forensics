#!/bin/bash
# Fix-C chain: blind scoring of the fresh runs, frozen swap test v2, blind-recipe membership.
cd "$(dirname "$0")/../.."
export TF_PROCS=8
NEW="runs/fixc/n01_clean_cos runs/fixc/n02_clean_isqrt runs/fixc/n03_swap_lin runs/fixc/n04_swap_cos runs/fixc/n05_swap_cmin"
echo "== score"; python3 scripts/controlled/score.py --cache --outdir results/fixc $NEW > runs/fixc_score.log 2>&1
echo "== swaps"; python3 scripts/gaps/swaps.py n01_clean_cos n02_clean_isqrt n03_swap_lin n04_swap_cos n05_swap_cmin > runs/fixc_swaps.log 2>&1
echo "== wait weights_only"; while pgrep -f "weights_only.py runs/ctl" > /dev/null; do sleep 60; done
echo "== weights_only new"; python3 scripts/controlled/weights_only.py runs/fixc/n06_mem_wiki2 $NEW >> runs/fixc_weights_only.log 2>&1
echo "== membership"
for r in runs/fixc/n06_mem_wiki2; do python3 scripts/gaps/membership3.py baselines $r > runs/fixc_mem_base.log 2>&1; done
for r in runs/ctl/m01 runs/gaps/g05_mem_wiki runs/fixc/n06_mem_wiki2; do
  TF_BLIND=1 python3 scripts/controlled/membership2.py feat $r >> runs/fixc_mem_feat.log 2>&1
  TF_BLIND=1 python3 scripts/gaps/membership3.py score $r >> runs/fixc_mem_score.log 2>&1
done
python3 scripts/controlled/membership2.py feat runs/fixc/n06_mem_wiki2 >> runs/fixc_mem_feat.log 2>&1
python3 scripts/gaps/membership3.py score runs/fixc/n06_mem_wiki2 >> runs/fixc_mem_score.log 2>&1
echo CHAIN_DONE
