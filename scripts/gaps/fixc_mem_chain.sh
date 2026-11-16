#!/bin/bash
# Fix-C membership chain (M2): starts as soon as the blind recipes of m01, g05, n06 exist.
cd "$(dirname "$0")/../.."
have() { python3 - "$1" <<'PY'
import json,sys,pathlib
d={}
for f in ("results/controlled/weights_only.json","results/controlled/weights_only_new.json"):
    p=pathlib.Path(f)
    if p.exists(): d.update(json.load(open(p)))
sys.exit(0 if sys.argv[1] in d else 1)
PY
}
python3 scripts/gaps/membership3.py baselines runs/fixc/n06_mem_wiki2 > runs/fixc_mem_base.log 2>&1
echo baselines_done
for r in runs/ctl/m01 runs/gaps/g05_mem_wiki runs/fixc/n06_mem_wiki2; do
  n=$(basename $r)
  until have $n; do sleep 120; done
  TF_BLIND=1 python3 scripts/controlled/membership2.py feat $r >> runs/fixc_mem_feat.log 2>&1
  TF_BLIND=1 python3 scripts/gaps/membership3.py score $r >> runs/fixc_mem_score.log 2>&1
  echo "blind $n done"
done
python3 scripts/controlled/membership2.py feat runs/fixc/n06_mem_wiki2 >> runs/fixc_mem_feat.log 2>&1
python3 scripts/gaps/membership3.py score runs/fixc/n06_mem_wiki2 >> runs/fixc_mem_score.log 2>&1
echo MEM_DONE
