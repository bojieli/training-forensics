#!/bin/bash
# run all candidate comparisons, 3 at a time; skips names with an existing ok result
cd ~/training-forensics
python3 - <<'PY' > /tmp/y26_jobs.txt
import json, pathlib
for n, p, c, ds in json.load(open("scripts/y2026_ft/candidates.json")):
    f = pathlib.Path(f"results/y2026_ft/pairs/{n}.json")
    if f.exists() and json.load(open(f)).get("status", "").startswith("ok"):
        continue
    print(n, p, c)
PY
cat /tmp/y26_jobs.txt | xargs -P 3 -L 1 sh -c 'python3 scripts/y2026_ft/pair.py "$0" "$1" "$2" >> results/y2026_ft/run.log 2>&1; echo "DONE $0" >> results/y2026_ft/run.log'
echo ALLDONE >> results/y2026_ft/run.log
