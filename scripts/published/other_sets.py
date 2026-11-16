#!/usr/bin/env python3
"""Dataset-version check: how many of a model's 'updated but absent' tokens occur in the other
published version of its dataset, and (exploratory) in other public reasoning sets we tokenized."""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "published"))
from trainforensics import remote
from compare import frac_changed, EMB, PUB
OUT = ROOT / "results" / "published"
base, child, data = sys.argv[1:4]
r = json.load(open(OUT / f"{child.replace('/', '__')}__{data}.json"))
remote.CACHE = EMB; W0 = remote.tensor(base, "model.embed_tokens.weight")
remote.CACHE = PUB; W1 = remote.tensor(child, "model.embed_tokens.weight")
n = min(len(W0), len(W1))
f = frac_changed(W0[:n], W1[:n], r["P"])
def S(d, key="main"):
    return set(int(k) for k in json.load(open(OUT / f"data_{d}.json"))["counts"][key])
mine = set()
for k in json.load(open(OUT / f"data_{data}.json"))["counts"]:
    mine |= S(data, k)
extra = set(np.nonzero(f > 0)[0].tolist()) - mine
res = dict(child=child, data=data, extra=len(extra), covered_by={})
for d in ("limo", "limo2", "s1", "s1.1", "skyt1", "stratos", "ot114k"):
    if d != data and (OUT / f"data_{d}.json").exists():
        s = S(d)
        res["covered_by"][d] = dict(covers=len(extra & s), set_size=len(s), set_tokens_not_updated=len(s - set(np.nonzero(f > 0)[0].tolist())))
json.dump(res, open(OUT / f"other_sets_{child.replace('/', '__')}.json", "w"), indent=1)
print(json.dumps(res))
