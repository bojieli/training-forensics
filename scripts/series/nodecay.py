#!/usr/bin/env python3
"""For series whose never-trained rows did not decay: confirm across the whole run
(first checkpoint vs last of each stage) that those rows stay bit-identical."""
import json, sys, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib import branches, emb, ROOT
R = ROOT / "results" / "series"
JOBS = [("allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B"), ("allenai/OLMo-2-1124-7B", "allenai/OLMo-2-0425-1B"),
        ("allenai/OLMo-2-1124-13B", "allenai/OLMo-2-0425-1B"), ("allenai/OLMo-2-0325-32B", "allenai/OLMo-2-0425-1B"),
        ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-1025-7B"), ("allenai/Olmo-3-1125-32B", "allenai/Olmo-3-1025-7B")]
out = []
for repo, detrepo in JOBS:
    ids = json.load(open(R / ("detect_" + detrepo.replace("/", "__") + ".json")))["ids"]
    lo, hi = max(0, ids[-1] - 255), ids[-1] + 1
    sel = torch.tensor([i - lo for i in ids if lo <= i < hi])
    br = branches(repo)
    stages = sorted({k[0] for k, _ in br})
    picks = [br[0]] + [[b for b in br if b[0][0] == s][-1] for s in stages]
    picks += [((99, 0, 0), "main")]
    X0 = None; rec = dict(repo=repo, rows=len(sel), window=[lo, hi], checks=[])
    for k, name in picks:
        try:
            X = emb(repo, name, rows=(lo, hi))[sel]
        except Exception as e:
            rec["checks"].append(dict(branch=name, error=repr(e)[:120])); continue
        if X0 is None:
            X0 = X; rec["first"] = name; continue
        same = (X.to(X0.dtype) == X0) if X.dtype != X0.dtype else (X == X0)
        r = (X.double() * X0.double()).sum() / (X0.double() ** 2).sum()
        rec["checks"].append(dict(branch=name, dtype=str(X.dtype), identical_rows=int(same.all(1).sum()), ratio=float(r)))
    print(json.dumps(rec), flush=True)
    out.append(rec)
json.dump(out, open(R / "nodecay.json", "w"), indent=1)
