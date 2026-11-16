#!/usr/bin/env python3
"""Find never-trained rows of a checkpoint series: rows whose late value is a
pure rescaling of their early value (B = c*A coordinatewise up to storage rounding),
with one common c. Writes results/series/detect_<repo>.json."""
import json, sys, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib import branches, emb, ROOT
repo, ia, ib = sys.argv[1], sys.argv[2], sys.argv[3]
br = branches(repo)
pick = lambda s: br[int(s)][1] if s.lstrip("-").isdigit() else s
ra, rb = pick(ia), pick(ib)
A = emb(repo, ra).double(); B = emb(repo, rb).double()
nA = (A * A).sum(1)
c = (A * B).sum(1) / nA.clamp_min(1e-300)
res = (B - c[:, None] * A).norm(dim=1) / B.norm(dim=1).clamp_min(1e-300)
ident = (A == B).all(1)
# candidates: residual tiny (only storage rounding)
cand = (res < 5e-3) & (nA > 0)
cs = c[cand]
med = cs.median().item() if cand.any() else None
tight = cand & ((c - med).abs() < 1e-3 * abs(med)) if med else cand
ids = torch.nonzero(tight).flatten().tolist()
out = dict(repo=repo, a=ra, b=rb, rows=A.shape[0], dtype=str(emb(repo, ra).dtype),
           identical=int(ident.sum()), candidates=int(cand.sum()), never_trained=len(ids), c_median=med,
           ids=ids, res_hist=[int(((res >= lo) & (res < hi)).sum()) for lo, hi in
                              [(0, 1e-6), (1e-6, 1e-4), (1e-4, 5e-3), (5e-3, 0.05), (0.05, 0.5), (0.5, 1e9)]],
           id_ranges=None)
# compress ids to ranges
rg = []
for i in ids:
    if rg and i == rg[-1][1] + 1: rg[-1][1] = i
    else: rg.append([i, i])
out["id_ranges"] = rg[:200]
p = ROOT / "results" / "series" / ("detect_" + repo.replace("/", "__") + ".json")
json.dump(out, open(p, "w"))
print(json.dumps({k: v for k, v in out.items() if k not in ("ids",)})[:1500])
