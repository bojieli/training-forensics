#!/usr/bin/env python3
"""Fetch the never-trained rows of every checkpoint in a series (one ranged read of
the densest window per checkpoint) and measure the decay between consecutive
checkpoints. Writes results/series/series_<repo>.json."""
import bisect, json, math, sys, pathlib
from concurrent.futures import ThreadPoolExecutor
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib import branches, emb, ROOT

repo = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
stage_filter = sys.argv[3] if len(sys.argv) > 3 else None   # e.g. "1" or "1,2"
det = json.load(open(ROOT / "results" / "series" / ("detect_" + repo.replace("/", "__") + ".json")))
ids = det["ids"]
k, i0 = max((bisect.bisect_left(ids, i + W) - j, i) for j, i in enumerate(ids))
lo = i0; hi = min(det["rows"], lo + W)
sel = torch.tensor([i - lo for i in ids if lo <= i < hi])
br = branches(repo)
if stage_filter:
    keep = {int(s) for s in stage_filter.split(",")}
    br = [b for b in br if b[0][0] in keep]

def get(b):
    for t in range(4):
        try:
            return b, emb(repo, b[1], rows=(lo, hi))[sel].clone()
        except Exception as e:
            err = e
    return b, err

with ThreadPoolExecutor(8) as ex:
    got = list(ex.map(get, br))
rows = []
prev = None
for (key, name), X in got:
    if isinstance(X, Exception):
        rows.append(dict(branch=name, key=key, error=repr(X)[:200])); continue
    X = X.double()
    rec = dict(branch=name, key=list(key), norm=float(X.norm()), dtype=str(X.dtype))
    if prev is not None:
        A = prev
        m = (A != 0) & (X != 0)
        r = X[m] / A[m]
        rec.update(P=float((A * X).sum() / (A * A).sum()), P_med=float(r.median()), P_lo=float(r.min()), P_hi=float(r.max()),
                   P_q01=float(r.quantile(0.01)) if r.numel() < 16_000_000 else None,
                   P_q99=float(r.quantile(0.99)) if r.numel() < 16_000_000 else None,
                   identical=bool((A == X).all()), coords=int(m.sum()))
    rows.append(rec)
    prev = X
out = dict(repo=repo, window=[lo, hi], rows_used=int(len(sel)), never_trained_total=len(ids), checkpoints=rows)
json.dump(out, open(ROOT / "results" / "series" / ("series_" + repo.replace("/", "__") + ".json"), "w"), indent=0)
print(repo, "window", lo, hi, "rows", len(sel), "ckpts", len(rows), "errors", sum("error" in r for r in rows))
