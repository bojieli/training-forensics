#!/usr/bin/env python3
"""Exact interval of the uniform shrink P for large pairs with P < 1 (scripts/roster/p_interval.py
logic: every number of every vector that is exactly round(P x base) confines P)."""
import json, math, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "roster"))
import sweep, p_interval as pi  # noqa
import torch
PAIRS = {p["key"]: p for p in json.load(open(ROOT / "results" / "large" / "pairs_large.json"))}
out = []
for l in open(ROOT / "results" / "large" / "sweep_large.jsonl"):
    r = json.loads(l)
    if r.get("status") != "ok":
        continue
    e = r["embed"]; P = e["P"]
    if not (0.5 < P < 1) or e["unmoved"] < 10:
        continue
    p = PAIRS[r["key"]]; ce = p["compat"]["embed"]; r0, r1 = ce["common_rows"]
    dt = sweep.TD[ce["dtype_child"]]
    A = sweep.fetch(p["parent"])[r0:r1].to(dt); B = sweep.fetch(p["child"])[r0:r1].to(dt)
    pred = (A.double() * P).to(torch.float32).to(dt)
    full = (pred == B).all(1) & ~(A == B).all(1)
    rows = torch.nonzero(full).flatten()
    lo, hi, n = pi.interval(A[rows], B[rows])
    res = dict(key=r["key"], P_fit=P, rows=int(len(rows)), coords=n, P_lo=lo, P_hi=hi, consistent=lo <= hi,
               shrink_pct_lo=100 * (1 - hi), shrink_pct_hi=100 * (1 - lo),
               sum_lr_wd_lo=-math.log(hi) if 0 < hi <= 1 else None, sum_lr_wd_hi=-math.log(lo) if lo > 0 else None)
    out.append(res); print(res, flush=True)
json.dump(out, open(ROOT / "results" / "large" / "decay_intervals.json", "w"), indent=1)
