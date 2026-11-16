#!/usr/bin/env python3
"""Classify each 2026 comparison and compute exact total-weight-decay intervals.

Comparability classes from scripts/frontier/comparability.py; exact P intervals with
scripts/roster/p_interval.py (rows that equal round(P x parent) in every coordinate).
Writes results/y2026_labs/analysis.json.
"""
import json, math, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "frontier")); sys.path.insert(0, str(ROOT / "scripts" / "roster"))
from comparability import classify  # noqa
import p_interval as pi  # noqa
import torch  # noqa

def main():
    R = {}
    for l in open(ROOT / "results" / "roster" / "sweep_2026_labs.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok": R[r["key"]] = r
    out = []
    for k, r in R.items():
        e = r["embed"]
        rec = dict(key=k, parent=r["parent"], child=r["child"], relation=r.get("relation"), tied=r.get("tied"),
                   cls=classify(r), rows=e.get("rows"), bit_identical=e.get("unmoved"), barely=e.get("near"),
                   changed=e.get("moved"), identical_raw=e.get("identical"), P=e.get("P"), fit=e.get("P_exact_frac"),
                   change_hist=r.get("change_hist", {}).get("frac"), dtype=r.get("dtype_child"),
                   special=r.get("special_summary"))
        P = e.get("P", 1.0)
        if rec["cls"] in ("token record", "record, poor fit") and 0 < P < 0.99999 and not r.get("tied"):
            dt = pi.DT.get(r["dtype_child"])
            try:
                A = pi.load(r["parent"], r["shape_parent"], dt); B = pi.load(r["child"], r["shape_child"], dt)
                if A is not None and B is not None and A.shape == B.shape:
                    pred = (A.double() * P).to(torch.float32).to(dt)
                    full = (pred == B).all(1) & ~(A == B).all(1)
                    rows = torch.nonzero(full).flatten()
                    if len(rows):
                        lo, hi, n = pi.interval(A[rows], B[rows])
                        rec.update(P_lo=lo, P_hi=hi, decay_rows=int(len(rows)),
                                   shrink_pct=[100 * (1 - hi), 100 * (1 - lo)],
                                   sum_lr_wd=[-math.log(hi) if 0 < hi <= 1 else None, -math.log(lo) if lo > 0 else None])
                else:
                    rec["P_interval_error"] = "cache miss or shape mismatch"
            except Exception as ex:
                rec["P_interval_error"] = str(ex)[:200]
        out.append(rec)
        print(rec["cls"][:18].ljust(18), k[:100], rec.get("shrink_pct"), flush=True)
    json.dump(out, open(ROOT / "results" / "y2026_labs" / "analysis.json", "w"), indent=1)

if __name__ == "__main__":
    main()
