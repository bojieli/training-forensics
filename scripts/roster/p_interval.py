#!/usr/bin/env python3
"""Exact feasible interval of the decay factor P for roster pairs with decay.

A 16-bit child value b is the rounding of P*a (a = parent value). Every
coordinate therefore confines P to [lo(b)/a, hi(b)/a], where [lo, hi] is the
set of reals that round to b. Intersecting over all coordinates of the rows
that never received a gradient gives the interval of P the weights allow;
sum(lr*wd) = -ln P lies in the mapped interval. A single 'fitted P' quoted to
six digits is only meaningful when this interval is that narrow.

Reads the embedding cache written by sweep.py; writes results/roster/p_intervals.json.
"""
from __future__ import annotations

import glob
import json
import math
import os
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = pathlib.Path(os.environ.get("CKPTF_CACHE", pathlib.Path.home() / ".cache" / "trainforensics" / "emb"))
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
INT = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def load(repo, shape, dtype):
    pref = repo.replace("/", "__") + "@"
    cands = [f for f in glob.glob(str(CACHE / (pref + "*"))) if "lm_head" not in f and "output" not in f.split("__")[-2]]
    cands = [f for f in cands if "embed" in f or "wte" in f or "tok_embeddings" in f or "word_embeddings" in f] or cands
    need = shape[0] * shape[1] * torch.finfo(dtype).bits // 8
    for f in cands:
        if os.path.getsize(f) == need:
            return torch.frombuffer(bytearray(open(f, "rb").read()), dtype=dtype).view(*shape)
    return None


def neighbours(b: torch.Tensor):
    """Next representable values above and below b (same dtype), as float64."""
    it = INT[b.dtype]
    bits = b.contiguous().view(it).to(torch.int64)
    sign = b.double() < 0
    # moving the integer representation by +-1 moves the magnitude by one step
    up_bits = torch.where(sign, bits - 1, bits + 1)
    dn_bits = torch.where(sign, bits + 1, bits - 1)
    to = lambda x: x.to(it if it != torch.int16 else torch.int16).view(b.dtype).double()
    return to(up_bits), to(dn_bits)


def interval(a: torch.Tensor, b: torch.Tensor):
    a64, b64 = a.double(), b.double()
    up, dn = neighbours(b)
    hi_v = (b64 + up) / 2            # reals in [lo_v, hi_v] round to b (ties aside)
    lo_v = (b64 + dn) / 2
    ok = (a64 != 0) & (b64 != 0) & torch.isfinite(up) & torch.isfinite(dn)
    a64, lo_v, hi_v = a64[ok], lo_v[ok], hi_v[ok]
    if a64.numel() == 0:
        return float("nan"), float("nan"), 0
    r1, r2 = lo_v / a64, hi_v / a64
    lo = torch.minimum(r1, r2).max().item()
    hi = torch.maximum(r1, r2).min().item()
    return lo, hi, int(ok.sum())


def main():
    recs = {}
    for fn in ("sweep.jsonl", "sweep_cml3.jsonl"):
        p = ROOT / "results" / "roster" / fn
        if p.exists():
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") == "ok":
                    recs[r["key"]] = r
    out = []
    for k, r in sorted(recs.items(), key=lambda kv: kv[1]["embed"].get("P", 1)):
        e = r["embed"]
        P = e.get("P", 1.0)
        if not (0 < P < 1) or e.get("unmoved", 0) < 10 or r.get("tied"):
            continue
        dt = DT.get(r["dtype_child"])
        if dt is None or r["dtype_parent"] != r["dtype_child"]:
            continue
        A = load(r["parent"], r["shape_parent"], dt)
        B = load(r["child"], r["shape_child"], dt)
        if A is None or B is None or A.shape != B.shape:
            out.append(dict(key=k, error="cache miss"))
            continue
        pred = (A.double() * P).to(torch.float32).to(dt)
        eq = pred == B
        full = eq.all(1) & ~(A == B).all(1)
        rows = torch.nonzero(full).flatten()
        mode = "full rows"
        if len(rows) == 0:
            # no row matches in every coordinate: use the matching coordinates of
            # rows that mostly match (these pairs are weak evidence; flagged)
            rows = torch.nonzero((eq.double().mean(1) > 0.5) & ~(A == B).all(1)).flatten()
            mode = "partial rows"
        if len(rows) == 0:
            out.append(dict(key=k, error="no rows reproduce round(P*parent)"))
            continue
        sel = eq[rows] if mode == "partial rows" else torch.ones_like(eq[rows])
        lo, hi, n = interval(A[rows][sel], B[rows][sel])
        steps = (A[rows].contiguous().view(INT[dt]).to(torch.int64) - B[rows].contiguous().view(INT[dt]).to(torch.int64)).abs()
        rn = A[rows].double().norm(dim=1)
        res = dict(key=k, parent=r["parent"], child=r["child"], P_fit=P, rows=int(len(rows)), mode=mode, coords=n,
                   P_lo=lo, P_hi=hi, consistent=lo <= hi,
                   sum_lr_wd_lo=(-math.log(hi) if 0 < hi <= 1 else None),
                   sum_lr_wd_hi=(-math.log(lo) if lo > 0 else None),
                   rel_width=(hi - lo) / P if lo <= hi else None,
                   ulp_steps=dict(zip(*[x.tolist() for x in torch.unique(steps, return_counts=True)])) if len(rows) else {},
                   row_norm_median=float(rn.median()) if len(rn) else None)
        out.append(res)
        print(f"{r['child'][:46]:46s} rows {res['rows']:6d}  P in [{lo:.6f}, {hi:.6f}]  sum lr*wd in "
              f"[{res['sum_lr_wd_lo'] if res['sum_lr_wd_lo'] is not None else float('nan'):.5f}, "
              f"{res['sum_lr_wd_hi'] if res['sum_lr_wd_hi'] is not None else float('nan'):.5f}]  "
              f"steps {dict(list(res['ulp_steps'].items())[:4])}  |row| {res['row_norm_median']:.2e}", flush=True)
    json.dump(out, open(ROOT / "results" / "roster" / "p_intervals.json", "w"), indent=1)


if __name__ == "__main__":
    main()
