#!/usr/bin/env python3
"""Rounding drift of many fp32 decay steps, released in 16 bit (review M6, coordinator note).

AdamW multiplies every untouched weight by (1 - lr_t wd) in fp32 at every step. After N steps
the stored fp32 value is not P x base exactly (each product is rounded), and the 16-bit release
can differ from round(P x base) in a few numbers per vector. This simulates that and scores three
readings of "untouched":
  strict  round(P x base) == release in every number (the sweep's `unmoved`)
  band    every number between round(P_min x base) and base, P_min = 1 - 2 (1 - P) (recount.py)
and checks that vectors updated once by an Adam-like step (+-D per number) are not admitted.
Writes results/frontier/drift_sim.json.
"""
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "roster"))
from trainforensics import adam  # noqa: E402
import sweep  # noqa: E402

BF = torch.bfloat16


def sim(N, total_decay, V=1200, d=1024, n_upd=200, D=None, seed=0, sched="cosine"):
    g = torch.Generator().manual_seed(seed)
    base = (torch.randn(V, d, generator=g) * 0.02).to(BF)
    lr = adam.schedule(sched, N, 1.0, max(1, N // 50))[1:]
    wd = -np.log(1 - total_decay) / lr.sum()           # sum lr*wd = -ln P
    w = base.to(torch.float32).clone()
    upd_rows = torch.arange(n_upd)
    upd_step = torch.randint(N // 4, N - N // 10, (n_upd,), generator=g)
    s = torch.sign(torch.randn(n_upd, d, generator=g))
    D = D if D is not None else 3e-5
    for t in range(N):
        w.mul_(float(1 - lr[t] * wd))
        hit = upd_rows[upd_step == t]
        if len(hit):
            w[hit] -= D * s[(upd_step == t).nonzero().flatten()]
    rel = w.to(BF)
    P_true = float(np.prod(1 - lr * wd))
    exact = base.double()[n_upd:] * P_true
    nzm = exact != 0
    drift = ((w.double()[n_upd:] - exact).abs() / exact.abs().clamp_min(1e-300))[nzm]
    P, frac, _, _ = sweep.fit_decay(base, rel, BF)
    b = base.double()
    q = rel.double()
    strict = (sweep.rnd(b * P, BF) == q).all(1)
    Pmin = 1 - 2 * (1 - P) if P < 1 else 1.0
    lo, hi = sweep.rnd(b * Pmin, BF), sweep.rnd(b, BF)
    band = ((q >= torch.minimum(lo, hi)) & (q <= torch.maximum(lo, hi))).all(1)
    nd = (sweep.rnd(b * P, BF) != q).sum(1)
    un = slice(n_upd, V)
    return dict(N=N, total_decay=total_decay, P_true=P_true, P_fit=P, fit=frac, D=D, dim=d,
                fp32_rel_drift_q50=float(drift.quantile(0.5)) if drift.numel() < 1.6e7 else float(drift[:10**7].quantile(0.5)),
                fp32_rel_drift_q99=float(drift[:10**7].quantile(0.99)), fp32_rel_drift_max=float(drift.max()),
                untouched=V - n_upd, updated=n_upd,
                untouched_strict=int(strict[un].sum()), untouched_band=int(band[un].sum()),
                untouched_numbers_off_median=int(nd[un].median()), untouched_numbers_off_max=int(nd[un].max()),
                updated_admitted_strict=int(strict[:n_upd].sum()), updated_admitted_band=int(band[:n_upd].sum()))


def main():
    out = []
    for N, td in ((2000, 0.01), (20000, 0.01), (20000, 0.003), (20000, 0.05), (5000, 0.10)):
        r = sim(N, td)
        print(r, flush=True)
        out.append(r)
    # small updates near the detection floor
    for D in (1e-6, 3e-6, 1e-5):
        r = sim(20000, 0.01, D=D)
        print(r, flush=True)
        out.append(r)
    json.dump(out, open(ROOT / "results" / "frontier" / "drift_sim.json", "w"), indent=1)


if __name__ == "__main__":
    main()
