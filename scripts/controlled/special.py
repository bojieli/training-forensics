#!/usr/bin/env python3
"""Checks for the regimes the thread lists under "what made it stronger".

  multi   runs/ctl/s01_2ep   rows touched twice: two shades D1+D2 and |D1-D2|
  batch   runs/ctl/s02_bigbatch  large batch: row max |delta| vs D(step)
  l2      runs/ctl/s03_adaml2   Adam + L2: replay the decay-only gradient on
                                never-seen rows with torch.optim.Adam, bit for bit

Truth is used to label rows (which steps touched them) and, for the L2 replay,
the logged learning rates; each section says what it used.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import blind  # noqa: E402
from trainforensics.diff import row_diff  # noqa: E402
from trainforensics.truth import Run  # noqa: E402


def two_shades(d: np.ndarray):
    """|delta| of one row -> (hi, lo, frac_explained). 1-D 2-means in log space."""
    x = np.log(np.maximum(d, 1e-30))
    a, b = np.quantile(x, 0.25), np.quantile(x, 0.9)
    for _ in range(30):
        m = np.abs(x - a) < np.abs(x - b)
        if m.all() or (~m).all():
            break
        a2, b2 = np.median(x[m]), np.median(x[~m])
        if a2 == a and b2 == b:
            break
        a, b = a2, b2
    lo, hi = np.exp(min(a, b)), np.exp(max(a, b))
    near = (np.abs(d - hi) < 0.01 * hi) | (np.abs(d - lo) < 0.01 * max(lo, 1e-30))
    return hi, lo, float(near.mean())


def multi(run):
    R = Run(run)
    rd = row_diff(R.emb["W0"], R.emb["W1"], torch.float32)
    D = blind.moves_curve(R.lr, R.t["args"]["wd"])
    W0, W1 = R.emb["W0"].double(), R.emb["W1"].double()
    ts = R.token_steps()
    rows = [(t, s) for t, s in ts.items() if len(s) == 2]
    out = []
    for t, (s1, s2) in rows:
        d = (W1[t] - rd.decay * W0[t]).abs().numpy()
        hi, lo, fr = two_shades(d)
        e1, e2 = (hi + lo) / 2, (hi - lo) / 2           # larger, smaller single moves
        t1, t2 = sorted((D[s1], D[s2]), reverse=True)
        out.append(dict(tok=t, s1=s1, s2=s2, gap=abs(s2 - s1), frac2=fr,
                        err_big=e1 / t1 - 1, err_small=e2 / t2 - 1 if t2 > 0 else None))
    res = {"rows_touched_twice": len(out)}
    for name, lo_gap, hi_gap in (("all", 0, 10 ** 9), ("gap<50", 0, 50), ("gap>=200", 200, 10 ** 9),
                                  ("gap>=N/2", R.N // 4, 10 ** 9)):
        sel = [o for o in out if lo_gap <= o["gap"] < hi_gap and o["err_small"] is not None]
        if not sel:
            continue
        both = np.mean([abs(o["err_big"]) < 0.01 and abs(o["err_small"]) < 0.01 for o in sel])
        res[name] = dict(rows=len(sel), both_within_1pct=float(both),
                         median_two_shade_frac=float(np.median([o["frac2"] for o in sel])))
    return res


def batch(run):
    R = Run(run)
    rd = row_diff(R.emb["W0"], R.emb["W1"], torch.float32)
    D = blind.moves_curve(R.lr, R.t["args"]["wd"])
    W0, W1 = R.emb["W0"].double(), R.emb["W1"].double()
    ts = R.token_steps()
    once = [(t, s[0]) for t, s in ts.items() if len(s) == 1]
    res = {"single_touch_rows": len(once), "flat_frac>0.9": float(np.mean([rd.flat_frac[t] > 0.9 for t, _ in once]))}
    q = {}
    for name, fn in (("median", lambda d: np.median(d)), ("q99", lambda d: np.quantile(d, 0.99)),
                     ("max", lambda d: d.max())):
        e = np.array([fn((W1[t] - rd.decay * W0[t]).abs().numpy()) / D[s] - 1 for t, s in once])
        q[name] = dict(median_err=float(np.median(e)), within_1pct=float(np.mean(np.abs(e) < 0.01)),
                       within_2pct=float(np.mean(np.abs(e) < 0.02)))
    res["row_stat_vs_D(step)"] = q
    # blind recipe from the row maxima of all moved rows that look single-touch-like
    rows = np.nonzero(rd.moved)[0]
    mx = np.array([(W1[t] - rd.decay * W0[t]).abs().max().item() for t in rows])
    # rows hit once: their max lies on the D(s) curve; multi-touch rows lie above it
    best, ranked = blind.fit_recipe(mx[rd.flat_frac[rows] > 0.0], wd=R.t["args"]["wd"])
    res["blind_recipe_from_row_max"] = best.as_dict()
    res["true"] = dict(sched=R.t["args"]["sched"], N=R.N, warmup=R.t["args"]["warmup"], lr=R.t["args"]["lr"])
    return res


def l2(run):
    """Replay Adam on g = wd * theta for rows that never had a data gradient."""
    R = Run(run)
    a = R.t["args"]
    wd = a["wd"]
    W0, W1 = R.emb["W0"], R.emb["W1"]
    seen = R.seen_mask()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = torch.nn.Parameter(W0.clone().to(dev))
    opt = torch.optim.Adam([p], lr=a["lr"], betas=(a["b1"], a["b2"]), eps=a["eps"], weight_decay=wd, foreach=True)
    for t in range(1, R.N + 1):
        for g in opt.param_groups:
            g["lr"] = float(R.lr[t])
        p.grad = torch.zeros_like(p)
        opt.step()
    x = p.detach().cpu()
    same = (x == W1).all(1).numpy()
    res = dict(rows=int(len(same)), replay_equal_rows=int(same.sum()), seen=int(seen.sum()),
               unseen=int((~seen).sum()), unseen_and_equal=int((same & ~seen).sum()),
               seen_but_equal=int((same & seen).sum()),
               bitwise_weights_equal=int((x[~seen] == W1[~seen]).sum()), unseen_weights=int((~seen).sum() * W0.shape[1]))
    res["tokens_from_replay_exact"] = bool(((~same) == seen).all())
    res["note"] = "uses the logged LR list (recipe recovery under L2 is separate)"
    # naive rule (moved = not identical) for contrast
    res["naive_identical_rows"] = int((W0 == W1).all(1).sum())
    return res


if __name__ == "__main__":
    kind, run = sys.argv[1], sys.argv[2]
    res = {"multi": multi, "batch": batch, "l2": l2}[kind](run)
    dst = ROOT / "results" / "controlled" / f"{pathlib.Path(run).name}.{kind}.json"
    json.dump(res, open(dst, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))
