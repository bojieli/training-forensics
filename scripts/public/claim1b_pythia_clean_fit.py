#!/usr/bin/env python3
"""Claim 1/2 robust refit: flag anomalous Pythia step branches, then fit the
cumulative weight decay of never-used padding rows on clean checkpoints only.

    y(s) = -log( <W_s, W_ref> / <W_ref, W_ref> ) = wd * sum_{i in (ref, s]} lr_i

A checkpoint is anomalous if its padding rows do not continue the monotone decay:
bit-identical to step 0 while the series decays, a jump back up (rows of a later
or earlier checkpoint), or a residual far above the series' rounding floor.
Uses the rows cached by claim1_pythia_lr.py (no network).
"""
import glob
import json
import math
import pathlib
import re
import sys

import numpy as np
import torch
from scipy.optimize import least_squares

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from claim1_pythia_lr import (CARD_LR, T_TRAIN, fetch_config, lr_cos_to_zero,  # noqa: E402
                              lr_linear, lr_neox_current, lr_pythia_era)

ROW = pathlib.Path.home() / ".cache/trainforensics/pythia_rows"
OUT = pathlib.Path(__file__).resolve().parents[2] / "results/public/claim1b_pythia_clean_fit.json"


def load(m):
    fs = glob.glob(str(ROW / f"EleutherAI__pythia-{m}_step*.pt"))
    return {int(re.search(r"_step(\d+)_", f).group(1)): torch.load(f).double() for f in fs}


def cum(fn, steps, **kw):
    i = np.arange(1, T_TRAIN + 1)
    cs = np.concatenate([[0.0], np.cumsum(fn(i - 1, **kw))])
    return cs[np.asarray(steps)]


def analyse(m):
    cfg = fetch_config(m)
    wd = cfg["weight_decay"]
    W = load(m)
    steps = sorted(W)
    W0 = W[0]
    r0 = {s: float((W0 * W[s]).sum() / (W0 * W0).sum()) for s in steps}
    res0 = {s: float((W[s] - r0[s] * W0).norm() / (r0[s] * W0).norm()) for s in steps}
    ident0 = [s for s in steps if s > 0 and torch.equal(W[s], W0)]
    late = [s for s in steps if s >= 1000]
    late_ident = [s for s in late if s in ident0]
    # If many post-warm-up checkpoints still equal step 0, the decaying series does not
    # start from this step 0: anchor on the earliest post-1000 checkpoint that moved.
    ref = 0
    if len(late_ident) > 0.2 * len(late):
        ref = [s for s in late if s not in ident0][0]
    Wr = W[ref]
    rr = {s: float((Wr * W[s]).sum() / (Wr * Wr).sum()) for s in steps if s >= ref}
    rres = {s: float((W[s] - rr[s] * Wr).norm() / (rr[s] * Wr).norm()) for s in rr}
    floor = float(np.median([rres[s] for s in rr if s > ref])) if ref == 0 else \
        float(np.median([rres[s] for s in rr if s > ref and s not in ident0]))
    # residual grows smoothly as rows shrink toward the rounding floor: compare each
    # checkpoint with the rolling median of its neighbours, not one global floor
    pool = [s for s in rr if s > ref and not (ref > 0 and s in ident0)]
    pres = np.array([rres[s] for s in pool])
    local = np.array([np.median(pres[max(0, k - 5): k + 6]) for k in range(len(pool))])
    cand = [s for s, v, lo in zip(pool, pres, local) if v <= 3 * max(lo, 1e-12)]
    # monotone: a clean checkpoint's ratio is below every earlier clean one and above every later one
    ratios = np.array([rr[s] for s in cand])
    clean = [ref]
    for k, s in enumerate(cand):
        before = ratios[:k]
        after = ratios[k + 1:]
        okb = (before >= ratios[k] * (1 - 1e-4)).sum() >= 0.9 * len(before) if len(before) else True
        oka = (after <= ratios[k] * (1 + 1e-4)).sum() >= 0.9 * len(after) if len(after) else True
        if okb and oka:
            clean.append(s)
    anomalies = sorted(set(s for s in steps if s > ref) - set(clean))
    y = np.array([-math.log(rr[s]) for s in clean])
    post = np.array([s >= 1000 for s in clean])
    cands = {}
    for name, fn in {"pythia_era_cos_to_0_clip_0.1": lr_pythia_era,
                     "neox_current_min_lr_cos": lr_neox_current,
                     "cos_to_0_no_clip": lr_cos_to_zero, "linear_to_0.1": lr_linear}.items():
        S = cum(fn, clean) - cum(fn, [ref])[0]
        A = float((S * y).sum() / (S * S).sum())
        rms = float(np.sqrt(np.mean(((y - A * S) / (A * 1000))[post] ** 2)))
        cands[name] = {"peak_lr": A / wd, "rms_resid_per_1000_steps_over_peak": rms}

    def f(p):
        pk, clip, T = p
        g = lambda n: lr_pythia_era(n, T=T, W=1430, clip=clip)  # noqa: E731
        S = cum(g, clean) - cum(g, [ref])[0]
        return (y - wd * pk * S)[post] / 1e-4

    sol = least_squares(f, x0=[cfg["lr"], 0.1, T_TRAIN], x_scale=[1e-4, 0.01, 1000])
    J = sol.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (sol.fun ** 2).sum() / max(1, post.sum() - 3)
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = [float("nan")] * 3
    pk, clip, T = sol.x
    t_clip = 1430 + T / math.pi * math.acos(max(-1, min(1, 2 * clip - 1)))
    lr_iv = [((a + b) / 2, (y[k + 1] - y[k]) / wd / (b - a))
             for k, (a, b) in enumerate(zip(clean[:-1], clean[1:])) if b - a >= 500]
    plateau = float(np.median([l for x, l in lr_iv if x > 135000]))
    first_flat = None
    for x, l in lr_iv:
        if x < 60000:
            continue
        if all(abs(ll / plateau - 1) < 0.02 for xx, ll in lr_iv if xx >= x):
            first_flat = x
            break
    return {"config_lr": cfg["lr"], "card_lr": CARD_LR[m], "wd": wd, "n_checkpoints": len(steps),
            "reference_step": ref, "n_clean": len(clean), "anomalous_checkpoints": anomalies,
            "identical_to_step0_after_1000": late_ident,
            "anomaly_detail": {str(s): {"ratio_vs_step0": r0[s], "resid_vs_step0": res0[s]} for s in anomalies},
            "fixed_shape_fits": cands,
            "free_fit": {"peak_lr": float(pk), "peak_err": float(err[0]), "clip_frac": float(clip),
                         "clip_err": float(err[1]), "T": float(T), "T_err": float(err[2]),
                         "clip_engages_at": float(t_clip), "flat_tail_steps": float(T_TRAIN - t_clip)},
            "model_free": {"plateau_lr": plateau, "plateau_over_free_peak": plateau / pk,
                           "flat_within_2pct_from_interval_midpoint": first_flat}}


if __name__ == "__main__":
    out = {}
    for m in sys.argv[1:] or ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]:
        out[m] = a = analyse(m)
        ff = a["free_fit"]
        fx = a["fixed_shape_fits"]
        an = a["anomalous_checkpoints"]
        print(f"{m:5s} cfg={a['config_lr']:.2e} card={a['card_lr']:.2e} ref=step{a['reference_step']} "
              f"clean={a['n_clean']}/{a['n_checkpoints']} anomalous({len(an)})={an[:10]}{'...' if len(an) > 10 else ''}")
        print(f"      pythia-era peak={fx['pythia_era_cos_to_0_clip_0.1']['peak_lr']:.5e} "
              f"rms={fx['pythia_era_cos_to_0_clip_0.1']['rms_resid_per_1000_steps_over_peak']:.4f} | "
              f"neox-current rms={fx['neox_current_min_lr_cos']['rms_resid_per_1000_steps_over_peak']:.4f} | "
              f"free peak={ff['peak_lr']:.5e}+-{ff['peak_err']:.1e} clip={ff['clip_frac']:.4f}+-{ff['clip_err']:.4f} "
              f"T={ff['T']:.0f} flat tail={ff['flat_tail_steps']:.0f} | plateau/peak="
              f"{a['model_free']['plateau_over_free_peak']:.4f} flat(2%) from ~{a['model_free']['flat_within_2pct_from_interval_midpoint']}")
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", OUT)
