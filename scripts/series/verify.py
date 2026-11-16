#!/usr/bin/env python3
"""Validation of the series claims against exact bounds (bounds_<repo>.json).
Writes results/series/verify.json."""
import json, math, pathlib, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib import ROOT
import analyze
R = ROOT / "results" / "series"
out = {}

def load(repo):
    p = R / ("bounds_" + repo.replace("/", "__") + ".json")
    return json.load(open(p)) if p.exists() else None

def inside(pred, rows):
    """fraction of intervals whose predicted total decay lies inside the exact bound"""
    ok = [(r["S_lo"] - 1e-15) <= p <= (r["S_hi"] + 1e-15) for p, r in zip(pred, rows)]
    return sum(ok), len(ok)

def ratio_bounds(pred, rows):
    lo = [r["S_lo"] / p for p, r in zip(pred, rows)]
    hi = [r["S_hi"] / p for p, r in zip(pred, rows)]
    return lo, hi

def sched(rows, wd, kind, peak, W, T, alpha):
    a = np.array([r["a"] for r in rows], float); b = np.array([r["b"] for r in rows], float)
    return analyze.interval_sum(a, b, kind, peak, W, T, alpha) * wd

# ---- OLMo-7B-0724: card == official-0724/OLMo-7B-0724.yaml (OLMo warm-up starts at 0.1x peak)
def olmo_lr(t, peak=3e-4, W=2500, T=3e12 / (1024 * 4096), alpha=0.1, start=0.1):
    t = np.asarray(t, float)
    warm = peak * (start + (1 - start) * np.minimum(t, W) / W)
    x = np.clip((t - W) / (T - W), 0, 1)
    dec = peak * (alpha + (1 - alpha) * 0.5 * (1 + np.cos(np.pi * x)))
    return np.where(t < W, warm, dec)

def olmo_sum(rows, wd, **kw):
    s = []
    for r in rows:
        ts = np.arange(r["a"] + 1, r["b"] + 1)      # lr used at steps a+1..b (step index after update)
        s.append(olmo_lr(ts - 1, **kw).sum() * wd)
    return np.array(s)

d = load("allenai/OLMo-7B-0724-hf")
if d:
    rows = [r for r in d["intervals"] if r["S_lo"] is not None and r["S_hi"] is not None]
    res = {}
    for name, kw in [("config OLMo-7B-0724.yaml (cosine, warm-up from 0.1x over 2500 steps, 3e-4 -> 3e-5 at 3T tokens)", {}),
                     ("same, warm-up from 0", dict(start=0.0)),
                     ("wrong file OLMo-7B.yaml (linear, warm-up 5000, 2T tokens)", None)]:
        if kw is None:
            pred = sched(rows, 0.1, "lin", 3e-4, 5000, 2e12 / (2048 * 2048), 0.1)
        else:
            pred = olmo_sum(rows, 0.1, **kw)
        k, n = inside(pred, rows)
        lo, hi = ratio_bounds(pred, rows)
        mid = np.array([(x + y) / 2 for x, y in zip(lo, hi)])
        res[name] = dict(inside=k, n=n, median_ratio=float(np.median(mid)), q10=float(np.quantile(mid, .1)), q90=float(np.quantile(mid, .9)),
                         first_interval=dict(a=rows[0]["a"], b=rows[0]["b"], ratio_lo=lo[0], ratio_hi=hi[0],
                                             lr_bound=[rows[0]["lr_lo"], rows[0]["lr_hi"]]))
    # where do the ratios leave 1? (anneal at the end)
    pred = olmo_sum(rows, 0.1)
    lo, hi = ratio_bounds(pred, rows)
    bad = [dict(a=r["a"], b=r["b"], ratio=[l, h]) for r, l, h in zip(rows, lo, hi) if not (l <= 1 <= h)]
    res["outside_intervals_vs_config"] = bad[:20]
    res["n_outside"] = len(bad)
    res["median_rel_width"] = float(np.median([(h - l) for l, h in zip(lo, hi)]))
    out["OLMo-7B-0724"] = res

# ---- Viking
d = load("LumiOpen/Viking-7B")
if d:
    rows = d["intervals"]
    sp = analyze.SPEC["LumiOpen/Viking-7B"]["published"][0]
    pred = sched(rows, 0.1, sp["kind"], sp["peak"], sp["W"], sp["T"], sp["alpha"])
    lo, hi = ratio_bounds(pred, rows)
    out["Viking-7B"] = dict(intervals=[dict(a_tokens=r["branch_a"], b_tokens=r["branch_b"], ratio_lo=l, ratio_hi=h) for r, l, h in zip(rows, lo, hi)])

# ---- Apertus: stable phase
d = load("swiss-ai/Apertus-8B-2509")
if d:
    rows = [r for r in d["intervals"] if r["lr_lo"] is not None and r["lr_hi"] is not None]
    lr_lo = np.array([r["lr_lo"] for r in rows]); lr_hi = np.array([r["lr_hi"] for r in rows])
    mid = (lr_lo + lr_hi) / 2
    stable = [i for i, m in enumerate(mid) if abs(m / 1.1e-4 - 1) < 0.05]
    cont = [i for i in stable if lr_lo[i] <= 1.1e-4 <= lr_hi[i]]
    out["Apertus-8B"] = dict(n_intervals=len(rows), stable_intervals=len(stable), containing_1p1e4=len(cont),
                             common_lr_interval=[float(lr_lo[stable].max()), float(lr_hi[stable].min())] if stable else None,
                             median_width=float(np.median((lr_hi - lr_lo)[stable])) if stable else None,
                             outside=[dict(a=rows[i]["a"], b=rows[i]["b"], lr=[float(lr_lo[i]), float(lr_hi[i])]) for i in stable if i not in cont][:10])

# ---- OLMoE: peak and horizon
d = load("allenai/OLMoE-1B-7B-0924")
if d:
    rows = [r for r in d["intervals"] if r["S_lo"] is not None and r["S_hi"] is not None]
    res = {}
    for name, T in [("horizon 5.0T tokens", 5.0e12 / (1024 * 4096)), ("horizon 5.133T tokens (all data)", 5.133e12 / (1024 * 4096))]:
        pred = sched(rows, 0.1, "cos", 4e-4, 2500, T, 0.1)
        k, n = inside(pred, rows)
        lo, hi = ratio_bounds(pred, rows)
        mid = np.array([(x + y) / 2 for x, y in zip(lo, hi)])
        res[name] = dict(inside=k, n=n, median_ratio=float(np.median(mid)), rms_dev=float(np.sqrt(np.mean((mid - 1) ** 2))))
    res["median_rel_width"] = float(np.median([(r["S_hi"] - r["S_lo"]) / ((r["S_hi"] + r["S_lo"]) / 2) for r in rows if r["S_hi"] + r["S_lo"] > 0]))
    out["OLMoE-1B-7B"] = res

# ---- K2: floor 0.1 vs fitted; spacing
d = load("LLM360/K2")
if d:
    rows = [r for r in d["intervals"] if r["S_lo"] is not None and r["S_hi"] is not None]
    a = np.array([r["a"] for r in rows], float); b = np.array([r["b"] for r in rows], float)
    S = np.array([(r["S_lo"] + r["S_hi"]) / 2 for r in rows]); Wd = np.array([(r["S_hi"] - r["S_lo"]) / 2 for r in rows])
    def fit(alpha_fixed=None):
        best = None
        for T0 in (330, 360, 400, 450):
            def f(p):
                al = alpha_fixed if alpha_fixed is not None else p[2]
                return (analyze.interval_sum(a, b, "cos", np.exp(p[0]), 1e-3, p[1], al) * 0.1 - S) / S
            x0 = [math.log(0.14), T0] + ([0.1] if alpha_fixed is None else [])
            lb = [-10, 200] + ([0] if alpha_fixed is None else []); ub = [5, 2000] + ([1] if alpha_fixed is None else [])
            r = least_squares(f, x0, bounds=(lb, ub))
            if best is None or r.cost < best.cost:
                best = r
        p = best.x
        return dict(peak_x_steps=float(math.exp(p[0])), T_chunks=float(p[1]), alpha=float(p[2]) if alpha_fixed is None else alpha_fixed,
                    rel_rms=float(math.sqrt(2 * best.cost / len(S))))
    free, fixed = fit(), fit(0.1)
    # bootstrap over intervals for alpha uncertainty
    rng = np.random.default_rng(0); al = []
    a0, b0, S0 = a.copy(), b.copy(), S.copy()
    for _ in range(100):
        i = np.sort(rng.integers(0, len(S0), len(S0)))
        a, b, S = a0[i], b0[i], S0[i]
        al.append(fit()["alpha"])
    a, b, S = a0, b0, S0
    out["K2"] = dict(n_intervals=len(rows), chunk_spacing="every 3rd chunk, uniform (ckpt_003..ckpt_360)",
                     rounding_rel_halfwidth_median=float(np.median(Wd / S)), fit_free=free, fit_alpha_0p1=fixed,
                     alpha_bootstrap_5_95=[float(np.quantile(al, .05)), float(np.quantile(al, .95))])

# ---- Amber: format switch and post-switch schedule
d = load("LLM360/Amber")
if d:
    rows = d["intervals"]
    first_change = next((r for r in rows if not r["identical"]), None)
    bf = [r for r in rows if r["dtype_a"] == "torch.bfloat16" and r["dtype_b"] == "torch.bfloat16"]
    fp = [r for r in rows if r["dtype_a"] == "torch.float32" and r["dtype_b"] == "torch.float32"]
    # post switch: published cosine 3e-4 -> 3e-5 over 360 chunks (shape only), fit steps per chunk
    a = np.array([r["a"] for r in fp], float); b = np.array([r["b"] for r in fp], float)
    S = np.array([(r["S_lo"] + r["S_hi"]) / 2 for r in fp])
    shape = analyze.interval_sum(a, b, "cos", 3e-4, 2000 / 900, 360, 0.1) * 0.1
    s = float(np.sum(S * shape) / np.sum(shape * shape))
    resid = S / (s * shape) - 1
    out["Amber"] = dict(bf16_intervals=len(bf), bf16_identical=sum(r["identical"] for r in bf),
                        fp32_intervals=len(fp), first_change=dict(a=first_change["branch_a"], b=first_change["branch_b"],
                        dtype=[first_change["dtype_a"], first_change["dtype_b"]], P=[first_change["P_lo"], first_change["P_hi"]]) if first_change else None,
                        per_step_relative_decay_at_peak=3e-4 * 0.1, bf16_relative_half_spacing_min=2.0 ** -9,
                        post_switch_steps_per_chunk_if_published_cosine=s, post_switch_rel_rms=float(np.sqrt(np.mean(resid ** 2))),
                        post_switch_max_abs_dev=float(np.max(np.abs(resid))))
json.dump(out, open(R / "verify.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:8000])
