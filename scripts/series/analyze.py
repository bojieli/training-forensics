#!/usr/bin/env python3
"""Per-interval weight decay of never-updated token vectors -> learning-rate schedule,
compared with the published recipe.

Between two checkpoints a < b, a token vector that receives no gradient is only
shrunk by decoupled weight decay: W_b = P * W_a with P = prod_{t in (a,b]} (1 - lr_t * wd).
So -ln P ~= wd * sum(lr_t): the total weight decay applied over the interval.
Reads results/series/series_<repo>.json, writes results/series/fit_<repo>.json and a figure.
"""
import json, math, re, sys, pathlib
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib import ROOT, step_of
R = ROOT / "results" / "series"

def olmo_step(b):
    return int(re.search(r"step-?(\d+)", b).group(1))
def tl_step(b):
    return int(re.match(r"step-(\d+)k-", b).group(1)) * 1000
def chunk(b):
    return step_of(b)[2]
def viking_step(b):
    return int(round(int(b[:-1]) * 1e9 / (1024 * 4096)))

TL = dict(kind="cos", peak=4e-4, W=2000, T=3e12 / (1024 * 2048), alpha=0.1)
SPEC = {
 "allenai/OLMo-7B-0724-hf": dict(wd=0.1, step=olmo_step, unit="step", published=[
     dict(src="HF card: cosine, warm-up 2500, 3e-4 -> 3e-5 at 3T tokens", kind="cos", peak=3e-4, W=2500, T=3e12 / (2048 * 2048), alpha=0.1),
     dict(src="GitHub configs/official-0724/OLMo-7B.yaml: linear, warm-up 5000, 3e-4 -> 3e-5 at 2T tokens", kind="lin", peak=3e-4, W=5000, T=2e12 / (2048 * 2048), alpha=0.1)]),
 "allenai/OLMo-7B-0424-hf": dict(wd=0.1, step=olmo_step, unit="step", published=[
     dict(src="HF card: cosine, warm-up 2500, 3e-4 -> 3e-5 at 3T tokens", kind="cos", peak=3e-4, W=2500, T=3e12 / (2048 * 2048), alpha=0.1)]),
 "allenai/OLMo-1B-0724-hf": dict(wd=0.1, step=olmo_step, unit="step", published=[
     dict(src="GitHub configs/official-0724/OLMo-1B.yaml: cosine, warm-up 2000, 4e-4 -> 4e-5 over 739,328 steps", kind="cos", peak=4e-4, W=2000, T=739328, alpha=0.1),
     dict(src="same, horizon 3.1T tokens at the released 2M-token batch", kind="cos", peak=4e-4, W=2000, T=3.1e12 / (1024 * 2048), alpha=0.1)]),
 "allenai/OLMoE-1B-7B-0924": dict(wd=0.1, step=olmo_step, unit="step", published=[
     dict(src="paper: cosine, warm-up 2500, 4e-4 -> 4e-5 (horizon not stated; 5.133T total tokens assumed)", kind="cos", peak=4e-4, W=2500, T=5.133e12 / (1024 * 4096), alpha=0.1)]),
 "TinyLlama/tinyLlama-intermediate-checkpoints": dict(wd=0.1, step=tl_step, unit="step", published=[
     dict(src="GitHub README: cosine, warm-up 2000, 4e-4 -> 4e-5 over 3T tokens (2M-token batch)", **TL)]),
 "TinyLlama/tinyLlama-intermediate-checkpoints-after-1T-token": dict(wd=0.1, step=tl_step, unit="step", published=[
     dict(src="GitHub README: cosine, warm-up 2000, 4e-4 -> 4e-5 over 3T tokens (2M-token batch)", **TL)]),
 "LumiOpen/Viking-7B": dict(wd=0.1, step=viking_step, unit="step (from token count / 4,194,304)", published=[
     dict(src="HF card: cosine, warm-up 10B tokens, 3e-4 -> 3e-5 over 2T tokens", kind="cos", peak=3e-4, W=10e9 / 4194304, T=2e12 / 4194304, alpha=0.1)]),
 "swiss-ai/Apertus-8B-2509": dict(wd=0.1, step=olmo_step, unit="step", stages=[1], published=[
     dict(src="paper: WSD, stable LR 1.1e-4 in this range (warm-up from 0.1x over 16.8B tokens)", kind="const", peak=1.1e-4, W=16.8e9 / 4194304, T=1e12, alpha=1.0)]),
 "LLM360/K2": dict(wd=0.1, step=chunk, unit="data chunk", stages=[1], peak_published=1.5e-4, published=[]),
 "LLM360/Amber": dict(wd=0.1, step=chunk, unit="data chunk", peak_published=3e-4, published=[]),
}

def lr_curve(t, kind, peak, W, T, alpha):
    t = np.asarray(t, float)
    warm = peak * np.minimum(t, W) / max(W, 1e-9)
    x = np.clip((t - W) / max(T - W, 1e-9), 0, 1)
    if kind == "cos":
        dec = peak * (alpha + (1 - alpha) * 0.5 * (1 + np.cos(np.pi * x)))
    elif kind == "lin":
        dec = peak * (alpha + (1 - alpha) * (1 - x))
    else:
        dec = peak * np.ones_like(x)
    return np.where(t < W, warm, dec)

_trap = getattr(np, "trapezoid", None) or np.trapz

def interval_sum(a, b, kind, peak, W, T, alpha):
    out = []
    for x, y in zip(a, b):
        ts = np.linspace(x, y, 400)
        out.append(_trap(lr_curve(ts, kind, peak, W, T, alpha), ts))
    return np.array(out)

def fit(a, b, S, wd, kind):
    best = None
    span = b.max()
    for T0 in (np.geomspace(span * 0.6, span * 4, 10) if kind != "const" else [span * 40]):
        for W0 in (span * 1e-3, span * 1e-2, span * 0.05):
            x0 = [np.log(S.max() / wd / max(1e-9, (b - a).mean())), np.log(W0), np.log(T0), 0.1]
            f = lambda p: (interval_sum(a, b, kind, np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3] if kind != "const" else 1) * wd - S) / np.maximum(S, 1e-12)
            try:
                r = least_squares(f, x0, bounds=([-30, np.log(span * 1e-5), np.log(span * 0.1), 0], [5, np.log(span), np.log(span * 50), 1]))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    p = best.x
    return dict(kind=kind, peak=float(np.exp(p[0])), W=float(np.exp(p[1])), T=float(np.exp(p[2])),
                alpha=float(p[3]) if kind != "const" else 1.0, rel_rms=float(np.sqrt(2 * best.cost / len(S))))

def main(repo):
    d = json.load(open(R / ("series_" + repo.replace("/", "__") + ".json")))
    sp = SPEC[repo]
    ck = [c for c in d["checkpoints"] if "error" not in c]
    if sp.get("stages"):
        ck = [c for c in ck if c["key"][0] in sp["stages"]]
    st = [sp["step"](c["branch"]) for c in ck]
    rows = []
    for i in range(1, len(ck)):
        c = ck[i]
        if st[i] <= st[i - 1] or "P" not in c:
            continue
        P = c["P"]
        S = -math.log(P) if 0 < P < 1 else 0.0
        rows.append(dict(a=st[i - 1], b=st[i], branch=c["branch"], P=P, P_lo=c.get("P_q01"), P_hi=c.get("P_q99"),
                         sum_lr_wd=S, mean_lr=S / sp["wd"] / (st[i] - st[i - 1]), identical=c.get("identical")))
    a = np.array([r["a"] for r in rows], float); b = np.array([r["b"] for r in rows], float)
    S = np.array([r["sum_lr_wd"] for r in rows], float)
    total = float(S.sum())
    res = dict(repo=repo, wd=sp["wd"], unit=sp["unit"], intervals=rows, first=st[0], last=st[-1], n_intervals=len(rows),
               rows_used=d["rows_used"], total_sum_lr_wd=total, fits=[], published=[])
    for pub in sp["published"]:
        pred = interval_sum(a, b, pub["kind"], pub["peak"], pub["W"], pub["T"], pub["alpha"]) * sp["wd"]
        res["published"].append(dict(pub, predicted_total=float(pred.sum()), ratio_total=float(total / pred.sum()),
                                     rel_rms=float(np.sqrt(np.mean((S / pred - 1) ** 2))),
                                     median_ratio=float(np.median(S / pred))))
    for kind in ("cos", "lin", "const"):
        res["fits"].append(fit(a, b, S, sp["wd"], kind))
    fb = min(res["fits"], key=lambda f: f["rel_rms"])
    res["best"] = fb["kind"]
    if sp.get("peak_published"):
        res["steps_per_unit_if_published_peak"] = fb["peak"] / sp["peak_published"]
    json.dump(res, open(R / ("fit_" + repo.replace("/", "__") + ".json"), "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        (R / "figures").mkdir(exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot((a + b) / 2, S / sp["wd"] / (b - a), ".", ms=3, label="recovered from never-updated token vectors")
        tt = np.linspace(0, b.max(), 3000)
        for pub in res["published"]:
            plt.plot(tt, lr_curve(tt, pub["kind"], pub["peak"], pub["W"], pub["T"], pub["alpha"]), "--", lw=1, label="published: " + pub["src"][:60])
        plt.plot(tt, lr_curve(tt, fb["kind"], fb["peak"], fb["W"], fb["T"], fb["alpha"]), lw=1, label=f"best fit ({fb['kind']})")
        plt.xlabel(sp["unit"]); plt.ylabel(f"mean learning rate per {sp['unit'].split(' ')[0]} (wd={sp['wd']})"); plt.title(repo)
        plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(R / "figures" / (repo.replace("/", "__") + ".png"), dpi=120)
        plt.close()
    except Exception as e:
        print("plot failed", e)
    print(json.dumps({k: v for k, v in res.items() if k != "intervals"}, indent=1))

if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
