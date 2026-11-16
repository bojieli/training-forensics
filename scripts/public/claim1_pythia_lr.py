#!/usr/bin/env python3
"""Claim 1: recover Pythia's learning-rate schedule from never-trained embedding rows.

Mechanism: with decoupled weight decay (DeepSpeed FusedAdam, adam_w_mode), an
embedding row whose token never occurs gets no gradient and only shrinks by
(1 - lr_t * wd) each step.  The input embedding of Pythia is untied
(no-weight-tying: true), and gpt-neox pads the 50,277-token vocabulary to
50,304 (or more) rows, so rows >= 50277 are never looked up.  For two
checkpoints a < b, the least-squares slope of W_b on W_a over those rows is
prod_{t in (a,b]} (1 - lr_t wd)  ~=  exp(-wd * sum lr_t).

Outputs results/public/claim1_pythia_lr.json and figures/pythia_lr_schedule.png.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import json
import math
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402

OUT = ROOT / "results" / "public"
FIG = OUT / "figures"
EMB = "gpt_neox.embed_in.weight"
TOKENIZER_VOCAB = 50277          # len(tokenizer) for the Pile 20B tokenizer
T_TRAIN = 143000
WARM_FRAC = 0.01
SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
CFG_URL = "https://raw.githubusercontent.com/EleutherAI/pythia/main/models/{d}/pythia-{m}.yml"
CARD_LR = {"70m": 1.0e-3, "160m": 6.0e-4, "410m": 3.0e-4, "1b": 3.0e-4, "1.4b": 2.0e-4,
           "2.8b": 1.6e-4, "6.9b": 1.2e-4, "12b": 1.2e-4}   # README table on HF model cards
STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, T_TRAIN + 1, 1000))


def fetch_config(m: str) -> dict:
    d = m.upper()
    txt = requests.get(CFG_URL.format(d=d, m=m), timeout=60).text
    def num(key):
        mm = re.search(r'"%s"\s*:\s*([0-9.eE+-]+)' % re.escape(key), txt)
        return float(mm.group(1)) if mm else None
    return {"url": CFG_URL.format(d=d, m=m), "lr": num("lr"), "min_lr": num("min_lr"),
            "weight_decay": num("weight-decay"), "warmup": num("warmup"),
            "train_iters": num("train-iters"), "lr_decay_iters": num("lr-decay-iters"),
            "decay_style": re.search(r'"lr-decay-style"\s*:\s*"(\w+)"', txt).group(1),
            "optimizer": re.search(r'"type"\s*:\s*"(\w+)"', txt).group(1),
            "fp16": '"fp16": true' in txt}


# --- candidate schedules: lr used by update i (1-indexed) ------------------------------
# gpt-neox AnnealingLR: the scheduler is constructed with last_iter=0, so update i uses
# get_lr(num_iters=i-1).  W = warmup iters = 0.01*T.

def lr_pythia_era(n, T=T_TRAIN, W=None, clip=0.1):
    """gpt-neox learning_rates.py as of 2022 (commit d7af1e7a8e .. 0535bfbb78):
       n_ = min(n, T-W); warmup linear; lr = peak/2*(cos(pi*(n_-W)/T)+1); max(lr, min_lr)."""
    W = int(WARM_FRAC * T) if W is None else W
    n = np.asarray(n, dtype=np.float64)
    n_ = np.minimum(n, T - W)
    warm = n_ / W
    x = n_ - W
    cos = 0.5 * (np.cos(np.pi * x / T) + 1)
    return np.where(n <= W, warm, np.maximum(cos, clip))


def lr_neox_current(n, T=T_TRAIN, W=None, minr=0.1):
    """gpt-neox main (2023-08 onwards): min_lr + (peak-min_lr)*0.5(1+cos(pi*x/(T-W)))."""
    W = int(WARM_FRAC * T) if W is None else W
    n = np.asarray(n, dtype=np.float64)
    warm = np.maximum(1, n) / W
    x = np.minimum(n - W, T - W)
    cos = minr + (1 - minr) * 0.5 * (np.cos(np.pi * x / (T - W)) + 1)
    return np.where(n <= W, warm, np.maximum(cos, minr))


def lr_cos_to_zero(n, T=T_TRAIN, W=None):
    return lr_pythia_era(n, T, W, clip=0.0)


def lr_linear(n, T=T_TRAIN, W=None, minr=0.1):
    W = int(WARM_FRAC * T) if W is None else W
    n = np.asarray(n, dtype=np.float64)
    return np.where(n <= W, n / W, np.maximum(1 - (n - W) / (T - W), minr))


def lr_constant(n, T=T_TRAIN, W=None):
    W = int(WARM_FRAC * T) if W is None else W
    n = np.asarray(n, dtype=np.float64)
    return np.where(n <= W, n / W, 1.0)


CANDIDATES = {
    "pythia_era_cosine_to_0_clipped_at_0.1peak": lr_pythia_era,
    "neox_current_cosine_to_min_lr": lr_neox_current,
    "cosine_to_0_no_clip": lr_cos_to_zero,
    "linear_to_0.1peak": lr_linear,
    "constant_after_warmup": lr_constant,
}


def interval_sums(fn, steps, **kw):
    """sum of lr_i/peak over updates i in (a, b] for consecutive checkpoint steps."""
    i = np.arange(1, steps[-1] + 1)
    lr = fn(i - 1, **kw)
    cs = np.concatenate([[0.0], np.cumsum(lr)])
    return np.array([cs[b] - cs[a] for a, b in zip(steps[:-1], steps[1:])])


# --- measurement --------------------------------------------------------------------

ROWCACHE = paths.cache("trainforensics") / "pythia_rows"


def fetch_rows(repo, step, r0, r1):
    """Rows [r0, r1) of the input embedding at a step branch (cached as .pt so reruns
    need no HTTP at all; Pythia step branches ship a single model.safetensors)."""
    cp = ROWCACHE / f"{repo.replace('/', '__')}_step{step}_{r0}_{r1}.pt"
    if cp.exists():
        return torch.load(cp).double()
    try:
        W = remote.open_file(repo, "model.safetensors", f"step{step}").get(EMB, rows=(r0, r1))
    except Exception:  # noqa: BLE001  (sharded checkpoint)
        W = remote.tensor(repo, EMB, f"step{step}", rows=(r0, r1))
    ROWCACHE.mkdir(parents=True, exist_ok=True)
    torch.save(W.clone(), cp)
    return W.double()


def measure(m: str):
    repo = f"EleutherAI/pythia-{m}"
    try:
        dtype, shape = remote.open_file(repo, "model.safetensors", "step1000").info(EMB)
    except Exception:  # noqa: BLE001
        dtype, shape = remote.tensor_info(repo, EMB, "step1000")
    V = shape[0]
    r0 = TOKENIZER_VOCAB
    with ThreadPoolExecutor(6) as ex:
        Ws = list(ex.map(lambda s: fetch_rows(repo, s, r0, V), STEPS))
    # stored precision: count of mantissa bits actually used (fp32 storage)
    w = Ws[50].float().contiguous().view(torch.int32)
    low16_zero = float(((w & 0xFFFF) == 0).float().mean())
    out = []
    for (a, b), Wa, Wb in zip(zip(STEPS[:-1], STEPS[1:]), Ws[:-1], Ws[1:]):
        s_all = float((Wa * Wb).sum() / (Wa * Wa).sum())
        rows = ((Wa * Wb).sum(1) / (Wa * Wa).sum(1)).numpy()
        bitident = bool(torch.equal(Wa, Wb))
        se = float(rows.std(ddof=1) / math.sqrt(len(rows)))
        # per-coordinate relative residual (quantisation noise)
        resid = (Wb - s_all * Wa)
        rel = float((resid.norm() / (s_all * Wa).norm()))
        out.append({"a": a, "b": b, "slope": s_all, "slope_se_rows": se,
                    "neglog": -math.log(s_all), "neglog_se": se / s_all,
                    "rel_resid": rel, "bit_identical": bitident})
    # whole-run ratio
    W0, WT = Ws[0], Ws[-1]
    tot = float((W0 * WT).sum() / (W0 * W0).sum())
    rows_tot = ((W0 * WT).sum(1) / (W0 * W0).sum(1)).numpy()
    return {"repo": repo, "dtype": dtype, "shape": shape, "pad_rows": [r0, V],
            "n_pad_rows": V - r0, "n_coords": (V - r0) * shape[1],
            "fp32_values_with_low16_bits_zero": low16_zero,
            "intervals": out, "total_slope": tot,
            "total_slope_row_spread": [float(rows_tot.min()), float(rows_tot.max())],
            "total_neglog": -math.log(tot)}


def fit(meas, cfg):
    wd = cfg["weight_decay"]
    iv = meas["intervals"]
    steps = [iv[0]["a"]] + [d["b"] for d in iv]
    y = np.array([d["neglog"] for d in iv])
    # first-order product correction: -log prod(1-lr wd) = wd*sum lr + wd^2/2 sum lr^2 ...;
    # wd*lr <= 1e-4 so the correction is < 1e-4 relative -- ignored but reported.
    se = np.array([max(d["neglog_se"], 1e-7) for d in iv])
    res = {}
    for name, fn in CANDIDATES.items():
        S = interval_sums(fn, steps)
        w = 1 / se ** 2
        A = float((w * S * y).sum() / (w * S * S).sum())       # A = wd * peak
        r = y - A * S
        # use only post-warmup thousand-step intervals for an interpretable RMS
        mask = np.array([d["a"] >= 1000 for d in iv])
        rms = float(np.sqrt(np.mean((r[mask] / (A * 1000)) ** 2)))   # in units of peak lr
        chi2 = float(((r / se) ** 2).sum())
        res[name] = {"A_wd_times_peak": A, "peak_lr": A / wd,
                     "rms_resid_per_step_over_peak": rms, "chi2": chi2,
                     "max_abs_resid_over_peak": float(np.max(np.abs(r[mask]) / (A * 1000)))}
    # free-form: per-interval mean lr (post warmup) in units of the config peak
    lr_obs = (y / wd) / np.diff(steps)
    # free fit of the clipped-cosine family: peak, clip level, cosine horizon
    from scipy.optimize import least_squares
    mask = np.array([d["a"] >= 1000 for d in iv])
    def resid(p):
        peak, clip, T = p
        S = interval_sums(lambda n: lr_pythia_era(n, T=T, W=1430, clip=clip), steps)
        return ((y - wd * peak * S) / se)[mask]
    sol = least_squares(resid, x0=[cfg["lr"], 0.1, T_TRAIN], x_scale=[1e-4, 0.01, 1000])
    J = sol.jac
    try:
        cov = np.linalg.inv(J.T @ J) * max(1.0, (sol.fun ** 2).sum() / (mask.sum() - 3))
        perr = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        perr = [float("nan")] * 3
    peak, clip, T = sol.x
    # step at which the clip engages: 0.5(1+cos(pi x/T)) = clip, x = n - W
    t_clip = 1430 + T / math.pi * math.acos(max(-1.0, min(1.0, 2 * clip - 1)))
    free = {"peak_lr": float(peak), "peak_lr_err": float(perr[0]), "clip_frac": float(clip),
            "clip_frac_err": float(perr[1]), "cosine_horizon_T": float(T), "T_err": float(perr[2]),
            "clip_engages_at_step": float(t_clip), "flat_tail_steps": float(T_TRAIN - t_clip)}
    # direct (model-free) flat tail: intervals whose mean lr is within 1% of the final plateau
    plateau = np.median(lr_obs[-10:])
    flat = [d["a"] for d, l in zip(iv, lr_obs) if d["a"] >= 50000 and abs(l / plateau - 1) < 0.01]
    # first step from which every later interval is on the plateau
    first_flat = None
    for d, l in zip(iv, lr_obs):
        if d["a"] < 50000:
            continue
        later = [abs(ll / plateau - 1) < 0.01 for dd, ll in zip(iv, lr_obs) if dd["a"] >= d["a"]]
        if all(later):
            first_flat = d["a"]
            break
    # total budget: peak implied by whole-run ratio under the pythia-era shape
    S_tot = interval_sums(lr_pythia_era, [0, T_TRAIN])[0]
    peak_total = meas["total_neglog"] / wd / S_tot
    return {"candidates": res, "free_fit_clipped_cosine": free,
            "observed_mean_lr_per_interval": [{"a": d["a"], "b": d["b"], "lr": float(l)}
                                              for d, l in zip(iv, lr_obs)],
            "plateau_lr": float(plateau), "plateau_over_peak_fit": float(plateau / free["peak_lr"]),
            "model_free_flat_tail_starts_at_checkpoint": first_flat,
            "model_free_flat_tail_steps": (T_TRAIN - first_flat) if first_flat else None,
            "peak_from_whole_run_ratio_pythia_era_shape": float(peak_total),
            "sum_lr_over_peak_pythia_era_shape": float(S_tot)}


def plot(all_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = [m for m in all_res if "fit" in all_res[m]]
    fig, axes = plt.subplots(2, (len(ms) + 1) // 2, figsize=(4.2 * ((len(ms) + 1) // 2), 7.5),
                             sharex=True)
    axes = axes.flatten()
    n = np.arange(0, T_TRAIN)
    for ax, m in zip(axes, ms):
        f = all_res[m]["fit"]
        cfgp = all_res[m]["config"]["lr"]
        obs = [d for d in f["observed_mean_lr_per_interval"] if d["a"] >= 1000]
        x = [(d["a"] + d["b"]) / 2 for d in obs]
        ax.plot(x, [d["lr"] * 1e4 for d in obs], "o", ms=2.5, color="#222", label="recovered (pad rows)")
        pk = f["candidates"]["pythia_era_cosine_to_0_clipped_at_0.1peak"]["peak_lr"]
        ax.plot(n, lr_pythia_era(n) * pk * 1e4, color="#d55e00", lw=1.4,
                label="cos→0, clip 0.1·peak (fit)")
        pk2 = f["candidates"]["neox_current_cosine_to_min_lr"]["peak_lr"]
        ax.plot(n, lr_neox_current(n) * pk2 * 1e4, color="#0072b2", lw=1.2, ls="--",
                label="min_lr+cos (today's neox, fit)")
        ax.axhline(cfgp * 1e4, color="#999", lw=0.8, ls=":")
        ax.set_title(f"pythia-{m}: fit peak {pk:.3g} (config {cfgp:.2g}, card {CARD_LR[m]:.2g})",
                     fontsize=8.5)
        ax.set_ylabel("lr ×1e-4", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(ms):]:
        ax.axis("off")
    axes[0].legend(fontsize=6.5, loc="upper right")
    for ax in axes[-((len(ms) + 1) // 2):]:
        ax.set_xlabel("step", fontsize=8)
    fig.suptitle("Pythia lr schedules recovered from weight decay of never-used padding rows "
                 "(wd=0.1 assumed from configs)", fontsize=10)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "pythia_lr_schedule.png", dpi=140)


def never_trained_real_tokens(m="1b"):
    """Which *real* vocabulary rows (< 50277) were also never trained?  A row is
    'never trained' if W_T is exactly c * W_0 with the same c as the padding rows."""
    repo = f"EleutherAI/pythia-{m}"
    W0 = remote.open_file(repo, "model.safetensors", "step0").get(EMB).double()
    WT = remote.open_file(repo, "model.safetensors", f"step{T_TRAIN}").get(EMB).double()
    pad = slice(TOKENIZER_VOCAB, W0.shape[0])
    c = float((W0[pad] * WT[pad]).sum() / (W0[pad] ** 2).sum())
    resid = (WT - c * W0).norm(dim=1) / (c * W0).norm(dim=1)
    pad_resid = resid[pad]
    thr = float(pad_resid.max()) * 1.5
    ids = [int(i) for i in torch.nonzero(resid[:TOKENIZER_VOCAB] < thr).flatten()]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo)
    return {"model": repo, "c_pad": c, "pad_rel_resid_max": float(pad_resid.max()),
            "threshold": thr, "n_real_tokens_never_trained": len(ids),
            "ids": ids, "tokens": [tok.convert_ids_to_tokens(i) for i in ids],
            "median_rel_resid_trained_rows": float(resid[:TOKENIZER_VOCAB].median())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_res = {}
    sizes = sys.argv[1:] or SIZES
    prev = json.loads((OUT / "claim1_pythia_lr.json").read_text()) if (OUT / "claim1_pythia_lr.json").exists() else {}
    all_res.update({k: v for k, v in prev.items() if k in SIZES and "fit" in v})
    for m in sizes:
        cfg = fetch_config(m)
        try:
            meas = measure(m)
        except Exception as e:  # noqa: BLE001
            print(m, "measurement failed:", e)
            all_res[m] = {"config": cfg, "error": str(e)}
            continue
        f = fit(meas, cfg)
        all_res[m] = {"config": cfg, "model_card_lr": CARD_LR[m], "measure": meas, "fit": f}
        c = f["candidates"]
        ff = f["free_fit_clipped_cosine"]
        print(f"\n=== pythia-{m}  dtype={meas['dtype']} pad rows={meas['n_pad_rows']} "
              f"coords={meas['n_coords']} low16-zero={meas['fp32_values_with_low16_bits_zero']:.3f}")
        print(f"config lr={cfg['lr']} min_lr={cfg['min_lr']} wd={cfg['weight_decay']} "
              f"card lr={CARD_LR[m]}")
        print(f"whole-run slope W_143000/W_0 = {meas['total_slope']:.6f} "
              f"(per-row range {meas['total_slope_row_spread'][0]:.5f}..{meas['total_slope_row_spread'][1]:.5f}) "
              f"-> peak (pythia-era shape) = {f['peak_from_whole_run_ratio_pythia_era_shape']:.5e}")
        for k, v in c.items():
            print(f"  {k:45s} peak={v['peak_lr']:.5e}  rms/peak={v['rms_resid_per_step_over_peak']:.4f} "
                  f"max/peak={v['max_abs_resid_over_peak']:.4f}")
        print(f"  free fit: peak={ff['peak_lr']:.5e}±{ff['peak_lr_err']:.1e} clip={ff['clip_frac']:.4f}"
              f"±{ff['clip_frac_err']:.4f} T={ff['cosine_horizon_T']:.0f}±{ff['T_err']:.0f} "
              f"clip engages at {ff['clip_engages_at_step']:.0f} -> flat tail {ff['flat_tail_steps']:.0f} steps")
        print(f"  model-free: plateau lr={f['plateau_lr']:.4e} (= {f['plateau_over_peak_fit']:.4f} x peak); "
              f"flat (±1%) from checkpoint {f['model_free_flat_tail_starts_at_checkpoint']} "
              f"-> {f['model_free_flat_tail_steps']} steps")
        e = meas["intervals"]
        print("  early: " + ", ".join(f"{d['a']}->{d['b']}: {d['slope']:.7f}{' (bit-identical)' if d['bit_identical'] else ''}"
                                      for d in e[:6]))
    try:
        all_res["never_trained_real_tokens_1b"] = never_trained_real_tokens("1b")
        nt = all_res["never_trained_real_tokens_1b"]
        print(f"\npythia-1b real tokens (<50277) whose row is a pure rescale of init: "
              f"{nt['n_real_tokens_never_trained']}: {nt['tokens'][:40]}")
    except Exception as e:  # noqa: BLE001
        print("never-trained scan failed:", e)
    # scheduler source references
    all_res["scheduler_sources"] = {
        "current": "https://github.com/EleutherAI/gpt-neox/blob/main/megatron/learning_rates.py",
        "pythia_era_commit": "https://github.com/EleutherAI/gpt-neox/blob/d7af1e7a8e/megatron/learning_rates.py",
        "change_commit": "https://github.com/EleutherAI/gpt-neox/commit/009018e9ac (PR #1008, 2023-08-07)",
    }
    (OUT / "claim1_pythia_lr.json").write_text(json.dumps(all_res, indent=1))
    plot({k: v for k, v in all_res.items() if k in SIZES})
    print("\nwrote", OUT / "claim1_pythia_lr.json")


if __name__ == "__main__":
    main()
