#!/usr/bin/env python3
"""Claim 4 (first look): is Adam's sign-like signature visible in Qwen2.5-7B-Instruct?

Background: a row that receives exactly one gradient g at some step and none after
moves, under Adam(beta1=0.9, beta2=0.999) without weight decay, by
  lr * sum_k (1-b1) b1^k / sqrt((1-b2) b2^k)  ~= 31.8 * lr   in *every* coordinate,
with the sign of -g (m/sqrt(v) = sign(g) * const).  So |delta| is constant across
coordinates within the row (CV ~ 0).  Rows touched many times have varying |delta|.

Complication: weights are stored in bf16 (8 significant bits).  A coordinate with
|w| has ulp(w) = 2^(floor(log2|w|) - 7); updates smaller than ~ulp/2 vanish and
larger ones are quantised.  We therefore compute the per-row CV of |delta| over
(a) all coordinates, and (b) only coordinates whose ulp is < 1/16 of the row's
median nonzero |delta| ("fine-grid" coordinates), where quantisation is negligible.

Uses the 7B embedding pair cached by claim3 (or fetches it).
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import json
import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402

OUT = ROOT / "results" / "public"
CACHE = paths.cache("trainforensics")
EMB = "model.embed_tokens.weight"


def single_touch_factor(b1=0.9, b2=0.999, n=200000, bias_correction_at=None):
    k = np.arange(n)
    return float(((1 - b1) * b1 ** k / np.sqrt((1 - b2) * b2 ** k)).sum())


def ulp_bf16(w):
    e = torch.floor(torch.log2(w.abs().clamp_min(1e-30)))
    return torch.pow(2.0, e - 7)


def main():
    cp = CACHE / "qwen25_7b_embed_pair.pt"
    if cp.exists():
        d = torch.load(cp)
        Wb, Wi = d["base"], d["inst"]
    else:
        Wb = remote.tensor("Qwen/Qwen2.5-7B", EMB)
        Wi = remote.tensor("Qwen/Qwen2.5-7B-Instruct", EMB)
    n_vocab = 151643
    Wb, Wi = Wb[:n_vocab].float(), Wi[:n_vocab].float()
    D = Wi - Wb
    moved = (D != 0).any(1)
    rel = D.norm(dim=1) / Wb.norm(dim=1)
    absd = D.abs()
    nz = (absd > 0)
    frac_nz = nz.float().mean(1)
    # per-row median nonzero |delta|
    med = torch.where(nz, absd, torch.nan).nanmedian(dim=1).values
    ulp = ulp_bf16(Wb)
    cv_all = absd.std(1) / absd.mean(1).clamp_min(1e-30)
    fine = ulp < (med[:, None] / 16)
    fa = torch.where(fine, absd, torch.nan)
    mu_f = fa.nanmean(1)
    sd_f = torch.sqrt(torch.nanmean((fa - mu_f[:, None]) ** 2, dim=1))
    n_fine = fine.sum(1)
    cv_fine = sd_f / mu_f
    # sign-like test on fine coordinates: fraction with |delta| within 25% of row median
    near = ((fa - med[:, None]).abs() < 0.25 * med[:, None]) & fine
    frac_near = near.sum(1) / n_fine.clamp_min(1)

    idx = torch.nonzero(moved).flatten()
    q = torch.quantile(rel[idx], torch.tensor([0.1, 0.5, 0.9]))
    groups = {
        "least_moved_10pct": idx[rel[idx] <= q[0]],
        "median_band_45_55pct": idx[(rel[idx] >= torch.quantile(rel[idx], 0.45)) &
                                    (rel[idx] <= torch.quantile(rel[idx], 0.55))],
        "most_moved_10pct": idx[rel[idx] >= q[2]],
    }

    def summ(ids):
        ok = ids[n_fine[ids] >= 64]
        cf = cv_fine[ok].numpy()
        ca = cv_all[ids].numpy()
        return {"n_rows": int(len(ids)), "n_rows_with_>=64_fine_coords": int(len(ok)),
                "rel_change_median": float(rel[ids].median()),
                "frac_coords_changed_median": float(frac_nz[ids].median()),
                "median_abs_delta_median": float(med[ids].nanmedian()),
                "cv_all_coords_quantiles": {p: float(np.quantile(ca, p)) for p in (0.1, 0.5, 0.9)},
                "cv_fine_coords_quantiles": {p: float(np.nanquantile(cf, p)) for p in (0.1, 0.5, 0.9)}
                if len(cf) else None,
                "frac_fine_coords_within_25pct_of_median_quantiles":
                    {p: float(np.quantile(frac_near[ok].numpy(), p)) for p in (0.1, 0.5, 0.9)}
                    if len(ok) else None}

    # reference: CV of |x| for x ~ Gaussian is sqrt(pi/2 - 1) = 0.756; for a constant it is 0
    res = {"model_pair": ["Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"],
           "n_vocab_rows": n_vocab, "n_moved_rows": int(moved.sum()),
           "single_touch_factor_b2_0.999": single_touch_factor(),
           "single_touch_factor_b2_0.95": single_touch_factor(b2=0.95),
           "reference_cv_gaussian_abs": math.sqrt(math.pi / 2 - 1),
           "groups": {k: summ(v) for k, v in groups.items()}}
    # implied lr if the least-moved rows were single-touch with beta2=0.999
    lm = res["groups"]["least_moved_10pct"]["median_abs_delta_median"]
    res["implied_lr_if_single_touch_0.999"] = lm / res["single_touch_factor_b2_0.999"]
    # histogram of per-row median |delta| for the least-moved decile
    lmids = groups["least_moved_10pct"]
    h, edges = np.histogram(np.log10(med[lmids].numpy()), bins=30)
    res["least_moved_log10_median_abs_delta_hist"] = {"counts": h.tolist(), "edges": edges.tolist()}
    # a few example least-moved rows
    from huggingface_hub import hf_hub_download
    vocab = json.loads(pathlib.Path(hf_hub_download("Qwen/Qwen2.5-7B", "tokenizer.json")).read_text())["model"]["vocab"]
    inv = {v: k for k, v in vocab.items()}
    ex = lmids[torch.argsort(rel[lmids])][:: max(1, len(lmids) // 15)][:15]
    res["examples_least_moved"] = [{"id": int(i), "token": inv.get(int(i)), "rel": float(rel[i]),
                                    "frac_coords_changed": float(frac_nz[i]),
                                    "median_abs_delta": float(med[i]), "cv_fine": float(cv_fine[i]),
                                    "n_fine": int(n_fine[i])} for i in ex]
    print(json.dumps(res, indent=1)[:6000])
    (OUT / "claim4_qwen25_adam_signature.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
