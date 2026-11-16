#!/usr/bin/env python3
"""Shape of the change for selected V4 named tokens vs reference tokens.

For each token: change = release - base (float32). Statistics of |change| across the
vector's numbers: median/mean ratio (constant magnitude -> 1.0, Gaussian -> 0.845),
excess kurtosis of the change, and the share of numbers whose |change| is within 2x of
the median. Reference: ordinary tokens with a similar change size in the same release,
plus a Gaussian with the same size. Also: pairwise cosine between the changes of the
named tokens (shared training contexts give correlated changes).
Writes results/frontier_ds/v4_change_shape.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ds_tokens import load, OUT  # noqa: E402

NAMES = {128829: "<｜action｜>", 128830: "<｜query｜>", 128831: "<｜authority｜>", 128832: "<｜domain｜>",
         128833: "<｜task｜>", 128834: "<｜political｜>", 128835: "<｜entity｜>", 128836: "<｜title｜>",
         128837: "<｜safety｜>", 128838: "<｜answer｜>", 128839: "<｜search｜>", 128825: "｜DSML｜",
         128826: "<｜begin▁sys｜>", 128828: "<｜latest_reminder｜>", 129279: "<｜image｜>"}


def stats(d):
    a = d.abs()
    med, mean = float(a.median()), float(a.mean())
    x = d - d.mean()
    kurt = float((x ** 4).mean() / (x ** 2).mean() ** 2 - 3)
    within2 = float(((a > med / 2) & (a < med * 2)).float().mean())
    return dict(median_over_mean=round(med / mean, 3), excess_kurtosis=round(kurt, 2), within_2x_of_median=round(within2, 3),
                change_norm=round(float(d.norm()), 4))


def main():
    out = {}
    for p, c in (("deepseek-ai/DeepSeek-V4-Pro-Base", "deepseek-ai/DeepSeek-V4-Pro"),
                 ("deepseek-ai/DeepSeek-V4-Flash-Base", "deepseek-ai/DeepSeek-V4-Flash"),
                 ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro-0813")):
        A, _ = load(p)
        B, _ = load(c)
        D = B.float() - A.float()
        dn = D.norm(dim=1)
        named = {NAMES[i]: stats(D[i]) for i in NAMES}
        # reference: ordinary tokens whose change size is within the named tokens' range
        lo, hi = 0.5 * min(v["change_norm"] for v in named.values()), 2 * max(v["change_norm"] for v in named.values())
        ids = torch.nonzero((dn > lo) & (dn < hi) & (torch.arange(len(dn)) < 128000)).flatten()
        g = torch.Generator().manual_seed(0)
        pick = ids[torch.randperm(len(ids), generator=g)[:500]]
        ref = [stats(D[i]) for i in pick]
        refsum = {k: [float(v) for v in np.percentile([r[k] for r in ref], [5, 50, 95])] for k in ref[0]}
        gauss = stats(torch.randn(D.shape[1], generator=g) * 0.0005)
        ni = list(NAMES)
        Dn = D[ni] / D[ni].norm(dim=1, keepdim=True)
        C = (Dn @ Dn.T).numpy()
        pr = D[pick[:200]] / D[pick[:200]].norm(dim=1, keepdim=True)
        Cr = (pr @ pr.T).numpy()
        out[f"{p.split('/')[1]} -> {c.split('/')[1]}"] = dict(
            named=named, reference_ordinary_same_change_size=dict(n=int(len(ids)), sampled=len(pick), p5_50_95=refsum),
            gaussian=gauss,
            named_change_pairwise_cos={f"{NAMES[a]} | {NAMES[b]}": round(float(C[i, j]), 3)
                                       for i, a in enumerate(ni) for j, b in enumerate(ni) if j > i},
            reference_change_pairwise_cos_p5_50_95=[float(v) for v in np.percentile(Cr[np.triu_indices(200, 1)], [5, 50, 95])])
        print(c, json.dumps(out[list(out)[-1]]["reference_ordinary_same_change_size"]), flush=True)
    json.dump(out, open(OUT / "v4_change_shape.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
