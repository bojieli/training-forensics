#!/usr/bin/env python3
"""Claim 6: Llama-4 Scout special/reserved tokens that are zero (or near zero) in the base and
non-zero in the Instruct release. Tests whether the new vectors look like Adam updates from zero
(every number about the same size: |x| nearly constant, ratio std(|x|)/mean(|x|) small) or like a
random Gaussian start (ratio ~0.76). Writes results/frontier/scout.json."""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import OUT, DT, changed_fraction, is_placeholder, pair, records, state  # noqa: E402


def main():
    recs = records()
    W0, W1, r = pair("unsloth/Llama-4-Scout-17B-16E", "unsloth/Llama-4-Scout-17B-16E-Instruct", recs)
    P = r["embed"]["P"]
    d = W0.shape[1]
    sp = [t for t in r["special"] if t["id"] < W0.shape[0]]
    ids = [t["id"] for t in sp]
    frac = changed_fraction(W0[ids], W1[ids], P, W1.dtype)
    n0 = W0[ids].float().norm(dim=1).numpy()
    n1 = W1[ids].float().norm(dim=1).numpy()
    A = W1[ids].double().abs()
    flatness = (A.std(1) / A.mean(1).clamp_min(1e-30)).numpy()
    zero0 = (W0[ids].float().abs().max(1).values == 0).numpy()
    rows = []
    for k, t in enumerate(sp):
        rows.append(dict(id=t["id"], content=t["content"], placeholder=is_placeholder(t["content"]),
                         state=state(frac[k], d), frac=float(frac[k]), base_norm=float(n0[k]), release_norm=float(n1[k]),
                         base_all_zero=bool(zero0[k]), release_abs_spread=float(flatness[k])))
    grew = [x for x in rows if x["base_norm"] < 0.01 and x["release_norm"] > 0.01]
    # pairwise cosine of grown vectors
    gi = [x["id"] for x in grew]
    U = W1[gi].double()
    U = U / U.norm(dim=1, keepdim=True)
    C = (U @ U.T).numpy()
    off = C[~np.eye(len(gi), dtype=bool)] if len(gi) > 1 else np.array([np.nan])
    summ = dict(P=P, n_special=len(rows),
                states={s: sum(x["state"] == s for x in rows) for s in ("bit-identical", "barely changed", "partly changed", "updated")},
                base_zero_vectors=int(zero0.sum()), base_near_zero=int(sum(x["base_norm"] < 0.01 for x in rows)),
                grew_from_near_zero=len(grew), grew_named=[x["content"] for x in grew if not x["placeholder"]],
                grew_placeholders=[x["content"] for x in grew if x["placeholder"]],
                grew_release_norm_median=float(np.median([x["release_norm"] for x in grew])) if grew else None,
                grew_abs_spread_median=float(np.median([x["release_abs_spread"] for x in grew])) if grew else None,
                grew_pairwise_cos_median=float(np.median(off)), grew_pairwise_cos_abs_mean=float(np.mean(np.abs(off))),
                gaussian_abs_spread_reference=0.7555)
    json.dump(dict(summary=summ, tokens=rows), open(OUT / "scout.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
