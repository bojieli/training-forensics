#!/usr/bin/env python3
"""Claim 4: why Qwen3-30B-A3B shows its vision/grounding tokens updated while 8B/14B leave them
bit-identical (after removing the weight-decay shrink). Compares the size of the change per token
vector against tokens surely in the data (chat markers) and ordinary tokens.
Writes results/frontier/qwen3_vision.json."""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import OUT, DT, pair, records  # noqa: E402

PAIRS = [("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"), ("Qwen/Qwen3-14B-Base", "Qwen/Qwen3-14B"),
         ("Qwen/Qwen3-30B-A3B-Base", "Qwen/Qwen3-30B-A3B"),
         ("Qwen/Qwen3-30B-A3B-Base", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
         ("Qwen/Qwen3-30B-A3B-Base", "Qwen/Qwen3-30B-A3B-Thinking-2507")]
VISION = list(range(151646, 151657))
CHAT = [151644, 151645, 151667, 151668]


def main():
    recs = records()
    out = {}
    for par, ch in PAIRS:
        try:
            W0, W1, r = pair(par, ch, recs)
        except Exception as e:
            out[f"{par} -> {ch}"] = dict(error=str(e)); continue
        P = r["embed"]["P"]
        pred = (W0.double() * P)
        D = (W1.double() - pred)
        med = D.abs().median(dim=1).values.numpy()
        mean_shift = D.mean(1).numpy()
        cos = torch.nn.functional.cosine_similarity(D[VISION], D[VISION].mean(0, keepdim=True)).numpy()
        rng = np.random.default_rng(0)
        ordi = rng.choice(np.arange(0, 150000), 3000, replace=False)
        trained = ordi[med[ordi] > 0]
        res = dict(P=P, fit=r["embed"]["P_exact_frac"],
                   vision_median_abs_change=float(np.median(med[VISION])),
                   chat_median_abs_change=float(np.median(med[CHAT])),
                   ordinary_trained_median_abs_change=float(np.median(med[trained])) if len(trained) else None,
                   vision_change_cos_to_their_mean=float(np.median(cos)),
                   vision_change_mean_component=float(np.median(np.abs(mean_shift[VISION]))),
                   vision_vs_base_cos=float(torch.nn.functional.cosine_similarity(D[VISION], W0[VISION].double()).median()),
                   )
        # are the vision-token changes nearly the same vector (a shared update, e.g. one common direction)?
        Dv = D[VISION]
        U = Dv / Dv.norm(dim=1, keepdim=True).clamp_min(1e-30)
        C = (U @ U.T).numpy()
        res["vision_pairwise_change_cos_median"] = float(np.median(C[~np.eye(len(VISION), dtype=bool)]))
        Dc = D[CHAT]
        Uc = Dc / Dc.norm(dim=1, keepdim=True).clamp_min(1e-30)
        res["chat_pairwise_change_cos_median"] = float(np.median((Uc @ Uc.T).numpy()[~np.eye(len(CHAT), dtype=bool)]))
        res["vision_to_chat_change_cos_median"] = float(np.median((U @ Uc.T).numpy()))
        out[f"{par} -> {ch}"] = res
        print(par, "->", ch, json.dumps(res), flush=True)
        json.dump(out, open(OUT / "qwen3_vision.json", "w"), indent=1)


if __name__ == "__main__":
    main()
