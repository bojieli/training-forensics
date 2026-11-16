#!/usr/bin/env python3
"""Kimi-Dev-72B vs Qwen2.5-72B: what are the 2,171 'bit-identical' vectors and why does the
shrink fit give P ~ 0? Checks zeroed vectors, the vocabulary tail and overall similarity."""
import json, pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "roster"))
import sweep  # noqa
P = {p["key"]: p for p in json.load(open(ROOT / "results" / "large" / "pairs_large.json"))}
out = {}
for k in ["Qwen/Qwen2.5-72B@ -> moonshotai/Kimi-Dev-72B", "Qwen/Qwen2.5-72B-Instruct@ -> moonshotai/Kimi-Dev-72B"]:
    p = P[k]
    A = sweep.fetch(p["parent"]).float(); B = sweep.fetch(p["child"]).float()
    na, nb = A.norm(dim=1), B.norm(dim=1)
    same = (A == B).all(1)
    zb = (nb == 0)
    cos = torch.nn.functional.cosine_similarity(A, B, dim=1)
    ids = torch.nonzero(same).flatten()
    res = dict(identical_raw=int(same.sum()), zero_in_child=int(zb.sum()), zero_in_both=int((zb & (na == 0)).sum()),
               identical_id_range=[int(ids.min()), int(ids.max())] if len(ids) else None,
               cos_median=float(cos[~same].median()), norm_ratio_median=float((nb[~same] / na[~same]).median()),
               cos_p05=float(cos[~same].quantile(0.05)))
    out[k] = res
    print(k, res, flush=True)
json.dump(out, open(ROOT / "results" / "large" / "kimidev_check.json", "w"), indent=1)
