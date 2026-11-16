#!/usr/bin/env python3
"""Lineage of a Llama-70B-family release against Llama 3 / 3.1 / 3.3 base and instruct.

  python3 scripts/fp32scan/lineage70b.py OUT.json CHILD[@rev] [CHILD2[@rev] ...]

Per child:
  - storage: share of values that sit on the 16-bit (bf16) grid
  - per candidate C: bit-identical vectors, median relative distance, and for
    every vector which candidate is nearest
  - direction test: cosine between (child - Llama-3.1 base) and
    (candidate instruct - Llama-3.1 base), per vector; an instruct-derived model
    shares the instruct tune's direction, a base-derived one does not
  - chat tokens that are ~zero in the base models: their sizes in every model
  - reserved (never-used) tokens: coordinate-wise ratio child / candidate
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "reflection_check"))
from pins import PIN  # noqa: E402  pinned revisions for children and references

EMB = "model.embed_tokens.weight"
CANDS = {
    "llama-3.1-70b": "unsloth/Meta-Llama-3.1-70B",
    "llama-3.1-70b-instruct": "unsloth/Meta-Llama-3.1-70B-Instruct",
    "llama-3.3-70b-instruct": "unsloth/Llama-3.3-70B-Instruct",
    "llama-3-70b": "NousResearch/Meta-Llama-3-70B",
    "llama-3-70b-instruct": "NousResearch/Meta-Llama-3-70B-Instruct",
}
CHAT = {128006: "<|start_header_id|>", 128007: "<|end_header_id|>", 128009: "<|eot_id|>",
        128000: "<|begin_of_text|>", 128001: "<|end_of_text|>"}


def load(spec, keep16=False):
    repo, _, rev = spec.partition("@")
    t = remote.tensor(repo, EMB, rev or PIN.get(repo))
    return t if keep16 else t.float()


out = sys.argv[1]
C = {k: load(v, keep16=True)[:128256] for k, v in CANDS.items()}   # bf16, upcast per chunk
V = 128256
CH = 8192
res = dict(candidates=CANDS, children={})
norms = {k: torch.cat([C[k][a:a + CH].float().norm(dim=1) for a in range(0, V, CH)]) for k in C}
res["chat_token_norms"] = {CHAT[i]: {k: float(norms[k][i]) for k in C} for i in CHAT}
res["candidate_pairs_rel"] = {}
for k in C:
    if k != "llama-3.1-70b":
        d = torch.cat([((C[k][a:a + CH].float() - C["llama-3.1-70b"][a:a + CH].float()).norm(dim=1)) for a in range(0, V, CH)])
        res["candidate_pairs_rel"][k + " vs llama-3.1-70b"] = float((d / norms["llama-3.1-70b"].clamp_min(1e-30)).median())
reserved = torch.nonzero(norms["llama-3.1-70b"] < 1e-3).flatten()
res["reserved_count"] = int(len(reserved))
ok = norms["llama-3.1-70b"] > 1e-3
keys = list(C)
for spec in sys.argv[2:]:
    Wfull = load(spec, keep16=True)
    Vc = Wfull.shape[0]
    r = dict(rows=int(Vc), stored_dtype=str(Wfull.dtype))
    on_grid = 0
    dist = {k: torch.zeros(V) for k in keys}
    ident = {k: 0 for k in keys}
    dcos = {k: torch.zeros(V, dtype=torch.float64) for k in keys if k != "llama-3.1-70b"}
    for a in range(0, V, CH):
        W = Wfull[a:min(a + CH, V)].float()
        if Wfull.dtype == torch.float32:
            on_grid += int(((W.view(torch.int32) & 0xFFFF) == 0).sum())
        base = C["llama-3.1-70b"][a:a + CH].float()
        dchild = (W - base).double()
        for k in keys:
            c = C[k][a:a + CH].float()
            dist[k][a:a + CH] = (W - c).norm(dim=1) / c.norm(dim=1).clamp_min(1e-30)
            ident[k] += int((W == c).all(1).sum())
            if k != "llama-3.1-70b":
                dcos[k][a:a + CH] = torch.nn.functional.cosine_similarity(dchild, (c - base).double(), dim=1)
    r["share_values_on_bf16_grid"] = on_grid / (V * Wfull.shape[1]) if Wfull.dtype == torch.float32 else None
    r["per_candidate"] = {k: dict(bit_identical=ident[k], rel_dist_median=float(dist[k][ok].median())) for k in keys}
    D = torch.stack([dist[k] for k in keys], 1)[ok]
    nearest = D.argmin(1)
    r["nearest_candidate_share"] = {keys[i]: float((nearest == i).float().mean()) for i in range(len(keys))}
    r["direction_cos_median"] = {k + " - llama-3.1-70b": float(dcos[k][ok].median()) for k in dcos}
    r["chat_token_norms"] = {CHAT[i]: float(Wfull[i].float().norm()) for i in CHAT}
    for k in ("llama-3.1-70b", "llama-3.1-70b-instruct"):
        a_, b_ = C[k][reserved].double(), Wfull[reserved].double()
        nz = a_.abs() > 0
        ratio = b_[nz] / a_[nz]
        r.setdefault("reserved_ratio", {})[k] = dict(median=float(ratio.median()), p01=float(ratio.quantile(0.01)),
                                                     p99=float(ratio.quantile(0.99)),
                                                     bit_identical_vectors=int((C[k][reserved].float() == Wfull[reserved].float()).all(1).sum()))
    if Vc > V:
        r["added_vectors_norms"] = Wfull[V:].float().norm(dim=1).tolist()
    res["children"][spec] = r
    print(spec, json.dumps(r, indent=1), flush=True)
    del Wfull
print(json.dumps({k: res[k] for k in ("chat_token_norms", "candidate_pairs_rel", "reserved_count")}, indent=1))
json.dump(res, open(out, "w"), indent=1)
