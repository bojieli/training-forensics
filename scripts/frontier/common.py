"""Shared helpers for the frontier checks: sweep records, cached embeddings, per-token change stats."""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import glob
import json
import os
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = pathlib.Path(os.environ.get("CKPTF_CACHE", paths.cache("trainforensics") / "emb"))
OUT = ROOT / "results" / "frontier"
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def records():
    recs = {}
    for fn in ("sweep.jsonl", "sweep_cml3.jsonl"):
        p = ROOT / "results" / "roster" / fn
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok" and r["key"] not in recs:
                recs[r["key"]] = r
    return recs


def rec_for(parent, child, recs=None):
    for r in (recs or records()).values():
        if r["parent"] == parent and r["child"] == child:
            return r
    raise KeyError(f"{parent} -> {child}")


def load(repo, shape, dtype):
    pref = repo.replace("/", "__") + "@"
    need = shape[0] * shape[1] * torch.finfo(dtype).bits // 8
    for f in sorted(glob.glob(str(CACHE / (pref + "*")))):
        tail = f.split("__")[-2]
        if "lm_head" in tail or tail.startswith("head") or "output" in tail:
            continue
        if os.path.getsize(f) == need:
            return torch.frombuffer(bytearray(open(f, "rb").read()), dtype=dtype).view(*shape)
    raise FileNotFoundError(f"{repo} {shape}")


def pair(parent, child, recs=None):
    """(W0, W1, record) restricted to the common vocabulary rows."""
    r = rec_for(parent, child, recs)
    dt = DT[r["dtype_child"]]
    W0 = load(parent, r["shape_parent"], DT[r["dtype_parent"]])
    W1 = load(child, r["shape_child"], dt)
    n = min(W0.shape[0], W1.shape[0])
    return W0[:n], W1[:n], r


def changed_fraction(W0, W1, P, dtype, chunk=8192):
    """Per token vector: fraction of its numbers that differ from round(P * base)."""
    V = W0.shape[0]
    out = np.zeros(V)
    for a in range(0, V, chunk):
        pred = (W0[a:a + chunk].double() * P).to(torch.float32).to(dtype)
        out[a:a + chunk] = (pred != W1[a:a + chunk].to(dtype)).double().mean(1).numpy()
    return out


def norms(W):
    return W.float().norm(dim=1).numpy()


def is_placeholder(s):
    s = (s or "").lower()
    return any(k in s for k in ("place▁holder", "placeholder", "reserved", "<unused", "<|unused"))


def state(frac, d):
    """Plain-language state of one token vector from its changed fraction."""
    if frac == 0:
        return "bit-identical"
    if frac <= max(2, 0.01 * d) / d:
        return "barely changed"
    if frac < 0.10:
        return "partly changed"
    return "updated"
