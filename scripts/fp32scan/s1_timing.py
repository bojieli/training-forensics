#!/usr/bin/env python3
"""Timing stage for s1-style releases: seed scan over thousands of seeds, batch
size, and a schedule fit with the seed fixed. Vectorised Adam over tokens.

  python3 scripts/fp32scan/s1_timing.py CHILD PARENT DATASET OUT.json [EPOCHS]
"""
from __future__ import annotations

import json
import math
import sys
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402

child, parent, dsname, outp = sys.argv[1:5]
EPOCHS = int(sys.argv[5]) if len(sys.argv) > 5 else 5
EMB = "model.embed_tokens.weight"
W0 = remote.tensor(parent, EMB).float()
W1 = remote.tensor(child, EMB).float()
V = W1.shape[0]
same = (W0 == W1).all(1).numpy()

from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(parent)
docs = [tok(t, add_special_tokens=False)["input_ids"][:32768] for t in load_dataset(dsname)["train"]["text"]]
n = len(docs)
cnt_docs = np.zeros(V, np.int64)
for ids in docs:
    cnt_docs += np.bincount(np.unique(np.asarray(ids[:-1])), minlength=V)[:V]
doc_of = {}
for di, ids in enumerate(docs):
    for t in set(ids[:-1]):
        if cnt_docs[t] == 1:
            doc_of[t] = di
rare = np.array(sorted(t for t in doc_of if not same[t]), dtype=np.int64)
rare_doc = np.array([doc_of[int(t)] for t in rare])
absd = (W1[torch.from_numpy(rare)] - W0[torch.from_numpy(rare)]).abs()
obs = torch.quantile(absd, 0.999, dim=1).numpy().astype(np.float64)
del W0, W1
res = dict(child=child, parent=parent, rare_tokens=int(len(rare)), docs=n, epochs=EPOCHS)
print("rare tokens", len(rare), flush=True)


def schedule(kind, N, W, peak=1.0, min_ratio=0.0):
    k = np.arange(N, dtype=np.float64)              # HF LambdaLR: k-th update uses factor(k)
    f = np.where(k < W, k / max(1, W), 1.0)
    p = np.clip((k - W) / max(1, N - W), 0, 1)
    if kind == "cosine":
        g = 0.5 * (1 + np.cos(np.pi * p))
    elif kind == "linear":
        g = 1 - p
    else:
        g = np.ones_like(p)
    lr = np.where(k < W, f, g) * peak
    return np.concatenate([[0.0], lr])


def doc_steps(seed, bs):
    spe = math.ceil(n / bs)
    st = np.zeros((n, EPOCHS * 2), np.int64)   # up to 2 uses per epoch (padding duplicates)
    cnt = np.zeros(n, np.int64)
    for e in range(EPOCHS):
        g = torch.Generator()
        g.manual_seed(seed + e)
        perm = torch.randperm(n, generator=g).numpy()
        order = np.concatenate([perm, perm[:(-n) % bs]])
        for pos, d in enumerate(order):
            st[d, cnt[d]] = e * spe + pos // bs + 1
            cnt[d] += 1
    return st, cnt, spe * EPOCHS


def adam_peaks(touch, lr, b1=0.9, b2=0.95):
    """touch: [R, N+1] 0/1; returns the move of a coordinate whose gradient is +1 at each touch."""
    R, T = touch.shape
    m = np.zeros(R)
    v = np.zeros(R)
    x = np.zeros(R)
    for t in range(1, T):
        g = touch[:, t]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (m / (1 - b1 ** t)) / np.sqrt(v / (1 - b2 ** t))
        x += lr[t] * np.where(v > 0, u, 0.0)
    return x


def score(seed, bs, kind="cosine", W=None, sub=None, b2=0.95):
    st, cnt, N = doc_steps(seed, bs)
    if W is None:
        W = math.ceil(0.05 * N)
    lr = schedule(kind, N, W)
    idx = np.arange(len(rare)) if sub is None else sub
    touch = np.zeros((len(idx), N + 1))
    for j, i in enumerate(idx):
        d = rare_doc[i]
        touch[j, st[d, :cnt[d]]] = 1
    pred = adam_peaks(touch, lr, b2=b2)
    l = np.log(obs[idx]) - np.log(pred)
    s = float(np.median(l))
    dev = np.abs(l - s)
    return dict(seed=seed, bs=bs, kind=kind, N=N, W=W, b2=b2, peak=math.exp(s), med_dev=float(np.median(dev)),
                within_2pct=float((dev < 0.02).mean()), within_5pct=float((dev < 0.05).mean()))


rng = np.random.default_rng(0)
sub = rng.choice(len(rare), size=min(600, len(rare)), replace=False)
# 1. seed x batch-size scan (published schedule shape: cosine, 5% warm-up)
scan = []
for bs in (8, 16, 32):
    for seed in range(0, 2001):
        scan.append(score(seed, bs, sub=sub))
    print("bs", bs, "done", flush=True)
scan.sort(key=lambda r: r["med_dev"])
devs = np.array([r["med_dev"] for r in scan])
res["seed_scan"] = dict(seeds=2001, batch_sizes=[8, 16, 32], top=scan[:8],
                        median_dev=float(np.median(devs)), p01_dev=float(np.quantile(devs, 0.01)))
best = scan[0]
print(json.dumps(res["seed_scan"]["top"][:3]), flush=True)
# 2. schedule fit with the seed fixed
fits = []
for kind in ("cosine", "linear", "constant"):
    for W in range(0, 41):
        fits.append(score(best["seed"], best["bs"], kind=kind, W=W, sub=sub))
for b2 in (0.95, 0.98, 0.99, 0.999):
    fits.append(score(best["seed"], best["bs"], kind="cosine", W=math.ceil(0.05 * best["N"]), sub=sub, b2=b2))
fits.sort(key=lambda r: r["med_dev"])
res["schedule_fit_top"] = fits[:8]
res["final"] = score(fits[0]["seed"], fits[0]["bs"], kind=fits[0]["kind"], W=fits[0]["W"], b2=fits[0]["b2"])
# 3. per-document step placement: documents with >=3 rare tokens whose sizes agree with the best seed
st, cnt, N = doc_steps(best["seed"], best["bs"])
res["placement"] = dict(note="every document's steps follow from the recovered seed and batch size; "
                        "documents with a rare token are directly checkable",
                        docs_with_rare_token=int(len(set(rare_doc.tolist()))), docs=n)
json.dump(res, open(outp, "w"), indent=1)
print(json.dumps({k: res[k] for k in ("schedule_fit_top", "final", "placement")}, indent=1), flush=True)
