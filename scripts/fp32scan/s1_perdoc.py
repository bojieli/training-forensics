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



# per-document check with the recovered seed: for each document with >=3 rare tokens,
# is the seed-42 step pattern a better fit for ITS tokens than other seeds' patterns?
SEED, BS = int(sys.argv[6]), int(sys.argv[7])
others = [s for s in range(1000, 1500)]
def per_tok_dev(seed):
    st, cnt, N = doc_steps(seed, BS)
    lr = schedule("cosine", N, math.ceil(0.05 * N))
    touch = np.zeros((len(rare), N + 1))
    for j in range(len(rare)):
        d = rare_doc[j]
        touch[j, st[d, :cnt[d]]] = 1
    pred = adam_peaks(touch, lr)
    l = np.log(obs) - np.log(pred)
    return l - np.median(l)
d0 = np.abs(per_tok_dev(SEED))
docs_ = {}
for j, d in enumerate(rare_doc):
    docs_.setdefault(int(d), []).append(j)
elig = {d: js for d, js in docs_.items() if len(js) >= 3}
score0 = {d: float(np.median(d0[js])) for d, js in elig.items()}
better = {d: 0 for d in elig}
for s_ in others:
    dv = np.abs(per_tok_dev(s_))
    for d, js in elig.items():
        if np.median(dv[js]) <= score0[d]:
            better[d] += 1
rank = np.array([better[d] for d in elig])
res = dict(seed=SEED, bs=BS, docs_with_3_rare=len(elig), docs=n, other_seeds=len(others),
           docs_where_seed_beats_all_others=int((rank == 0).sum()),
           docs_where_seed_in_top_1pct=int((rank <= 5).sum()),
           docs_where_seed_in_top_5pct=int((rank <= 25).sum()))
print(json.dumps(res, indent=1))
json.dump(res, open(outp, "w"), indent=1)
