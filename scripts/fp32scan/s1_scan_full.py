#!/usr/bin/env python3
"""s1 / s1.1 seed and batch-size scan, saved in full, with a bootstrap (tech review M10).

Same size model as s1_timing.py (single-document tokens; Adam on +1 gradients at each of a
token's touches; cosine, 5% warm-up, beta2 0.95: these three are ASSUMED from the published
train/sft.sh, not recovered). Sampler ASSUMED: accelerate's SeedableRandomSampler as used by
transformers' Trainer (accelerate >= 0.25 / transformers >= 4.36): the order of epoch e is
torch.randperm(n) with the generator seeded at seed + e. A different sampler or accelerate
version invalidates the seed reading.

Outputs
  results/fp32scan/<model>.seedscan.npz   every (seed, batch size) and its misfit (6,003 rows)
  results/fp32scan/<model>.seedscan.json  null spread, rank of seed 42, and a bootstrap over
                                          single-document tokens of (a) the best batch size
                                          and (b) whether seed 42 stays first

  python3 scripts/fp32scan/s1_scan_full.py simplescaling/s1-32B Qwen/Qwen2.5-32B-Instruct simplescaling/s1K_tokenized
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "published"))
import compare as C  # noqa: E402

child, parent, dsname = sys.argv[1:4]
EPOCHS = 5
W0 = C.get(parent, C.EMB).float()
W1 = C.get(child, C.PUB).float()
V = W1.shape[0]
same = (W0[:V] == W1).all(1).numpy()
from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
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
obs = torch.quantile((W1[torch.from_numpy(rare)] - W0[torch.from_numpy(rare)]).abs(), 0.999, dim=1).numpy().astype(np.float64)
del W0, W1
print("rare tokens", len(rare), flush=True)


def schedule(N, W):
    k = np.arange(N, dtype=np.float64)
    f = np.where(k < W, k / max(1, W), 1.0)
    p = np.clip((k - W) / max(1, N - W), 0, 1)
    return np.concatenate([[0.0], np.where(k < W, f, 0.5 * (1 + np.cos(np.pi * p)))])


def doc_steps(seed, bs):
    spe = math.ceil(n / bs)
    st = np.zeros((n, EPOCHS * 2), np.int64)
    cnt = np.zeros(n, np.int64)
    for e in range(EPOCHS):
        g = torch.Generator(); g.manual_seed(seed + e)
        perm = torch.randperm(n, generator=g).numpy()
        order = np.concatenate([perm, perm[:(-n) % bs]])
        for pos, d in enumerate(order):
            st[d, cnt[d]] = e * spe + pos // bs + 1
            cnt[d] += 1
    return st, cnt, spe * EPOCHS


def adam(touch, lr, b1=0.9, b2=0.95):
    R, T = touch.shape
    m = np.zeros(R); v = np.zeros(R); x = np.zeros(R)
    for t in range(1, T):
        g = touch[:, t]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (m / (1 - b1 ** t)) / np.sqrt(v / (1 - b2 ** t))
        x += lr[t] * np.where(v > 0, u, 0.0)
    return x


def resid(seed, bs, idx):
    st, cnt, N = doc_steps(seed, bs)
    lr = schedule(N, math.ceil(0.05 * N))
    touch = np.zeros((len(idx), N + 1))
    for j, i in enumerate(idx):
        d = rare_doc[i]
        touch[j, st[d, :cnt[d]]] = 1
    return np.log(obs[idx]) - np.log(adam(touch, lr))


def misfit(l):
    return float(np.median(np.abs(l - np.median(l))))


rng = np.random.default_rng(0)
sub = rng.choice(len(rare), size=min(600, len(rare)), replace=False)
rows = []
for bs in (8, 16, 32):
    for seed in range(0, 2001):
        rows.append((seed, bs, misfit(resid(seed, bs, sub))))
    print("bs", bs, "done", flush=True)
arr = np.array(rows, dtype=[("seed", "i8"), ("bs", "i8"), ("misfit", "f8")])
name = child.split("/")[-1]
out = ROOT / "results" / "fp32scan"
np.savez_compressed(out / f"{name}.seedscan.npz", scan=arr)
order = np.argsort(arr["misfit"])
rank42 = {int(bs): int(np.nonzero(arr[order]["seed"][arr[order]["bs"] == bs] == 42)[0][0]) + 1 for bs in (8, 16, 32)}
null = arr["misfit"][arr["seed"] != 42]
res = dict(child=child, rare_tokens=int(len(rare)), subsample=int(len(sub)), candidates=int(len(arr)),
           assumed=dict(schedule="cosine, 5% warm-up (published sft.sh)", beta2=0.95, epochs=EPOCHS,
                        sampler="accelerate SeedableRandomSampler: epoch e order = torch.randperm(n), generator seed = seed + e"),
           seed_range="0..2000", top=[dict(seed=int(r["seed"]), bs=int(r["bs"]), misfit=float(r["misfit"])) for r in arr[order][:10]],
           seed42_rank_within_batch_size=rank42,
           null=dict(median=float(np.median(null)), p01=float(np.quantile(null, 0.01)), min=float(null.min())))
# bootstrap over all single-document tokens: top-10 seeds per batch size from the scan
cand = []
for bs in (8, 16, 32):
    sel = arr[arr["bs"] == bs]
    cand += [(int(s), bs) for s in sel[np.argsort(sel["misfit"])][:10]["seed"]]
allidx = np.arange(len(rare))
L = {c: resid(c[0], c[1], allidx) for c in cand}
B = 300
best_bs, s42_first = [], 0
for b in range(B):
    ix = rng.integers(0, len(rare), len(rare))
    sc = {c: misfit(L[c][ix]) for c in cand}
    c0 = min(sc, key=sc.get)
    best_bs.append(c0[1])
    s42_first += int(c0[0] == 42)
res["bootstrap"] = dict(resamples=B, over="single-document tokens (all)", candidates="top 10 seeds per batch size",
                        best_batch_size_share={str(bs): best_bs.count(bs) / B for bs in (8, 16, 32)},
                        seed42_first_share=s42_first / B,
                        full_sample_misfit={f"{s}/{bs}": misfit(L[(s, bs)]) for (s, bs) in cand if s == 42})
json.dump(res, open(out / f"{name}.seedscan.json", "w"), indent=1)
print(json.dumps(res, indent=1), flush=True)
