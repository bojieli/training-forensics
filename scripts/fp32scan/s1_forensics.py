#!/usr/bin/env python3
"""Blind-then-check forensics of an s1-style public fine-tune (fp32 release, bf16 base).

  python3 scripts/fp32scan/s1_forensics.py CHILD PARENT DATASET OUT.json

Stages (the published recipe is used only in the 'check' stages, and says so):
  1. token vectors: bit-identical to the base (upcast) vs changed
  2. token set: changed vectors vs tokens present in the published training text
  3. weight decay on never-updated vectors
  4. timing: per-token change sizes vs the published schedule, and the shuffle
     seed / training step of every document (HF Trainer + accelerate convention:
     epoch e order = torch.randperm(n, generator seeded seed+e), global batch =
     consecutive blocks of `bs`, last batch padded from the start of the order)
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
BS = int(sys.argv[5]) if len(sys.argv) > 5 else 16
EPOCHS = int(sys.argv[6]) if len(sys.argv) > 6 else 5
BLOCK = 32768
EMB = "model.embed_tokens.weight"
res = dict(child=child, parent=parent, dataset=dsname)

W0 = remote.tensor(parent, EMB).float()
W1 = remote.tensor(child, EMB).float()
V = W1.shape[0]
assert W0.shape == W1.shape
same = (W0 == W1).all(1).numpy()
D = (W1 - W0)
absd = D.abs()
frac_changed = (absd > 0).float().mean(1).numpy()
res["vectors"] = dict(total=int(V), bit_identical=int(same.sum()), changed=int((~same).sum()))
# never-updated vectors: is there any multiplicative shrink?
if same.sum() == 0:
    r = ((W0 * W1).sum(1) / (W0 * W0).sum(1).clamp_min(1e-30)).numpy()
    res["shrink_median"] = float(np.median(r))
# changed vectors that sit on the 16-bit grid would mean a 16-bit save; check low bits
low = (W1.view(torch.int32) & 0xFFFF)
res["changed_vectors_with_16bit_values_only"] = int(((low == 0).all(1).numpy() & ~same).sum())

# ---- token set from the published text
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(parent)
ds = load_dataset(dsname)["train"]
docs = []
for t in ds["text"]:
    ids = tok(t, add_special_tokens=False)["input_ids"][:BLOCK]
    docs.append(ids)
n = len(docs)
# a token gets a gradient if it sits at any position that feeds a later
# labelled position: every position except the last of the document
cnt_docs = np.zeros(V, np.int64)
cnt_occ = np.zeros(V, np.int64)
for ids in docs:
    a = np.asarray(ids[:-1])
    cnt_occ += np.bincount(a, minlength=V)[:V]
    cnt_docs += np.bincount(np.unique(a), minlength=V)[:V]
in_data = cnt_docs > 0
changed = ~same
tp = int((changed & in_data).sum())
fp = int((changed & ~in_data).sum())
fn = int((~changed & in_data).sum())
res["token_set"] = dict(docs=n, distinct_tokens_in_text=int(in_data.sum()), changed_and_in_text=tp,
                        changed_not_in_text=fp, in_text_not_changed=fn,
                        exact=bool(fp == 0 and fn == 0))
res["token_set"]["changed_not_in_text_ids"] = [int(i) for i in np.nonzero(changed & ~in_data)[0][:50]]
res["token_set"]["changed_not_in_text_strs"] = [tok.convert_ids_to_tokens(int(i)) for i in np.nonzero(changed & ~in_data)[0][:50]]
res["token_set"]["in_text_not_changed_strs"] = [tok.convert_ids_to_tokens(int(i)) for i in np.nonzero(~changed & in_data)[0][:50]]
print(json.dumps(res, indent=1), flush=True)

# ---- timing: tokens that appear in exactly one document (and few times)
steps_per_epoch = math.ceil(n / BS)
N = steps_per_epoch * EPOCHS
res["steps_assumed"] = dict(bs=BS, epochs=EPOCHS, steps_per_epoch=steps_per_epoch, total=N)
doc_of = {}
for di, ids in enumerate(docs):
    for t in set(ids[:-1]):
        if cnt_docs[t] == 1:
            doc_of[t] = di
rare = np.array(sorted(doc_of), dtype=np.int64)
rare = rare[changed[rare]]
rare_doc = np.array([doc_of[int(t)] for t in rare])
# observed size per token: high quantile of |change| (coordinates whose
# per-touch signs agree reach the top)
q = torch.quantile(absd[torch.from_numpy(rare)], torch.tensor([0.5, 0.99, 0.999]), dim=1).numpy()
obs_med, obs_hi = q[0], q[2]
res["rare_tokens"] = int(len(rare))


def lr_sched(peak, N, W):
    lr = np.zeros(N + 1)
    for s in range(1, N + 1):
        k = s - 1  # HF LambdaLR: factor evaluated at step index k for the k-th update
        if k < W:
            f = k / max(1, W)
        else:
            p = (k - W) / max(1, N - W)
            f = max(0.0, 0.5 * (1 + math.cos(math.pi * p)))
        lr[s] = peak * f
    return lr


def touches_for_seed(seed):
    """doc -> list of 1-based global steps it was used in."""
    st = [[] for _ in range(n)]
    for e in range(EPOCHS):
        g = torch.Generator()
        g.manual_seed(seed + e)
        perm = torch.randperm(n, generator=g).tolist()
        pad = (-n) % BS
        order = perm + perm[:pad]
        for pos, d in enumerate(order):
            st[d].append(e * steps_per_epoch + pos // BS + 1)
    return st


def adam_peak(touch_steps, lr, b1=0.9, b2=0.95):
    """move of a coordinate whose gradient is +1 at each touch, 0 elsewhere."""
    m = v = 0.0
    x = 0.0
    T = set(touch_steps)
    for t in range(1, len(lr)):
        gt = 1.0 if t in T else 0.0
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * gt * gt
        if v > 0:
            x += lr[t] * (m / (1 - b1 ** t)) / (math.sqrt(v / (1 - b2 ** t)) + 1e-30)
    return x


def score_seed(seed, lr, sub):
    st = touches_for_seed(seed)
    pred = np.array([adam_peak(st[rare_doc[i]], lr) for i in sub])
    o = obs_hi[sub]
    lr_ = np.log(o) - np.log(pred)
    scale = float(np.median(lr_))
    dev = np.abs(lr_ - scale)
    return dict(seed=seed, scale=math.exp(scale), med_abs_log_dev=float(np.median(dev)),
                within_5pct=float((dev < 0.05).mean()), corr=float(np.corrcoef(np.log(o), np.log(pred))[0, 1]))


lr_pub = lr_sched(1.0, N, math.ceil(0.05 * N))
rng = np.random.default_rng(0)
sub = rng.choice(len(rare), size=min(400, len(rare)), replace=False)
scores = [score_seed(s, lr_pub, sub) for s in range(0, 200)]
scores.sort(key=lambda r: r["med_abs_log_dev"])
res["seed_scan_top"] = scores[:5]
res["seed_scan_median_dev"] = float(np.median([r["med_abs_log_dev"] for r in scores]))
best = scores[0]["seed"]
full = score_seed(best, lr_pub, np.arange(len(rare)))
res["best_seed_all_rare"] = full
# per-document step agreement: docs with >=3 rare tokens, all consistent
st = touches_for_seed(best)
res["published_recipe_checked"] = dict(kind="cosine", warmup=math.ceil(0.05 * N), N=N, betas=[0.9, 0.95],
                                       peak_lr_estimated=full["scale"])
print(json.dumps({k: res[k] for k in ("seed_scan_top", "seed_scan_median_dev", "best_seed_all_rare",
                                        "published_recipe_checked")}, indent=1), flush=True)
json.dump(res, open(outp, "w"), indent=1)
