#!/usr/bin/env python3
"""Updated token vectors of a published fine-tune vs the tokens in its published data.

For each (base, fine-tune, dataset):
  - base input-embedding (16-bit) and fine-tune input-embedding (fp32 / bf16 / fp16), read
    with ranged requests; the base is cast to the fine-tune's format (exact for fp32);
  - per token vector: fraction of its numbers that differ from round(P * base), where P is
    the uniform shrink (weight decay) that reproduces the most never-updated vectors
    (P = 1 checked first; fitted P otherwise);
  - states: bit-identical (0 numbers differ), barely changed (< 10%), updated (>= 10%);
  - compared with the token occurrence counts of the dataset (tokens_in_data.py).
Writes results/published/<child>.json.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402

OUT = ROOT / "results" / "published"
EMB = paths.cache("trainforensics") / "emb"
PUB = paths.cache("tf_published")


def get(repo, cache):
    remote.CACHE = cache
    return remote.tensor(repo, "model.embed_tokens.weight")


def frac_changed(W0, W1, P, chunk=8192):
    V = W0.shape[0]
    out = np.zeros(V)
    for a in range(0, V, chunk):
        pred = (W0[a:a + chunk].double() * P).to(torch.float32).to(W1.dtype)
        out[a:a + chunk] = (pred != W1[a:a + chunk]).double().mean(1).numpy()
    return out


def fit_P(W0, W1, frac1):
    """Shrink factor from vectors that are almost unchanged (median ratio)."""
    cand = np.nonzero(frac1 < 1.0)[0]
    if len(cand) < 50:
        # every vector changed somewhere: use vectors with the smallest relative change
        rel = ((W1.double() - W0.double()).norm(dim=1) / W0.double().norm(dim=1).clamp_min(1e-30)).numpy()
        cand = np.argsort(rel)[:2000]
    a = W0[cand].double().flatten()
    b = W1[cand].double().flatten()
    m = a.abs() > 1e-3
    P0 = float(torch.median(b[m] / a[m]))
    # refine: the P that reproduces the most sampled vectors exactly (round(P * base) == release)
    g = torch.Generator().manual_seed(0)
    samp = torch.randperm(W0.shape[0], generator=g)[:3000]
    A, B = W0[samp].double(), W1[samp]
    best, bestP = -1, P0
    for span, k in ((2e-3, 401), (1e-5, 201)):
        for P in np.linspace(bestP * (1 - span), bestP * (1 + span), k):
            hit = int(((A * P).to(torch.float32).to(B.dtype) == B).all(1).sum())
            if hit > best:
                best, bestP = hit, float(P)
    return bestP


def main():
    base, child, data = sys.argv[1], sys.argv[2], sys.argv[3]
    W0 = get(base, EMB)
    W1 = get(child, PUB)
    n = min(W0.shape[0], W1.shape[0])
    W0, W1 = W0[:n], W1[:n]
    W0c = W0  # cast to the release format happens after the shrink, in frac_changed
    f1 = frac_changed(W0c, W1, 1.0)
    P = 1.0
    best = f1
    if (f1 == 0).sum() < 0.001 * n:
        Pf = fit_P(W0c, W1, f1)
        fP = frac_changed(W0c, W1, Pf)
        if (fP == 0).sum() > (f1 == 0).sum():
            P, best = Pf, fP
    frac = best
    d = json.load(open(OUT / f"data_{data}.json"))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    res = dict(base=base, child=child, data=data, dtype=str(W1.dtype), vocab_rows=n, P=P,
               bit_identical=int((frac == 0).sum()), barely=int(((frac > 0) & (frac < 0.1)).sum()),
               updated=int((frac >= 0.1).sum()), variants={})
    # tokens the tokenizer defines (rows beyond len(tok) are unused padding slots)
    real = np.zeros(n, bool)
    real[:min(n, len(tok))] = True
    for v, counts in d["counts"].items():
        cnt = np.zeros(n, np.int64)
        for k, c in counts.items():
            if int(k) < n:
                cnt[int(k)] = c
        present = cnt > 0
        changed = frac > 0
        a_and = int((present & changed).sum())
        p_not = np.nonzero(present & ~changed)[0]
        c_not = np.nonzero(changed & ~present)[0]
        rec = dict(present=int(present.sum()), changed=int(changed.sum()), present_and_changed=a_and,
                   present_not_changed=len(p_not), changed_not_present=len(c_not),
                   changed_not_present_real=int(real[c_not].sum()),
                   present_not_changed_examples=[(int(i), tok.convert_ids_to_tokens(int(i)), int(cnt[i]))
                                                 for i in p_not[:40]],
                   present_not_changed_occ_hist={str(k): int(((cnt[p_not] == k)).sum()) for k in (1, 2, 3, 4, 5)},
                   present_not_changed_occ_max=int(cnt[p_not].max()) if len(p_not) else 0,
                   changed_not_present_examples=[(int(i), tok.convert_ids_to_tokens(int(i)) if i < len(tok) else None,
                                                  round(float(frac[i]), 4)) for i in c_not[:60]],
                   changed_not_present_frac_hist=np.histogram(frac[c_not], bins=[0, 1e-9, 0.01, 0.1, 0.5, 1.0001])[0].tolist()
                   if len(c_not) else [],
                   # among present tokens, how the state depends on how often the token occurs
                   by_occurrence={})
        for lo, hi in ((1, 1), (2, 2), (3, 5), (6, 20), (21, 100), (101, 10 ** 12)):
            sel = present & (cnt >= lo) & (cnt <= hi)
            if sel.sum():
                rec["by_occurrence"][f"{lo}-{hi}"] = dict(n=int(sel.sum()), bit_identical=int((frac[sel] == 0).sum()),
                                                          barely=int(((frac[sel] > 0) & (frac[sel] < 0.1)).sum()))
        res["variants"][v] = rec
    # special tokens
    sp = {}
    for t in tok.all_special_tokens + list(getattr(tok, "added_tokens_encoder", {}).keys()):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i < n:
            sp[t] = dict(id=int(i), frac=round(float(frac[i]), 4),
                         in_data=int(d["counts"]["main"].get(str(i), 0)))
    res["special"] = sp
    name = child.replace("/", "__")
    json.dump(res, open(OUT / f"{name}__{data}.json", "w"), indent=1)
    m = res["variants"]["main"]
    print(child, res["dtype"], "P", P, "bit-identical", res["bit_identical"], "barely", res["barely"],
          "| data present", m["present"], "changed", m["changed"], "present&changed", m["present_and_changed"],
          "present-not-changed", m["present_not_changed"], "changed-not-present", m["changed_not_present"])


if __name__ == "__main__":
    main()
