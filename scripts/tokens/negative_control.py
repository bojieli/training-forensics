#!/usr/bin/env python3
"""Negative control: the token-record pipeline on pairs that are NOT base -> fine-tune.

A sound reading finds no bit-identical token vectors between unrelated models. Two
runs from the same random start may share vectors, but only for tokens neither
run trained; those must all be unreachable or still at starting size.
  - pythia-160m vs pythia-160m-seed1 / -seed2: same data and recipe, different seed
  - pythia-410m vs pythia-410m-deduped: different data
  - Qwen2-7B vs Qwen2.5-7B: same tokenizer and shape, separate pretraining runs
Writes results/tokens/negative_control.json.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import json
import math
import os
import pathlib
import sys

os.environ.setdefault("CKPTF_CACHE", str(paths.cache("trainforensics") / "emb"))
ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "roster", ROOT / "scripts" / "tokens"):
    sys.path.insert(0, str(p))
import numpy as np  # noqa: E402
import sweep  # noqa: E402
import mine  # noqa: E402
from trainforensics import remote  # noqa: E402

PAIRS = [("EleutherAI/pythia-160m", "EleutherAI/pythia-160m-seed1"),
         ("EleutherAI/pythia-160m", "EleutherAI/pythia-160m-seed2"),
         ("EleutherAI/pythia-410m", "EleutherAI/pythia-410m-deduped"),
         ("Qwen/Qwen2-7B", "Qwen/Qwen2.5-7B")]


def emb(repo):
    ix = remote.index(repo)
    k = [n for n in ix if n.endswith("embed_in.weight") or n.endswith("embed_tokens.weight")][0]
    return remote.open_file(repo, ix[k]).get(k)


def main():
    out = []
    for a, b in PAIRS:
        W0, W1 = emb(a), emb(b)
        V = min(W0.shape[0], W1.shape[0])
        W0, W1 = W0[:V], W1[:V]
        st, unmoved, near, moved, _ = sweep.analyse(W0, W1, W1.dtype, None)
        d = W0.shape[1]
        rec = dict(parent=a, child=b, rows=V, P=st["P"], P_exact_frac=st["P_exact_frac"],
                   unmoved=st["unmoved"], near=st["near"], moved=st["moved"])
        if st["unmoved"]:
            voc = mine.vocab(b, V)
            ids = np.nonzero(unmoved)[0]
            unreach = [i for i in ids if i in voc and not voc[i][3]]
            n0 = W0.double().norm(dim=1).numpy()
            bad = [i for i, v in voc.items() if v[4] and i < V]
            std = mine.card_std(a)
            if std is None and len(bad) >= 5:
                std = float(np.median(n0[bad])) / math.sqrt(d)
            init = np.abs(n0 / (std * math.sqrt(d)) - 1) < 0.03 if std else np.zeros(V, bool)
            at_init = [i for i in ids if init[i] and i not in set(unreach)]
            left = [i for i in ids if i not in set(unreach) and not init[i]]
            rec.update(unmoved_unreachable=len(unreach), unmoved_at_init=len(at_init), unmoved_left=len(left),
                       left_tokens=[voc[i][0] if i in voc else None for i in left[:50]])
            rec["note"] = ("same random start (shared values for tokens neither run trained)" if not left
                           else f"{len(left)} shared vectors are reachable and trained-size: inspect")
        else:
            rec["note"] = "no shared vectors: no token record produced"
        print(rec, flush=True)
        out.append(rec)
    json.dump(out, open(ROOT / "results" / "tokens" / "negative_control.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
