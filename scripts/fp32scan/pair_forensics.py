#!/usr/bin/env python3
"""Generic blind forensics of a public fp32 fine-tune against candidate parents.

  python3 scripts/fp32scan/pair_forensics.py OUT.json CHILD PARENT1 [PARENT2 ...]

For each candidate parent: how many token vectors are bit-identical (after
upcasting the parent to fp32), and whether the rest is a uniform shrink. For the
best parent: changed tokens, added vocabulary entries, special tokens, the share
of numbers changed per vector, single-touch ('flat') vectors and the per-token
change sizes (the input to a schedule fit).
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

out, child, *parents = sys.argv[1:]
EMB = "model.embed_tokens.weight"
res = dict(child=child, candidates={})
crepo, _, crev = child.partition("@")
W1 = remote.tensor(crepo, EMB, crev or None).float()
V1 = W1.shape[0]
best = None
for p in parents:
    try:
        prepo, _, prev = p.partition("@")
        W0 = remote.tensor(prepo, EMB, prev or None).float()
    except Exception as e:
        res["candidates"][p] = dict(error=str(e)[:120])
        continue
    V = min(V1, W0.shape[0])
    a, b = W0[:V], W1[:V]
    same = (a == b).all(1)
    nb = (a.abs() > 0).any(1)
    r = ((a.double() * b.double()).sum(1) / (a.double() ** 2).sum(1).clamp_min(1e-30))
    cos = torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=1)
    rec = dict(rows_parent=int(W0.shape[0]), compared=V, bit_identical=int(same.sum()),
               cos_median=float(cos.median()), ratio_median=float(r[nb].median()),
               rel_change_median=float(((b - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-30)).median()))
    res["candidates"][p] = rec
    print(p, rec, flush=True)
    if best is None or (rec["bit_identical"], rec["cos_median"]) > (res["candidates"][best]["bit_identical"], res["candidates"][best]["cos_median"]):
        best, Wb = p, W0
res["best_parent"] = best
V = min(V1, Wb.shape[0])
a, b = Wb[:V], W1[:V]
d = (b - a)
same = (d == 0).all(1).numpy()
fr = (d != 0).float().mean(1).numpy()
absd = d.abs()
med = absd.median(1).values
flat = (((absd - med[:, None]).abs() <= 0.01 * med[:, None]).float().mean(1)).numpy()
res["best"] = dict(bit_identical=int(same.sum()), changed=int((~same).sum()),
                   share_numbers_changed_hist=np.histogram(fr, bins=[0, 1e-9, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0001])[0].tolist(),
                   flat_vectors=int(((flat > 0.9) & ~same).sum()),
                   added_rows=int(V1 - Wb.shape[0]))
# vocabulary entries the child added
if V1 > Wb.shape[0]:
    ex = W1[Wb.shape[0]:]
    res["best"]["added_rows_norms"] = ex.norm(dim=1).tolist()
    res["best"]["typical_norm"] = float(a.norm(dim=1).median())
# special tokens
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(crepo, revision=crev or None)
    sp = sorted(set(tok.all_special_ids) | {i for i, t in tok.added_tokens_decoder.items()})
    sp = [i for i in sp if i < V1]
    res["special"] = [dict(id=i, tok=tok.convert_ids_to_tokens(i),
                           state=("added" if i >= V else ("bit-identical" if same[i] else "changed")),
                           share_changed=(None if i >= V else float(fr[i])),
                           norm_child=float(W1[i].norm()), norm_parent=(None if i >= V else float(a[i].norm())))
                      for i in sp]
except Exception as e:
    res["special_error"] = str(e)[:200]
# per-token change sizes of flat vectors (single-touch candidates)
fl = np.nonzero((flat > 0.9) & ~same)[0]
res["flat_moves"] = sorted(float(med[i]) for i in fl)[:20000]
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("flat_moves", "special")}, indent=1))
