"""Earliest position of each token in the s1.1 texts: do the not-updated in-data tokens all
occur only beyond some position (a truncation length the published data doesn't show)?"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import json, sys, pathlib
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "published"
child, data, dsname = sys.argv[1], sys.argv[2], sys.argv[3]
r = json.load(open(OUT / f"{child.replace('/', '__')}__{data}.json"))
tok = AutoTokenizer.from_pretrained(r["base"])
ds = load_dataset(dsname, split="train")
earliest = {}
lens = []
for t in ds["text"]:
    ids = tok(t)["input_ids"]
    lens.append(len(ids))
    seen = set()
    for p, i in enumerate(ids):
        if i not in seen:
            seen.add(i)
            if p < earliest.get(i, 1 << 30):
                earliest[i] = p
# fraction changed per token: recompute from the comparison's lists is not stored; reload embeddings cheaply
sys.path.insert(0, str(ROOT))
import torch
from trainforensics import remote
remote.CACHE = paths.cache("trainforensics") / "emb"
W0 = remote.tensor(r["base"], "model.embed_tokens.weight").float()
remote.CACHE = paths.cache("tf_published")
W1 = remote.tensor(child, "model.embed_tokens.weight")
changed = (W0 != W1).any(1).numpy()
pres = np.array(sorted(earliest))
e = np.array([earliest[i] for i in pres])
ch = changed[pres]
lo = int(e[~ch].min()) if (~ch).any() else None
hi = int(e[ch].max())
print("docs", len(lens), "max len", max(lens), "docs longer than 20000:", sum(l > 20000 for l in lens),
      "longer than 32768:", sum(l > 32768 for l in lens))
print("in data, not updated:", int((~ch).sum()), "their earliest positions min", lo)
print("in data, updated: latest earliest position", hi)
res = dict(child=child, data=data, n_docs=len(lens), max_len=max(lens),
           docs_over={str(c): int(sum(l > c for l in lens)) for c in (8192, 16384, 20000, 24576, 32768)},
           not_updated=int((~ch).sum()), not_updated_min_earliest=lo, updated_max_earliest=hi,
           not_updated_tokens=[(int(i), tok.convert_ids_to_tokens(int(i)), int(earliest[i])) for i in pres[~ch]])
json.dump(res, open(OUT / f"cutoff_{child.replace('/', '__')}.json", "w"), indent=1)
