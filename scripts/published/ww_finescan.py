#!/usr/bin/env python3
"""Fine scan of the truncation length C for the WebWorld models (a token is in the trained text
if its earliest position in any sample is < C, and it is not only a final token).
Writes results/published/webworld/finescan.json."""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ww_compare import OUT, PAIRS, emb, frac_changed  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

d = np.load(OUT / "data_webworld.npz")
chat, minpos, last = d["chat"], d["minpos"], d["last"]
tok = AutoTokenizer.from_pretrained("Qwen/WebWorld-8B")
V = len(chat)
reg = np.ones(V, bool)
for t, i in tok.get_added_vocab().items():
    if i < V:
        reg[i] = False
out = {}
for parent, child in PAIRS:
    W0, W1 = emb(parent), emb(child)
    n = min(W0.shape[0], W1.shape[0], V)
    fc = frac_changed(W0[:n], W1[:n])
    ch, bit = fc > 0, fc == 0
    r = reg[:n]
    notlast = (chat[:n] - last[:n]) > 0
    rows = []
    for C in list(range(16000, 30001, 250)):
        pr = (chat[:n] > 0) & (minpos[:n] < C) & notlast & r
        rows.append(dict(cutoff=C, present_bit_identical=int((pr & bit).sum()), changed_absent=int((~pr & r & ch).sum())))
    best = min(rows, key=lambda x: x["present_bit_identical"] + x["changed_absent"])
    # tokens that are in the data but whose every occurrence is beyond position C, by state
    out[child] = dict(scan=rows, best=best)
    print(child, best, flush=True)
json.dump(out, open(OUT / "finescan.json", "w"), indent=1)
