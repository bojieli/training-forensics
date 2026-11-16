#!/usr/bin/env python3
"""Adversarial follow-up to scan.py: middle and tail of every tensor of the current revision (scan.py read only the head)."""
import json, sys, collections, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import st
N = 1 << 18
R = "mattshumer/Reflection-Llama-3.1-70B"
REF = {"L3-I": "NousResearch/Meta-Llama-3-70B-Instruct", "L3.1-I": "unsloth/Meta-Llama-3.1-70B-Instruct"}
ix = {k: st.index(r) for k, r in REF.items()}; cx = st.index(R)
def one(name):
    e = cx[name]; tot = int(np.prod(e[2])); rtot = int(np.prod(ix["L3-I"][name][2])); row = dict(name=name, shard=e[0])
    if tot <= (1 << 20): return None                       # fully covered by scan.py
    for tag, off in (("mid", (rtot // 2) // 8192 * 8192), ("tail", rtot - N)):
        Wb = st.read(R, None, e, off, N, workers=1); Wf = Wb.view(np.float32).astype(np.float64); o = {}
        for k in REF:
            Cb = st.bits32(st.read(REF[k], None, ix[k][name], off, N, workers=1), ix[k][name][1]); Cf = Cb.view(np.float32).astype(np.float64)
            o[k] = dict(eq=float((Wb == Cb).mean()), rel=float(np.linalg.norm(Wf - Cf) / np.linalg.norm(Cf)))
        row[tag] = o
    return row
with ThreadPoolExecutor(12) as ex: rows = [r for r in ex.map(one, list(cx)) if r]
bad = [r for r in rows for t in ("mid", "tail") if r[t]["L3-I"]["rel"] >= r[t]["L3.1-I"]["rel"]]
ratio = [r[t]["L3.1-I"]["rel"] / max(r[t]["L3-I"]["rel"], 1e-12) for r in rows for t in ("mid", "tail")]
s = dict(tensors=len(rows), shards=len(set(r["shard"] for r in rows)), samples=len(ratio), samples_not_nearest_L3I=[b["name"] for b in bad],
         min_ratio_relL31I_over_relL3I=float(min(ratio)), median_ratio=float(np.median(ratio)),
         rel_L3I_range=[min(r[t]["L3-I"]["rel"] for r in rows for t in ("mid", "tail")), max(r[t]["L3-I"]["rel"] for r in rows for t in ("mid", "tail"))])
print(json.dumps(s, indent=1)); json.dump(dict(summary=s, rows=rows), open(sys.argv[1], "w"))
