#!/usr/bin/env python3
"""Step 2/3: EVERY tensor of EVERY shard of the current revision (and of the first upload): the first N elements,
bitwise, vs Llama-3 / 3.1 base and instruct. Nothing is written to disk but the result table."""
import json, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import st
N = 1 << 20
R = "mattshumer/Reflection-Llama-3.1-70B"
REF = {"L3-I": "NousResearch/Meta-Llama-3-70B-Instruct", "L3": "NousResearch/Meta-Llama-3-70B",
       "L3.1-I": "unsloth/Meta-Llama-3.1-70B-Instruct", "L3.1": "unsloth/Meta-Llama-3.1-70B"}
CH = {"current": None, "first": "e7bfef9796"}
ix = {k: st.index(r) for k, r in REF.items()}; cx = {c: st.index(R, rev) for c, rev in CH.items()}
names = list(cx["current"]); assert set(names) == set(cx["first"]) == set(ix["L3-I"]) == set(ix["L3.1-I"]), "tensor name sets differ"
def one(name):
    Rb = {k: st.bits32(st.read(REF[k], None, ix[k][name], 0, N, workers=1), ix[k][name][1]) for k in REF}
    Rf = {k: v.view(np.float32).astype(np.float64) for k, v in Rb.items()}
    row = dict(name=name, n=int(len(Rb["L3-I"])), shape=cx["current"][name][2],
               ref_rel_L3I_vs_L31I=float(np.linalg.norm(Rf["L3-I"] - Rf["L3.1-I"]) / np.linalg.norm(Rf["L3-I"])),
               ref_eq_L3I_vs_L31I=float((Rb["L3-I"] == Rb["L3.1-I"]).mean()))
    for c, rev in CH.items():
        e = cx[c][name]; Wb = st.bits32(st.read(R, rev, e, 0, N, workers=1), e[1])[:row["n"]]; Wf = Wb.view(np.float32).astype(np.float64)
        o = dict(shard=e[0], grid=float(((Wb & 0xFFFF) == 0).mean()))
        for k in REF:
            o[k] = dict(eq=float((Wb == Rb[k]).mean()), rel=float(np.linalg.norm(Wf - Rf[k]) / np.linalg.norm(Rf[k])))
        d = Rf["L3.1-I"] - Rf["L3-I"]; o["alpha"] = float(((Wf - Rf["L3-I"]) * d).sum() / max((d * d).sum(), 1e-300))
        o["nearest"] = min(REF, key=lambda k: o[k]["rel"])
        row[c] = o
    return row
rows = []
with ThreadPoolExecutor(12) as ex:
    for i, r in enumerate(ex.map(one, names)):
        rows.append(r)
        if i % 40 == 0: print(i, r["name"], r["current"]["shard"], r["current"]["nearest"], r["current"]["L3-I"], r["current"]["L3.1-I"], "| first:", r["first"]["nearest"], r["first"]["L3.1-I"], flush=True)
json.dump(dict(sample_elements=N, refs=REF, rows=rows), open(sys.argv[1], "w"))
