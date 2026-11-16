#!/usr/bin/env python3
"""Step 2/3/4: embedding rows, bitwise, both uploads vs every reference; reference-pair separability; extra rows.
Rows are fetched in blocks with ranged reads and never written to disk."""
import json, sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import st
from pins import REFL_MAIN
E = sys.argv[2] if len(sys.argv) > 2 else "model.embed_tokens.weight"; V = 128256; D = 8192; B = 4096
REF = {"L3-I": "NousResearch/Meta-Llama-3-70B-Instruct", "L3": "NousResearch/Meta-Llama-3-70B",
       "L3.1-I": "unsloth/Meta-Llama-3.1-70B-Instruct", "L3.1": "unsloth/Meta-Llama-3.1-70B", "L3.3-I": "unsloth/Llama-3.3-70B-Instruct"}
CH = {"current": ("mattshumer/Reflection-Llama-3.1-70B", REFL_MAIN), "first": ("mattshumer/Reflection-Llama-3.1-70B", "e7bfef9796")}
ix = {k: st.index(r)[E] for k, r in REF.items()}
cx = {k: st.index(*rv)[E] for k, rv in CH.items()}
print({k: v[:3] for k, v in {**ix, **cx}.items()}, flush=True)
keys = list(REF)
ident = {c: {k: np.zeros(V, bool) for k in keys} for c in CH}
valeq = {c: {k: 0 for k in keys} for c in CH}
rel = {c: {k: np.zeros(V, np.float32) for k in keys} for c in CH}
grid = {c: 0 for c in CH}
alpha = {c: np.zeros(V, np.float32) for c in CH}      # projection of (child-L3I) on (L3.1I-L3I): 0 = L3-I, 1 = L3.1-I, 0.5 = average
PAIRS = [("L3-I", "L3.1-I"), ("L3-I", "L3"), ("L3.1-I", "L3.1"), ("L3-I", "L3.3-I"), ("L3.1-I", "L3.3-I"), ("L3", "L3.1")]
pid = {p: np.zeros(V, bool) for p in PAIRS}; prel = {p: np.zeros(V, np.float32) for p in PAIRS}; pval = {p: 0 for p in PAIRS}
nrm = {k: np.zeros(V, np.float32) for k in keys}
for a in range(0, V, B):
    n = min(B, V - a)
    Rb = {k: st.bits32(st.read(REF[k], None, ix[k], a * D, n * D), ix[k][1]).reshape(n, D) for k in keys}
    Rf = {k: v.view(np.float32) for k, v in Rb.items()}
    for k in keys: nrm[k][a:a + n] = np.linalg.norm(Rf[k], axis=1)
    for p in PAIRS:
        eq = Rb[p[0]] == Rb[p[1]]; pid[p][a:a + n] = eq.all(1); pval[p] += int(eq.sum())
        prel[p][a:a + n] = np.linalg.norm(Rf[p[0]] - Rf[p[1]], axis=1) / np.maximum(nrm[p[0]][a:a + n], 1e-30)
    for c, (repo, rev) in CH.items():
        Wb = st.bits32(st.read(repo, rev, cx[c], a * D, n * D), cx[c][1]).reshape(n, D); Wf = Wb.view(np.float32)
        grid[c] += int(((Wb & 0xFFFF) == 0).sum())
        for k in keys:
            eq = Wb == Rb[k]; ident[c][k][a:a + n] = eq.all(1); valeq[c][k] += int(eq.sum())
            rel[c][k][a:a + n] = np.linalg.norm(Wf - Rf[k], axis=1) / np.maximum(nrm[k][a:a + n], 1e-30)
        d = (Rf["L3.1-I"] - Rf["L3-I"]).astype(np.float64); x = (Wf - Rf["L3-I"]).astype(np.float64)
        alpha[c][a:a + n] = (x * d).sum(1) / np.maximum((d * d).sum(1), 1e-30)
    print(a, {c: {k: int(ident[c][k].sum()) for k in keys} for c in CH}, flush=True)
live = nrm["L3-I"] > 1e-3
q = lambda x: [float(v) for v in np.quantile(x, [0, .01, .05, .5, .95, .99, 1])]
out = dict(refs=REF, rows=V, live_rows=int(live.sum()), quantiles="min,p1,p5,p50,p95,p99,max", children={}, ref_pairs={})
for p in PAIRS:
    out["ref_pairs"][" vs ".join(p)] = dict(rows_bit_identical=int(pid[p].sum()), rows_bit_identical_live=int((pid[p] & live).sum()),
        values_identical_share=pval[p] / (V * D), rel_dist_live=q(prel[p][live]))
for c in CH:
    o = dict(layout=cx[c][:3], values_on_bf16_grid=grid[c] / (V * D), per_ref={})
    Dm = np.stack([rel[c][k] for k in keys], 1)[live]; near = Dm.argmin(1)
    srt = np.sort(Dm, 1); o["nearest_margin_ratio_2nd_over_1st"] = q(srt[:, 1] / np.maximum(srt[:, 0], 1e-12))
    for i, k in enumerate(keys):
        o["per_ref"][k] = dict(rows_bit_identical=int(ident[c][k].sum()), rows_bit_identical_live=int((ident[c][k] & live).sum()),
            values_identical_share=valeq[c][k] / (V * D), rel_dist_live=q(rel[c][k][live]), nearest_share=float((near == i).mean()))
    idL3 = ident[c]["L3-I"]
    o["rows_identical_to_L3I_and_also_to_L3.1I"] = int((idL3 & ident[c]["L3.1-I"]).sum())
    o["rows_identical_to_L3I_where_L3I==L3.1I"] = int((idL3 & pid[("L3-I", "L3.1-I")]).sum())
    o["rows_identical_to_L3I_where_L3I==L3"] = int((idL3 & pid[("L3-I", "L3")]).sum())
    o["merge_alpha_live_all"] = q(alpha[c][live]); 
    if (~idL3 & live).any(): o["merge_alpha_live_changed_rows"] = q(alpha[c][~idL3 & live])
    o["identical_rows_id_range"] = dict(reserved_ge_128000=int(idL3[128000:].sum()), lt_128000=int(idL3[:128000].sum()))
    out["children"][c] = o
# extra rows
for c, (repo, rev) in CH.items():
    rows = cx[c][2][0]
    if rows > V:
        X = st.f32(st.read(repo, rev, cx[c], V * D, (rows - V) * D), cx[c][1]).reshape(-1, D)
        mean_cur = None
        out["children"][c]["extra_rows"] = dict(count=rows - V, norms=[float(v) for v in np.linalg.norm(X, axis=1)],
            on_bf16_grid=float(((X.view(np.uint32) & 0xFFFF) == 0).mean()), std=[float(v) for v in X.std(1)], mean=[float(v) for v in X.mean(1)],
            pairwise_cos=np.round((X / np.linalg.norm(X, axis=1, keepdims=True)) @ (X / np.linalg.norm(X, axis=1, keepdims=True)).T, 3).tolist(),
            distinct_values_row0=int(len(np.unique(X[0]))))
        np.save(sys.argv[1].replace(".json", "_extra_rows.npy"), X)
out["norm_live_median"] = {k: float(np.median(nrm[k][live])) for k in keys}
np.savez_compressed(sys.argv[1].replace(".json", "_rows.npz"), ident_current_L3I=ident["current"]["L3-I"], rel_current_L3I=rel["current"]["L3-I"], rel_current_L31I=rel["current"]["L3.1-I"], alpha_current=alpha["current"])
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
