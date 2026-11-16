#!/usr/bin/env python3
"""Aggregate all_tensors.json per shard and per tensor type."""
import json, sys, re, collections, numpy as np
d = json.load(open(sys.argv[1])); rows = d["rows"]
out = {}
for c in ("current", "first"):
    sh = collections.defaultdict(list); ty = collections.defaultdict(list)
    for r in rows:
        sh[r[c]["shard"]].append(r); t = re.sub(r"model\.layers\.\d+\.", "", r["name"]); ty[t].append(r)
    near = collections.Counter(r[c]["nearest"] for r in rows)
    shard_near = {s: sorted(set(r[c]["nearest"] for r in v)) for s, v in sh.items()}
    o = dict(tensors=len(rows), shards_covered=len(sh), nearest_counts=dict(near),
             shards_all_tensors_nearest=dict(collections.Counter("+".join(v) for v in shard_near.values())),
             shards_not_pure=[s for s, v in shard_near.items() if len(v) > 1], by_type={})
    for t, v in sorted(ty.items()):
        g = lambda f: [float(np.min([f(r) for r in v])), float(np.median([f(r) for r in v])), float(np.max([f(r) for r in v]))]
        o["by_type"][t] = dict(n=len(v), grid=g(lambda r: r[c]["grid"]), eq_L3I=g(lambda r: r[c]["L3-I"]["eq"]), eq_L31I=g(lambda r: r[c]["L3.1-I"]["eq"]),
            rel_L3I=g(lambda r: r[c]["L3-I"]["rel"]), rel_L31I=g(lambda r: r[c]["L3.1-I"]["rel"]), rel_L3=g(lambda r: r[c]["L3"]["rel"]),
            alpha=g(lambda r: r[c]["alpha"]), ref_rel_L3I_vs_L31I=g(lambda r: r["ref_rel_L3I_vs_L31I"]), ref_eq=g(lambda r: r["ref_eq_L3I_vs_L31I"]),
            min_ratio_relL31I_over_relL3I=float(np.min([r[c]["L3.1-I"]["rel"] / max(r[c]["L3-I"]["rel"], 1e-12) for r in v])))
    out[c] = o
json.dump(out, open(sys.argv[2], "w"), indent=1)
for c in out:
    o = out[c]; print("==", c, {k: o[k] for k in ("tensors", "shards_covered", "nearest_counts", "shards_all_tensors_nearest", "shards_not_pure")})
    print("type n | grid med | eq_L3I min/med/max | eq_L31I med | rel_L3I min/med/max | rel_L31I min/med/max | alpha min/med/max | ref L3I-L31I rel min/med | min ratio")
    for t, b in o["by_type"].items():
        f = lambda x: "/".join("%.3g" % y for y in x)
        print(t, b["n"], "|", "%.3g" % b["grid"][1], "|", f(b["eq_L3I"]), "|", "%.3g" % b["eq_L31I"][1], "|", f(b["rel_L3I"]), "|", f(b["rel_L31I"]), "|", f(b["alpha"]), "|", f(b["ref_rel_L3I_vs_L31I"][:2]), "|", "%.3g" % b["min_ratio_relL31I_over_relL3I"])
