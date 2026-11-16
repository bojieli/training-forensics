#!/usr/bin/env python3
"""Print notable findings from the merged roster sweep (weight decay, single-step tokens, tied matrices, output layers, Qwen special tokens)."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
recs = {}
for f in ["sweep.jsonl", "sweep_cml3.jsonl"]:
    p = ROOT / "results" / "roster" / f
    if not p.exists():
        continue
    for l in open(p):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        o = recs.get(r["key"])
        if o is None or not (o["status"] == "ok" and r["status"] != "ok"):
            recs[r["key"]] = r
R = [r for r in recs.values() if r["status"] == "ok"]


def E(r, k):
    return r["embed"][k]


print("== weight decay visible (>=10 untouched token vectors reproduced by one shrink factor P != 1)")
for r in sorted(R, key=lambda r: r["embed"]["P"]):
    e = r["embed"]
    if not r["tied"] and e["unmoved"] >= 10 and (e["P_exact_frac"] or 0) >= 0.5 and abs(e["P"] - 1) > 1e-6:
        print(f'{r["child"][:48]:48s} <- {r["parent"][:40]:40s} P={e["P"]:.6f} sum_lr_wd={e["lr_wd_integral"] if e["lr_wd_integral"] is None else round(e["lr_wd_integral"], 6)} '
              f'unmoved={e["unmoved"]} coordfrac={e["P_exact_frac"]:.3f}')
print("== tokens with the single-step signature (>=20, change above one 16-bit step)")
for r in R:
    e = r["embed"]
    D = e["flat_D_q"][2]
    if not r["tied"] and e["flat"] >= 20 and D and (r["dtype_child"] == "F32" or D >= 2e-5):
        print(r["child"], r["dtype_child"], e["flat"], "D med %.3g" % D, "lr med %.3g" % e["flat_lr_q"][1])
tied = [r for r in R if r["tied"]]
print("== tied input/output matrix:", len(tied), "pairs; with >=10 untouched token vectors:", sum(1 for r in tied if E(r, "unmoved") >= 10),
      "; token vectors unchanged:", sum(1 for r in tied if E(r, "moved") == 0))
H = [r for r in R if r.get("head") and "moved" in r["head"]]
print("== separate output layers measured:", len(H), "; >=99% of output vectors updated:",
      sum(1 for r in H if r["head"]["moved"] + r["head"]["near"] >= 0.99 * r["head"]["rows"] and r["head"]["moved"] > 0),
      "; output layer unchanged:", sum(1 for r in H if r["head"]["moved"] == 0))
print("== fraction of vocabulary never updated, among untied pairs with a token record")
U = [r for r in R if not r["tied"] and E(r, "unmoved") >= 10 and E(r, "moved") > 0]
fr = sorted(E(r, "unmoved") / E(r, "rows") for r in U)
if fr:
    print("n", len(fr), "median %.3f" % fr[len(fr) // 2], "min %.4f" % fr[0], "max %.3f" % fr[-1])
print("== special tokens updated, Qwen-parent fine-tunes (untied, >=10 untouched token vectors)")
for r in sorted(R, key=lambda r: r["child"]):
    if "qwen" in r["parent"].lower() and not r["tied"] and E(r, "unmoved") >= 10:
        sp = [t for t in r["special"] if t["in_parent"]]
        mv = [t["content"] for t in sp if t["state"] == "moved"]
        print(f'{r["parent"][:34]:34s} -> {r["child"][:42]:42s} {len(mv)}/{len(sp)} {mv[:9]}')
