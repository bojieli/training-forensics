#!/usr/bin/env python3
"""Re-label deepseek.json with the STYLE.md threshold: 'barely changed' = under 1% of a
vector's numbers differ (the sweep's 'near'). Raw fractions are kept; only labels/counts change."""
import json, pathlib
R = pathlib.Path(__file__).resolve().parents[2] / "results" / "frontier_ds"
T = 0.01
d = json.load(open(R / "deepseek.json"))
d["barely_threshold"] = T


def lab(f):
    return "bit-identical" if f == 0 else ("barely changed" if f < T else "updated")


for c in d["comparisons"]:
    if "error" in c:
        continue
    for t in c["named_detail"] + c["placeholder_changed"] + c["unused_changed"]:
        t["status"] = lab(t["frac_changed"])
    for grp, det in (("named", c["named_detail"]), ("placeholder", c["placeholder_changed"]), ("unused", c["unused_changed"])):
        n = c[grp]["n"]
        st = [t["status"] for t in det]
        if grp == "named":
            c[grp]["counts"] = {k: st.count(k) for k in ("bit-identical", "barely changed", "updated")}
        else:
            ch = {k: st.count(k) for k in ("barely changed", "updated")}
            c[grp]["counts"] = {"bit-identical": n - len(det), **ch}
json.dump(d, open(R / "deepseek.json", "w"), indent=1)
print({(c["child"].split("/")[1]): c["placeholder"]["counts"] for c in d["comparisons"] if "error" not in c})
