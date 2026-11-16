#!/usr/bin/env python3
"""Calibrate the comparability classes on comparisons that are NOT parent -> child (review B1, M4).

null      same tokenizer, separately trained (Pythia seeds / deduped, Qwen2 vs Qwen2.5): any
          "token record" or "changed throughout" here is a false positive of the reader.
          T_LEFT = above the largest cleaned count (`left`) any null pair shows;
          T_FIT  = above the largest shrink-fit score any null pair shows.
sibling   two releases trained from the same parent: they share the vectors neither training
          touched, so a "token record" between them is expected, and is the false-positive rate
          of reading a class as "direct parent".
Inputs: results/frontier/negctl_pairs.json, results/frontier/recount/{NULL,SIB}*.json.
Writes results/frontier/calibration.json and negative_control_expanded.json.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "frontier"))
OUT = ROOT / "results" / "frontier"


def main():
    pairs = json.load(open(OUT / "negctl_pairs.json"))
    import comparability as C
    rows = []
    for p in pairs:
        f = OUT / "recount" / (C.safe_key(p["key"]) + ".json")
        if not f.exists():
            continue
        rc = json.loads(f.read_text())
        rows.append(dict(key=p["key"], kind=p["kind"], a=p["parent"], b=p["child"], common_parent=p.get("common_parent"),
                         P=rc["P"], fit=rc["P_exact_frac"], strict=rc["strict"], band=rc["band"], left=rc["left"],
                         removed=rc["removed"]))
    null = [r for r in rows if r["kind"] == "null"]
    sib = [r for r in rows if r["kind"] == "sibling"]
    max_left = max([r["left"] for r in null] or [0])
    # the fit threshold separates "no shared vectors but still the parent plus broad training" from
    # unrelated models, so it is calibrated on the null pairs with no vector consistent with a shrink
    max_fit = max([r["fit"] or 0 for r in null if r["band"] == 0] or [0])
    T_LEFT = max(50, 2 * max_left + 1)
    T_FIT = min(0.99, round(max_fit + 0.05, 3)) if null else 0.5
    cal = dict(T_LEFT=T_LEFT, T_FIT=T_FIT, null_pairs=len(null), sibling_pairs=len(sib),
               null_max_left=max_left, null_max_fit_without_shared_vectors=max_fit,
               source="scripts/frontier/calibrate.py on results/frontier/negctl_pairs.json")
    for r in rows:
        rec = dict(tied=False, embed=dict(rows=1, identical=0, P=r["P"], P_exact_frac=r["fit"], unmoved=r["strict"]))
        r["class"] = C.classify(rec, r, T_LEFT=T_LEFT, T_FIT=T_FIT)
        r["class_before"] = C.classify_old(rec)
    def rate(rs, cls):
        return (sum(r["class"] == cls for r in rs), len(rs))
    cal["null_token_record"] = rate(null, "token record")
    cal["null_changed_throughout"] = rate(null, "changed throughout")
    cal["null_token_record_before"] = (sum(r["class_before"] == "token record" for r in null), len(null))
    cal["sibling_token_record"] = rate(sib, "token record")
    cal["sibling_token_record_before"] = (sum(r["class_before"] == "token record" for r in sib), len(sib))
    json.dump(cal, open(OUT / "calibration.json", "w"), indent=1)
    json.dump(rows, open(OUT / "negative_control_expanded.json", "w"), indent=1)
    print(json.dumps(cal, indent=1))
    for r in sorted(rows, key=lambda r: (r["kind"], -r["left"])):
        print(f"{r['kind']:7s} {r['class']:18s} left={r['left']:7d} band={r['band']:7d} fit={r['fit'] or 0:.3f} {r['a']} ~ {r['b']}")


if __name__ == "__main__":
    main()
