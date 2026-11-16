#!/usr/bin/env python3
"""Resolve candidate frontier comparisons into sweep-ready pair records (results/frontier_x/pairs_x.json)."""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "roster")); sys.path.insert(0, str(ROOT / "scripts" / "frontier_x")); sys.path.insert(0, str(ROOT))
import build_pairs as bp  # noqa
from candidates import all_pairs  # noqa

def existing_keys():
    ks = set()
    for f in (ROOT / "results" / "roster").glob("sweep*.jsonl"):
        if f.name == "sweep_frontier_x.jsonl":
            continue
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("status") == "ok":
                ks.add((r["parent"], r["child"]))
    return ks

def main():
    have = existing_keys()
    out, skipped = [], []
    sides = {}
    def S(r):
        if r not in sides:
            sides[r] = bp.side(r)
        return sides[r]
    for par, ch, kind in all_pairs():
        if (par, ch) in have:
            skipped.append(dict(parent=par, child=ch, why="already in roster sweep")); continue
        a, b = S(par), S(ch)
        bad = [x["repo"] + ": " + ("missing" if not x.get("exists") else "gated" if x.get("gated") else "no embedding tensor found" if not x.get("embed") else "")
               for x in (a, b) if not x.get("exists") or x.get("gated") or not x.get("embed")]
        if bad:
            skipped.append(dict(parent=par, child=ch, why="; ".join(bad))); continue
        cmp_ = bp.compare(a, b)
        ce = cmp_.get("embed") or {}
        if not ce.get("compatible"):
            skipped.append(dict(parent=par, child=ch, why="embedding not comparable: " + str(ce.get("why")), shapes=[a["embed"].get("shape"), b["embed"].get("shape")])); continue
        out.append(dict(key=f"{par}@ -> {ch}", parent=a, child=b, src=["frontier_x:" + kind], compat=cmp_,
                        tied=bool(a.get("tied") or b.get("tied")), relation=kind, rank=0,
                        bytes=a["embed"]["bytes"] + b["embed"]["bytes"]))
        print("ok ", par, "->", ch, a["embed"]["shape"], a["embed"]["dtype"], b["embed"]["dtype"], "tied" if out[-1]["tied"] else "", flush=True)
    for s in skipped:
        print("skip", s["parent"], "->", s["child"], ":", s["why"], flush=True)
    od = ROOT / "results" / "frontier_x"; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "pairs_x.json", "w"), indent=1, default=str)
    json.dump(skipped, open(od / "skipped_x.json", "w"), indent=1, default=str)
    print(len(out), "pairs;", len(skipped), "skipped; GB", round(sum(p["bytes"] for p in out) / 1e9, 1))

if __name__ == "__main__":
    main()
