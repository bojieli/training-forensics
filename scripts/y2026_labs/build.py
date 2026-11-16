#!/usr/bin/env python3
"""Resolve 2026 candidates into sweep-ready pair records (results/y2026_labs/pairs.json)."""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "roster")); sys.path.insert(0, str(ROOT / "scripts" / "y2026_labs"))
import build_pairs as bp  # noqa
from candidates import all_pairs, CARD  # noqa
OD = ROOT / "results" / "y2026_labs"

def covered():
    ks = set()
    for f in (ROOT / "results" / "roster").glob("sweep*.jsonl"):
        if f.name == "sweep_2026_labs.jsonl":
            continue
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("status") == "ok": ks.add((r["parent"], r["child"]))
    for f in [ROOT / "results" / "frontier_x" / "pairs_x.json"]:
        if f.exists():
            for p in json.load(open(f)): ks.add((p["parent"]["repo"], p["child"]["repo"]))
    for f in (ROOT / "results" / "large").glob("*.json*"):
        try:
            txt = open(f).read()
        except Exception: continue
        for l in txt.splitlines():
            try: r = json.loads(l)
            except Exception: continue
            if isinstance(r, dict) and "parent" in r and "child" in r:
                pa = r["parent"]["repo"] if isinstance(r["parent"], dict) else r["parent"]
                ch = r["child"]["repo"] if isinstance(r["child"], dict) else r["child"]
                ks.add((pa, ch))
    return ks

def main():
    have = covered(); sides = {}
    def S(r):
        if r not in sides: sides[r] = bp.side(r)
        return sides[r]
    cands = list(all_pairs())
    card_info = []
    for c in CARD:
        s = S(c); bm = s.get("card_base_model")
        if isinstance(bm, str): bm = [bm]
        card_info.append(dict(repo=c, base_model=bm, exists=s.get("exists"), gated=s.get("gated")))
        for b in (bm or [])[:2]:
            if b and b != c: cands.append((b, c, "card"))
    seen = set(); out, skipped = [], []
    for par, ch, kind in cands:
        if (par, ch) in seen: continue
        seen.add((par, ch))
        if (par, ch) in have:
            skipped.append(dict(parent=par, child=ch, why="already covered")); continue
        a, b = S(par), S(ch)
        bad = [x["repo"] + ": " + ("missing" if not x.get("exists") else "gated" if x.get("gated") else "no embedding tensor found" if not x.get("embed") else "")
               for x in (a, b) if not x.get("exists") or x.get("gated") or not x.get("embed")]
        if bad:
            skipped.append(dict(parent=par, child=ch, why="; ".join(bad))); continue
        cmp_ = bp.compare(a, b); ce = cmp_.get("embed") or {}
        if not ce.get("compatible"):
            skipped.append(dict(parent=par, child=ch, why="embedding not comparable: " + str(ce.get("why")), shapes=[a["embed"].get("shape"), b["embed"].get("shape")])); continue
        out.append(dict(key=f"{par}@ -> {ch}", parent=a, child=b, src=["y2026_labs:" + kind], compat=cmp_,
                        tied=bool(a.get("tied") or b.get("tied")), relation=kind, rank=0,
                        bytes=a["embed"]["bytes"] + b["embed"]["bytes"]))
        print("ok ", par, "->", ch, a["embed"]["shape"], a["embed"]["dtype"], b["embed"]["dtype"], "tied" if out[-1]["tied"] else "", flush=True)
    for s in skipped: print("skip", s["parent"], "->", s["child"], ":", s["why"][:120], flush=True)
    OD.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OD / "pairs.json", "w"), indent=1, default=str)
    json.dump(skipped, open(OD / "skipped.json", "w"), indent=1, default=str)
    json.dump(card_info, open(OD / "card_bases.json", "w"), indent=1, default=str)
    json.dump({k: dict(created=None, downloads=v.get("downloads"), card_base_model=v.get("card_base_model"), gated=v.get("gated"), exists=v.get("exists")) for k, v in sides.items()}, open(OD / "sides_meta.json", "w"), indent=1, default=str)
    print(len(out), "pairs;", len(skipped), "skipped; GB", round(sum(p["bytes"] for p in out) / 1e9, 1))

if __name__ == "__main__":
    main()
