#!/usr/bin/env python3
"""Token-category mining (scripts/tokens/mine.py) on the 2026 lab comparisons with a token record."""
import concurrent.futures as cf, json, pathlib, sys, traceback
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tokens")); sys.path.insert(0, str(ROOT / "scripts" / "roster"))
import mine  # noqa
import sweep  # noqa
mine.OUT = ROOT / "results" / "y2026_labs" / "tokens"
PAIRS = {p["key"]: p for p in json.load(open(ROOT / "results" / "y2026_labs" / "pairs.json"))}
sweep.load_pairs = lambda: list(PAIRS.values())

def recs():
    R = {}
    for l in open(ROOT / "results" / "roster" / "sweep_2026_labs.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok": R[r["key"]] = r
    return [r for r in R.values() if not r["tied"] and mine.E(r)["unmoved"] >= 10 and mine.E(r)["moved"] > 0]

def one(r):
    import mine as m, sweep as s
    m.OUT = ROOT / "results" / "y2026_labs" / "tokens"
    s.load_pairs = lambda: list(PAIRS.values())
    return m.run(r)

if __name__ == "__main__":
    rs = recs(); print(len(rs), "pairs", flush=True)
    with cf.ProcessPoolExecutor(3) as ex:
        fs = {ex.submit(one, r): r["key"] for r in rs}
        for f in cf.as_completed(fs):
            try: print(*f.result(), flush=True)
            except Exception: print(fs[f], "ERROR", traceback.format_exc().splitlines()[-1], flush=True)
