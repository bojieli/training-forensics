#!/usr/bin/env python3
"""Token-category mining (scripts/tokens/mine.py) on the large-fine-tune pairs.
Reuses mine.run() unchanged; points it at results/large/pairs_large.json and
results/large/sweep_large.jsonl and writes results/large/tokens/*.json."""
import argparse, concurrent.futures as cf, json, pathlib, sys, traceback
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tokens")); sys.path.insert(0, str(ROOT / "scripts" / "roster"))
import mine, sweep  # noqa

PAIRS = json.load(open(ROOT / "results" / "large" / "pairs_large.json"))
mine.OUT = ROOT / "results" / "large" / "tokens"
sweep.load_pairs = lambda: PAIRS


def records():
    R = {}
    for l in open(ROOT / "results" / "large" / "sweep_large.jsonl"):
        r = json.loads(l)
        if r.get("status") == "ok":
            R[r["key"]] = r
    return [r for r in R.values() if not r["tied"] and mine.E(r)["unmoved"] > 0 and mine.E(r)["moved"] > 0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=2); a = ap.parse_args()
    rs = sorted(records(), key=lambda r: r["emb_gb"])
    print(len(rs), "pairs", flush=True)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        fs = {ex.submit(mine.run, r): r["key"] for r in rs}
        for f in cf.as_completed(fs):
            try:
                print(*f.result(), flush=True)
            except Exception:
                print(fs[f], "ERROR", traceback.format_exc().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
