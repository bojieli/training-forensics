#!/usr/bin/env python3
"""Run the roster sweep on the 2026 lab pairs, adding per-token change-fraction histograms.

Writes results/roster/sweep_frontier_x.jsonl (same record format as sweep.jsonl, plus
`change_hist`: share of token vectors by fraction of numbers that differ from the decayed
base, round(P*base)), and logs every cache file it created to results/frontier_x/new_cache.txt.
"""
import argparse, concurrent.futures as cf, glob, importlib.util, json, os, pathlib, sys, time, traceback
ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("CKPTF_CACHE", str(pathlib.Path.home() / ".cache" / "trainforensics" / "emb"))
CACHE = pathlib.Path(os.environ["CKPTF_CACHE"])
spec = importlib.util.spec_from_file_location("sweep", ROOT / "scripts" / "roster" / "sweep.py")
sw = importlib.util.module_from_spec(spec); spec.loader.exec_module(sw)
import numpy as np, torch  # noqa
OUT = ROOT / "results" / "roster" / "sweep_2026_labs.jsonl"
NEW = ROOT / "results" / "y2026_labs" / "new_cache.txt"
BINS = [0, 1e-12, 0.01, 0.10, 0.50, 1.0001]
LAB = ["0 (bit-identical)", "<=1%", "1-10%", "10-50%", ">50%"]


def hist(p, rec):
    ce = p["compat"]["embed"]; r0, r1 = ce["common_rows"]
    W0 = sw.fetch(p["parent"])[r0:r1]; W1 = sw.fetch(p["child"])[r0:r1]   # cache hits
    cdt = sw.TD[ce["dtype_child"]]
    P = rec["embed"]["P"]
    nd, _, _ = sw.row_stats(W0, W1.to(cdt), cdt, P)
    f = nd / W0.shape[1]
    c, _ = np.histogram(f, bins=BINS)
    # raw (no decay removed) bit-identical count too
    return dict(labels=LAB, counts=c.tolist(), frac=(c / len(f)).tolist())


def work(p, head):
    try:
        before = set(glob.glob(str(CACHE / "*")))
        rec = sw.run_pair(p, head)
        if rec.get("status") == "ok":
            rec["change_hist"] = hist(p, rec)
        rec["src"] = p["src"]; rec["relation"] = p.get("relation")
        rec["_new_cache"] = sorted(set(glob.glob(str(CACHE / "*"))) - before)
        return rec
    except Exception as e:
        return dict(key=p["key"], parent=p["parent"]["repo"], child=p["child"]["repo"], src=p["src"], status="error",
                    error=f"{type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-1500:])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--head-max-gb", type=float, default=0.0); ap.add_argument("--max-gb", type=float, default=12)
    a = ap.parse_args()
    pairs = json.load(open(ROOT / "results" / "y2026_labs" / "pairs.json"))
    done = set()
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l)
            if r.get("status") == "ok":
                done.add(r["key"])
    todo = [p for p in pairs if p["key"] not in done and p["bytes"] <= a.max_gb * 1e9]
    big = [p["key"] for p in pairs if p["bytes"] > a.max_gb * 1e9]
    print(len(pairs), "pairs", len(todo), "to run; too big:", big, flush=True)
    # order: small first; run successive chains in order so shared models are cached once
    todo.sort(key=lambda p: p["bytes"])
    with cf.ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(work, p, a.head_max_gb): p for p in todo}
        for fu in cf.as_completed(futs):
            r = fu.result()
            nc = r.pop("_new_cache", [])
            with open(NEW, "a") as f:
                f.writelines(x + "\n" for x in nc)
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            e = r.get("embed", {})
            print(f'[{r["status"][:20]}] {r["key"]} unmoved={e.get("unmoved")} near={e.get("near")} moved={e.get("moved")} '
                  f'P={e.get("P")} fit={e.get("P_exact_frac")} hist={r.get("change_hist", {}).get("frac")} {r.get("error", "")[:150]}', flush=True)


if __name__ == "__main__":
    main()
