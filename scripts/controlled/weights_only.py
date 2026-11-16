#!/usr/bin/env python3
"""Weights-only recipe reading for every single-epoch Pythia controlled run (tech review M1).

score.py's default reading may replace the step count by the one the claimed
document list and batch size imply (N = k * ceil(docs / batch)) when that is
within 2% and fits about as well. A third party usually lacks the batch size,
so this script re-reads every run with with_data=False (W0 and W1 only, plus the
bit-exact replay refinement, which also uses the weights only) and records
whether that reading is exact, next to what the data-implied N would have been.

  python3 scripts/controlled/weights_only.py runs/ctl/r01_lin runs/gaps/g01_swap_wd ...
Writes results/controlled/weights_only.json (merged across invocations).
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "controlled"))
sys.path.insert(0, str(ROOT))
import score as S  # noqa: E402

import os
OUT = pathlib.Path(os.environ.get("TF_WO_OUT", ROOT / "results" / "controlled" / "weights_only.json"))


def main():
    res = json.load(open(OUT)) if OUT.exists() else {}
    for r in sys.argv[1:]:
        run = pathlib.Path(r)
        if run.name in res:
            continue
        t = json.load(open(run / "truth.json"))
        a = t["args"]
        if a.get("epochs", 1) != 1:
            continue
        t0 = time.time()
        out = S.read(run, "fp32", with_data=False)
        rec = out["recipe"]
        from trainforensics.truth import Run
        R = Run(run)
        n1 = math.ceil(len(t["claimed"]) / (a["bs"] * a["ga"]))
        lr_read = S.hf_lr(dict(rec, peak=rec["peak_snap"]), rec["N"])
        exact = len(lr_read) == len(R.lr) and float(abs(lr_read[1:] - R.lr[1:]).max() / R.lr.max()) < 1e-9
        res[run.name] = dict(
            true=dict(kind=a["sched"], N=R.N, warmup=a["warmup"], peak=a["lr"], wd=a["wd"], seed=a["seed"]),
            read=dict(kind=rec["kind"], N=rec["N"], warmup=rec["warmup"], peak=rec["peak"],
                      peak_snap=rec["peak_snap"], wd=rec.get("wd"), wd_snap=rec.get("wd_snap")),
            schedule_exact_weights_only=bool(exact and (abs((rec.get("wd_snap") or 0) - (a["wd"] or 0)) < 1e-12)),
            N_exact_weights_only=rec["N"] == R.N,
            data_implied_N=n1, data_would_change_N=(n1 != rec["N"] and abs(n1 / rec["N"] - 1) <= 0.02),
            replay_refine=out.get("replay_refine"), seconds=round(time.time() - t0, 1))
        cur = json.load(open(OUT)) if OUT.exists() else {}
        cur[run.name] = res[run.name]
        json.dump(cur, open(OUT, "w"), indent=1, default=float)
        print(run.name, res[run.name]["schedule_exact_weights_only"], rec["kind"], rec["N"], "true", a["sched"], R.N, flush=True)


if __name__ == "__main__":
    main()
