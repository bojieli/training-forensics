#!/usr/bin/env python3
"""Claim 5 and 7: comparability map of every analysed parent -> release comparison, and the
total weight decay ranges by family.

Inputs: results/roster/sweep*.jsonl, results/large/sweep_large.jsonl (the sweep: fitted shrink P,
`fit_reproduced`), results/frontier/recount/*.json (scripts/frontier/recount.py: per comparison
`band` = vectors consistent with "parent x shrink, nothing else" including fp32 rounding drift,
`left` = band vectors after removing unreachable, reserved/added, at-init, zero and duplicate rows,
`resolvable_shrink_pct`), results/frontier/calibration.json (thresholds from separately trained
models and sibling releases, scripts/frontier/calibrate.py) and results/roster/p_intervals.json.

What a class does and does not say (review B1): bit-identical token vectors show that two models
share ancestry. Separately trained runs from one random start share every vector neither run
could train (unreachable, reserved, still at the random start), so those are removed before
counting (`left`). Even then siblings - two releases trained from the same parent - share the
vectors neither training touched, so no class names a *direct* parent. Lineage statements are
made separately with a margin rule against every plausible candidate (scripts/frontier/lineage.py).

Classes:
  shared matrix          tied input/output embeddings: every vector is trained, no record
  unchanged              embedding bit-identical to the base
  token record           >= T_LEFT trainable vectors consistent with parent x shrink, and the shrink
                         fit reproduces >= 99% of numbers (or there is no resolvable shrink):
                         same lineage; *if* this parent is the starting point, the remaining
                         reachable tokens were not in the post-training input
  untrainable only       vectors consistent with parent x shrink exist, but fewer than T_LEFT
                         trainable ones: the shared vectors are tokens no run could train, which
                         separately trained models from the same random start also share. No
                         lineage evidence beyond a shared start; a token list only if the
                         parent is known from elsewhere
  record, poor fit       some trainable vectors bit-identical, but the shrink fit < 0.99
  changed throughout     no bit-identical trainable vectors; fit >= T_FIT (above every separately
                         trained null pair): consistent with the parent plus broad training. Below
                         the null maximum the class is "not comparable" (review M4: threshold
                         calibrated, no longer a fixed 0.5)
  not comparable         none of the above, or shrink factor implausible
For P = 1 in a 16-bit release, "no shrink" means "no shrink resolvable below x %", with x from the
recount (`resolvable_shrink_pct`).
Writes results/frontier/comparability.json, decay_ranges.json, comparability_before_after.json.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import OUT, ROOT  # noqa: E402

RECOUNT = OUT / "recount"
CAL = OUT / "calibration.json"
ORDER = ["token record", "untrainable only", "record, poor fit", "changed throughout", "not comparable",
         "unchanged", "shared matrix"]
SWEEPS = [ROOT / "results" / "roster" / f for f in
          ("sweep.jsonl", "sweep_cml3.jsonl", "sweep_frontier_x.jsonl", "sweep_2026_labs.jsonl", "sweep_gated.jsonl")] + \
         [ROOT / "results" / "large" / "sweep_large.jsonl"]


def _cal():
    d = dict(T_LEFT=50, T_FIT=0.5, source="defaults (calibration not run)")
    if CAL.exists():
        d.update(json.load(open(CAL)))
    return d


CALIB = _cal()


def safe_key(key):
    return key.replace("/", "__").replace(" -> ", "--").replace("@", "_")


def recount_for(r):
    f = RECOUNT / (safe_key(r["key"]) + ".json")
    return json.loads(f.read_text()) if f.exists() else None


def records(files=None):
    recs = {}
    for p in files or SWEEPS:
        if not p.exists():
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status", "ok") == "ok" and "embed" in r and r["key"] not in recs:
                r["_src_file"] = p.name
                recs[r["key"]] = r
    return recs


def family(repo):
    n = repo.split("/")[-1].lower()
    org = repo.split("/")[0].lower()
    for k, v in (("deepseek", "DeepSeek"), ("qwen", "Qwen"), ("llama", "Llama"), ("kimi", "Kimi"),
                 ("glm", "GLM"), ("mistral", "Mistral"), ("ministral", "Mistral"), ("magistral", "Mistral"),
                 ("nemotron", "NVIDIA Nemotron"), ("gemma", "Gemma"), ("olmo", "OLMo"), ("phi", "Phi"),
                 ("gpt-oss", "gpt-oss"), ("mimo", "MiMo"), ("minimax", "MiniMax"), ("step", "StepFun"),
                 ("seed", "Seed"), ("ling", "Ling"), ("intern", "InternLM"), ("pythia", "Pythia"),
                 ("llada", "LLaDA"), ("klear", "Klear")):
        if k in n or k in org:
            return v
    return org


def _fit(e):
    fit = e.get("P_exact_frac")
    return 0.0 if fit is None or (isinstance(fit, float) and math.isnan(fit)) else fit


def classify(r, rc=None, T_LEFT=None, T_FIT=None):
    """r: sweep record; rc: recount record (looked up by key when omitted)."""
    T_LEFT = CALIB["T_LEFT"] if T_LEFT is None else T_LEFT
    T_FIT = CALIB["T_FIT"] if T_FIT is None else T_FIT
    e = r.get("embed") or {}
    if r.get("tied"):
        return "shared matrix"
    if e.get("rows") and e.get("identical") == e.get("rows"):
        return "unchanged"
    if rc is None and "key" in r:
        rc = recount_for(r)
    if rc and rc.get("changed_vs_parent") and rc.get("roundtrip_explained", 0) >= 0.99 * rc["changed_vs_parent"]:
        return "unchanged"                 # a 16-bit format round trip, not training (review minor 10)
    fit = _fit(e)
    P = e.get("P", 1.0)
    bad_P = not (0.5 < P <= 1.0)
    if rc is not None:
        band, left = rc.get("band", 0), rc.get("left", 0)
    else:                                  # no recount yet: raw count, no cleaning (flagged)
        band, left = e.get("unmoved", 0), None
    no_shrink = abs(P - 1.0) < 1e-9
    if band > 0 and not bad_P:
        if fit >= 0.99 or no_shrink:
            if left is None:
                return "token record" if band >= T_LEFT else "untrainable only"
            return "token record" if left >= T_LEFT else "untrainable only"
        if left is not None and left < T_LEFT:
            return "untrainable only"
        return "record, poor fit"
    if fit >= T_FIT and not bad_P:
        return "changed throughout"
    return "not comparable"


def classify_old(r):
    """The pre-review rule (kept for the before/after table)."""
    e = r["embed"]
    if r.get("tied"):
        return "shared matrix"
    if e.get("identical") == e.get("rows"):
        return "unchanged"
    fit = _fit(e)
    P = e.get("P", 1.0)
    bad_P = not (0.5 < P <= 1.0001)
    if e.get("unmoved", 0) > 1 and not bad_P:
        if fit >= 0.99:
            return "token record"
        if abs(P - 1.0) < 1e-9 and e.get("unmoved", 0) >= 100:
            return "token record"
        return "record, poor fit"
    if fit >= 0.5 and not bad_P:
        return "changed throughout"
    return "not comparable"


LANES = {"sweep.jsonl": "roster", "sweep_cml3.jsonl": "roster", "sweep_frontier_x.jsonl": "frontier_x",
         "sweep_2026_labs.jsonl": "y2026_labs", "sweep_gated.jsonl": "gated", "sweep_large.jsonl": "large"}


def main():
    recs = records()
    rows = []
    for r in recs.values():
        e = r["embed"]
        rc = recount_for(r)
        rows.append(dict(key=r["key"], lane=LANES.get(r["_src_file"], r["_src_file"]), parent=r["parent"],
                         child=r["child"], family=family(r["child"]), cls=classify(r, rc), cls_before=classify_old(r),
                         fit_reproduced=e.get("P_exact_frac"), P=e.get("P"), bit_identical_raw=e.get("unmoved"),
                         consistent_with_shrink=rc and rc.get("band"), trainable_left=rc and rc.get("left"),
                         removed=rc and rc.get("removed"), zeroed_edits=rc and rc.get("zeroed_edits"),
                         recounted=rc is not None, vocab=e.get("rows"),
                         shrink_pct=(100 * (1 - e["P"]) if e.get("P") else None),
                         no_shrink_resolvable_below_pct=(rc.get("resolvable_shrink_pct")
                                                         if rc and abs((e.get("P") or 1) - 1) < 1e-9 else None),
                         detection_floor_10=rc and rc.get("floor_10"), detection_floor_10_rel=rc and rc.get("floor_10_rel"),
                         hist_band=rc and rc.get("hist_band")))
    rows.sort(key=lambda x: (ORDER.index(x["cls"]) if x["cls"] in ORDER else 99, x["family"], x["child"]))
    json.dump(rows, open(OUT / "comparability.json", "w"), indent=1)
    ba = {}
    for lane in sorted({x["lane"] for x in rows}):
        sub = [x for x in rows if x["lane"] == lane]
        ba[lane] = dict(n=len(sub), recounted=sum(x["recounted"] for x in sub),
                        before=dict(Counter(x["cls_before"] for x in sub)), after=dict(Counter(x["cls"] for x in sub)),
                        moved=dict(Counter(f"{x['cls_before']} -> {x['cls']}" for x in sub if x["cls_before"] != x["cls"])))
    ba["calibration"] = CALIB
    json.dump(ba, open(OUT / "comparability_before_after.json", "w"), indent=1)
    for k, v in ba.items():
        print(k, v)
    pi = json.load(open(ROOT / "results" / "roster" / "p_intervals.json"))
    dr = []
    for x in pi:
        if x.get("error") or not x.get("consistent") or x.get("weak") or x.get("mode") == "partial rows":
            continue
        lo, hi = x.get("P_lo_drift", x["P_lo"]), x.get("P_hi_drift", x["P_hi"])
        dr.append(dict(child=x["child"], parent=x["parent"], family=family(x["child"]), mode=x["mode"],
                       vectors=x["rows"], P_lo=lo, P_hi=hi, P_lo_one_mult=x["P_lo"], P_hi_one_mult=x["P_hi"],
                       shrink_pct_lo=100 * (1 - hi), shrink_pct_hi=100 * (1 - lo),
                       total_wd_lo=(-math.log(hi) if hi < 1 else 0.0), total_wd_hi=-math.log(lo),
                       what="range of P that reproduces the same vectors, widened by fp32 rounding drift",
                       one_16bit_step=(x.get("ulp_steps") and set(map(str, x["ulp_steps"].keys())) <= {"1", "0"})))
    dr.sort(key=lambda x: x["shrink_pct_lo"])
    json.dump(dr, open(OUT / "decay_ranges.json", "w"), indent=1)
    print(len(dr), "decay ranges (partial-row cases excluded)")


if __name__ == "__main__":
    main()
