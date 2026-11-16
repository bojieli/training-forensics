#!/usr/bin/env python3
"""Claim 5 and 7: comparability map of every analysed base -> release comparison, and the
total weight decay ranges by family. Uses only results/roster/sweep*.jsonl and p_intervals.json.

Per comparison, `fit_reproduced` is the sweep's check of its fitted shrink factor P: the
fraction of numbers, in the 512 token vectors closest to a pure shrink, that round(P*base)
reproduces exactly. It is lenient (best-case vectors), so a low value is strong evidence
that the release is not a decayed copy of that base.

Classes:
  shared matrix        tied input/output embeddings: every vector is trained, no record
  unchanged            embedding bit-identical to the base
  token record         some vectors bit-identical after removing decay AND fit >= 0.99; or no decay
                       (shrink exactly 1) and >= 100 bit-identical vectors: the 512-vector fit score is
                       then capped by how few vectors are untouched, not a sign of a different base
                       (results/frontier_ds/RECONCILE.md section 1)
  record, poor fit     some vectors bit-identical but fit < 0.99 (treat lists with care)
  changed throughout   no bit-identical vectors, fit >= 0.5: same base, broad training (+ decay)
  not comparable       no bit-identical vectors and fit < 0.5 or shrink factor implausible
Writes results/frontier/comparability.json, decay_ranges.json.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import OUT, ROOT, records  # noqa: E402

MIN_IDENT_NO_DECAY = 100
ORDER = ["token record", "record, poor fit", "changed throughout", "not comparable", "unchanged", "shared matrix"]


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


def classify(r):
    e = r["embed"]
    if r.get("tied"):
        return "shared matrix"
    if e.get("identical") == e.get("rows"):
        return "unchanged"
    fit = e.get("P_exact_frac")
    P = e.get("P", 1.0)
    fit = 0.0 if fit is None or (isinstance(fit, float) and math.isnan(fit)) else fit
    bad_P = not (0.5 < P <= 1.0001)
    if e.get("unmoved", 0) > 1 and not bad_P:
        if fit >= 0.99:
            return "token record"
        # no decay: bit-identical vectors are exact copies of the base, which only a
        # continuation of that base produces; the fit score is capped by their scarcity
        if abs(P - 1.0) < 1e-9 and e.get("unmoved", 0) >= MIN_IDENT_NO_DECAY:
            return "token record"
        return "record, poor fit"
    if fit >= 0.5 and not bad_P:
        return "changed throughout"
    return "not comparable"


def main():
    recs = records()
    rows = []
    for r in recs.values():
        e = r["embed"]
        rows.append(dict(parent=r["parent"], child=r["child"], family=family(r["child"]), cls=classify(r),
                         fit_reproduced=e.get("P_exact_frac"), P=e.get("P"), bit_identical=e.get("unmoved"),
                         vocab=e.get("rows"), shrink_pct=(100 * (1 - e["P"]) if e.get("P") else None)))
    rows.sort(key=lambda x: (ORDER.index(x["cls"]), x["family"], x["child"]))
    json.dump(rows, open(OUT / "comparability.json", "w"), indent=1)
    c = Counter(x["cls"] for x in rows)
    print(len(rows), {k: c[k] for k in ORDER})
    # decay ranges: exact intervals from p_intervals.json (full-row mode only)
    pi = json.load(open(ROOT / "results" / "roster" / "p_intervals.json"))
    dr = []
    for x in pi:
        if x.get("error") or not x.get("consistent"):
            continue
        dr.append(dict(child=x["child"], parent=x["parent"], family=family(x["child"]), mode=x["mode"],
                       vectors=x["rows"], P_lo=x["P_lo"], P_hi=x["P_hi"],
                       shrink_pct_lo=100 * (1 - x["P_hi"]), shrink_pct_hi=100 * (1 - x["P_lo"]),
                       total_wd_lo=x["sum_lr_wd_lo"], total_wd_hi=x["sum_lr_wd_hi"],
                       one_16bit_step=(x.get("ulp_steps") and set(map(str, x["ulp_steps"].keys())) <= {"1", "0"})))
    dr.sort(key=lambda x: x["shrink_pct_lo"])
    json.dump(dr, open(OUT / "decay_ranges.json", "w"), indent=1)
    for x in dr:
        print(f"{x['child'][:48]:48s} shrink {x['shrink_pct_lo']:.3f}-{x['shrink_pct_hi']:.3f}% {x['mode']} n={x['vectors']} 1step={x['one_16bit_step']}")


if __name__ == "__main__":
    main()
