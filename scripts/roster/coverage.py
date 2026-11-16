#!/usr/bin/env python3
"""One row per roster model (all 265): what its weights say about training.

Combines
  - results/roster/pretrain_decay.json   (base models: decay of never-trained rows over pretraining)
  - every parent -> fine-tune comparison: results/roster/sweep.jsonl, sweep_cml3.jsonl,
    sweep_frontier_x.jsonl, sweep_2026_labs.jsonl, results/large/sweep_large.jsonl and the
    results/y2026_ft/pairs/*.json records, deduplicated by (parent, fine-tune), first source wins
    (the roster sweep first), classified with scripts/frontier/comparability.py
  - checkpoint-forensics results/summary.json (why a model has no regenerable init)
Writes results/roster/COVERAGE.md and coverage.json.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import glob
import json
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
CK = paths.CKPTF_SUMMARY


SWEEPS = [("roster", "results/roster/sweep.jsonl"), ("roster", "results/roster/sweep_cml3.jsonl"),
          ("frontier_x", "results/roster/sweep_frontier_x.jsonl"), ("y2026_labs", "results/roster/sweep_2026_labs.jsonl"),
          ("large", "results/large/sweep_large.jsonl")]
# display order only: the classes come from scripts/frontier/comparability.py (or a y2026_ft
# record's own "cls"), so a reworked classifier is picked up on rerun
CLASSES = ["token record", "record, poor fit", "changed throughout", "not comparable", "unchanged", "shared matrix"]


def all_comparisons():
    """Every analysed parent -> fine-tune comparison, deduplicated by (parent, child)."""
    sys.path.insert(0, str(ROOT / "scripts" / "frontier"))
    from comparability import classify  # noqa: E402
    out = {}
    for src, fn in SWEEPS:
        for line in open(ROOT / fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                continue
            k = (r["parent"], r["child"])
            if k not in out:
                out[k] = dict(parent=r["parent"], child=r["child"], source=src, cls=classify(r), rec=r)
    for f in sorted(glob.glob(str(ROOT / "results" / "y2026_ft" / "pairs" / "*.json"))):
        d = json.load(open(f))
        if not d.get("parent") or not d.get("cls") or str(d.get("status", "")).startswith("error"):
            continue
        k = (d["parent"], d["child"])
        if k not in out:
            out[k] = dict(parent=d["parent"], child=d["child"], source="y2026_ft", cls=d["cls"], rec=None, ft=d)
    return list(out.values())


def describe(c):
    """One-line fine-tune reading of a comparison, from its class and record."""
    r, cls, par = c.get("rec"), c["cls"], c["parent"]
    if cls == "shared matrix":
        return f"vs {par}: tied input/output matrix (no record)"
    if cls == "unchanged":
        return f"vs {par}: token vectors unchanged"
    if cls == "not comparable":
        return f"vs {par}: not a decayed copy of that parent (not comparable)"
    if cls == "changed throughout":
        return f"vs {par}: every token vector changed"
    if r is None:
        d = c["ft"]
        return f"vs {par}: {d.get('bit_identical', 0):,} tokens never updated ({cls})"
    e = r["embed"]
    s = f"vs {par}: {e.get('unmoved', 0):,} tokens never updated"
    if e.get("P", 1) < 1:
        s += f", shrink factor P={e['P']:.4f}"
    ss = r.get("special_summary") or {}
    if isinstance(ss, dict) and ss.get("n"):
        s += f", special tokens updated {ss.get('moved', 0)}/{ss['n']}"
    if cls == "record, poor fit":
        s += " (decay fit below 99%: read with care)"
    return s


def main():
    summ = json.load(open(CK))
    comps = all_comparisons()
    pre = {r["repo"]: r for r in json.load(open(ROOT / "results" / "roster" / "pretrain_decay.json"))}
    pre_by_name = {r["name"]: r for r in pre.values()}
    as_child, as_parent = {}, {}
    for c in comps:
        as_child.setdefault(c["child"], []).append(c)
        as_parent.setdefault(c["parent"], []).append(c)
    rows = []
    for m in summ:
        repo, name = m["repo"], m["name"]
        rec = dict(name=name, repo=repo)
        p = pre.get(repo) or pre_by_name.get(name)
        if p:
            S = p["sum_lr_wd"]
            if S is not None and S > 15:
                v = f"P={p['P']:.3g}: implausible for decay alone (vectors zeroed or rescaled otherwise)"
            elif p["P"] == 1.0:
                v = ("no decoupled weight decay on the token vectors (" + p["init"] + ")"
                     + (", inferred from a round number" if "round" in p["verdict"] else ""))
            elif S is not None:
                v = (f"total weight decay sum lr*wd = {S:.2f}"
                     + (" (assumes init std = initializer_range)" if "assum" in p["verdict"] else " (validated)"))
            else:
                v = "init std unknown"
            rec["pretraining"] = v
        else:
            kind = m.get("reason_kind") or m.get("status")
            label = {"fingerprint": "init recognised from trained vectors only, none left untouched",
                     "rows-present": "untouched vectors present but no searched seed regenerates them",
                     "framework": "no GPU random-number layout in its vectors (JAX / Paddle / CPU / Ascend init)",
                     "no-rows": "no untouched vectors"}.get(kind, kind)
            import re as _re
            mm = _re.search(r"([\d,]+) untouched rows", m.get("reason") or "")
            if kind == "rows-present" and mm:
                rec["pretraining"] = (f"{mm.group(1)} never-trained token vectors (tokens absent from pretraining); "
                                      "starting values not regenerated, so no decay estimate")
            else:
                rec["pretraining"] = f"no regenerable init: {label}"
        ch = as_child.get(repo, [])
        if ch:
            rec["finetune"] = "; ".join(describe(c) for c in ch)
        else:
            rec["finetune"] = "-" if not as_parent.get(repo) else f"parent of {len(as_parent[repo])} analysed fine-tunes"
        rec["readable"] = ((not rec["pretraining"].startswith("no regenerable")) or
                           (rec["finetune"] not in ("-",) and not rec["finetune"].startswith("parent of")))
        rows.append(rec)
    json.dump(rows, open(ROOT / "results" / "roster" / "coverage.json", "w"), indent=1)
    n = len(rows)
    n_pre = sum(1 for r in rows if not r["pretraining"].startswith("no regenerable"))
    n_rowsonly = sum(1 for r in rows if "never-trained token vectors (tokens absent" in r["pretraining"])
    n_ft = sum(1 for r in rows if r["finetune"] not in ("-",) and not r["finetune"].startswith("parent of"))
    n_par = sum(1 for r in rows if r["finetune"].startswith("parent of"))
    n_read = sum(1 for r in rows if r["readable"])
    n_any = sum(1 for r in rows if r["readable"] or r["finetune"].startswith("parent of"))
    why = Counter(r["pretraining"] for r in rows if r["pretraining"].startswith("no regenerable"))
    pre_rows = [r for r in rows if not r["pretraining"].startswith("no regenerable")]
    k_val = sum(1 for r in pre_rows if r["pretraining"].startswith("total weight decay") and "(validated)" in r["pretraining"])
    k_ass = sum(1 for r in pre_rows if r["pretraining"].startswith("total weight decay") and "assumes" in r["pretraining"])
    k_exact = sum(1 for r in pre_rows if r["pretraining"].startswith("no decoupled") and "round number" not in r["pretraining"])
    k_round = sum(1 for r in pre_rows if "round number" in r["pretraining"])
    k_impl = sum(1 for r in pre_rows if "implausible" in r["pretraining"])
    k_unk = sum(1 for r in pre_rows if r["pretraining"] == "init std unknown")
    lines = ["# Coverage: every roster model", "",
             f"Generated by `scripts/roster/coverage.py`. {n} models. Every token in a model's vocabulary has its own vector",
             "of numbers. A model is covered when its own weights give a pretraining estimate (token vectors that never trained,",
             "compared with their regenerated random starting values) or it is the fine-tune in an analysed parent -> fine-tune",
             "pair; a model that is only a parent serves as the reference for its fine-tunes.", "",
             f"- pretraining estimate: **{n_pre}** ({n_pre - n_rowsonly} where the never-trained vectors could be compared with "
             f"their regenerated starting values, {n_rowsonly} where only the never-trained tokens are known). Of the {n_pre - n_rowsonly}: "
             f"{k_val} validated decay value (Pythia), {k_exact} exactly no decay, {k_round} no decay inferred from a round number, "
             f"{k_ass} decay value that assumes the config's init std, {k_impl} implausible for decay alone, {k_unk} init std unknown (no value)",
             f"- fine-tune comparison (as the fine-tune): **{n_ft}**",
             f"- models with either: **{n_read} / {n}**",
             f"- serving as the reference parent of an analysed fine-tune: {n_par} (of these, {n_any - n_read} have neither themselves)",
             f"- neither: {n - n_any}", "",
             "Models whose starting values cannot be regenerated, by reason: " +
             ", ".join(f"{k.replace('no regenerable init: ', '')}: {v}" for k, v in why.most_common()), "",
             "| model | repo | pretraining (never-trained token vectors vs their starting values) | fine-tune (vs parent) |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['repo']} | {r['pretraining']} | {r['finetune']} |")
    # every comparison, roster and beyond
    models = {c["parent"] for c in comps} | {c["child"] for c in comps}
    cc = Counter(c["cls"] for c in comps)
    by_src = Counter(c["source"] for c in comps)
    roster_repos = {m["repo"] for m in summ}
    off = sum(1 for c in comps if c["child"] not in roster_repos)
    lines += ["", "## Every parent -> fine-tune comparison (roster and follow-up studies)", "",
              f"{len(comps)} distinct comparisons over {len(models)} distinct models "
              f"({off} of them with a fine-tune outside the 265-model roster). By source (first source wins): "
              + ", ".join(f"{k} {v}" for k, v in by_src.most_common()) + ".", "",
              "| class | comparisons |", "|---|---|"]
    order = [k for k in CLASSES if k in cc] + sorted(k for k in cc if k not in CLASSES)
    lines += [f"| {k} | {cc[k]} |" for k in order]
    lines += ["", "| parent | fine-tune | source | class |", "|---|---|---|---|"]
    for c in sorted(comps, key=lambda c: (order.index(c["cls"]), c["child"])):
        lines.append(f"| {c['parent']} | {c['child']} | {c['source']} | {c['cls']} |")
    json.dump([{k: v for k, v in c.items() if k not in ("rec", "ft")} for c in comps],
              open(ROOT / "results" / "roster" / "comparisons_all.json", "w"), indent=1)
    print("comparisons", len(comps), "models", len(models), dict(cc), dict(by_src))
    (ROOT / "results" / "roster" / "COVERAGE.md").write_text("\n".join(lines) + "\n")
    print(n, 'pre', n_pre, 'ft', n_ft, 'read', n_read, 'parent', n_par, 'any', n_any)
    print(why.most_common())


if __name__ == "__main__":
    main()
