#!/usr/bin/env python3
"""One row per roster model (all 265): what its weights say about training.

Combines
  - results/roster/pretrain_decay.json   (base models: decay of never-trained rows over pretraining)
  - results/roster/sweep*.jsonl          (fine-tunes: parent -> child embedding diff)
  - checkpoint-forensics results/summary.json (why a model has no regenerable init)
Writes results/roster/COVERAGE.md and coverage.json.
"""
from __future__ import annotations

import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
CK = pathlib.Path.home() / "checkpoint-forensics" / "results" / "summary.json"


def main():
    summ = json.load(open(CK))
    pre = {r["repo"]: r for r in json.load(open(ROOT / "results" / "roster" / "pretrain_decay.json"))}
    pre_by_name = {r["name"]: r for r in pre.values()}
    pairs = {}
    for fn in ("sweep.jsonl", "sweep_cml3.jsonl"):
        for line in open(ROOT / "results" / "roster" / fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                continue
            k = r["key"]
            if k not in pairs:
                pairs[k] = r
    as_child, as_parent = {}, {}
    for r in pairs.values():
        as_child.setdefault(r["child"], []).append(r)
        as_parent.setdefault(r["parent"], []).append(r)
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
            parts = []
            for r in ch:
                e = r["embed"]
                if r.get("tied"):
                    parts.append(f"vs {r['parent']}: tied input/output matrix (no record)")
                elif e.get("identical") == e.get("rows"):
                    parts.append(f"vs {r['parent']}: token vectors unchanged")
                elif e.get("unmoved", 0) > 0:
                    s = f"vs {r['parent']}: {e['unmoved']:,} tokens never updated"
                    if e.get("P", 1) < 1:
                        s += f", shrink factor P={e['P']:.4f}"
                    ss = r.get("special_summary") or {}
                    if isinstance(ss, dict) and ss.get("n"):
                        s += f", special tokens updated {ss.get('moved', 0)}/{ss['n']}"
                    parts.append(s)
                else:
                    parts.append(f"vs {r['parent']}: every token vector changed")
            rec["finetune"] = "; ".join(parts)
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
    (ROOT / "results" / "roster" / "COVERAGE.md").write_text("\n".join(lines) + "\n")
    print(n, 'pre', n_pre, 'ft', n_ft, 'read', n_read, 'parent', n_par, 'any', n_any)
    print(why.most_common())


if __name__ == "__main__":
    main()
