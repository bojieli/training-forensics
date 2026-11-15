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
                v = f"P={p['P']:.3g}: too small for decay alone (rows zeroed or rescaled otherwise)"
            elif p["P"] == 1.0:
                v = "no decoupled decay on the embedding (" + p["init"] + ")"
            elif S is not None:
                v = f"sum lr*wd = {S:.2f}" + (" (init assumed = initializer_range)" if "assum" in p["verdict"] else "")
            else:
                v = "init std unknown"
            rec["pretraining"] = v
        else:
            kind = m.get("reason_kind") or m.get("status")
            label = {"fingerprint": "init read from trained rows only, none left untouched",
                     "rows-present": "untouched rows present but no searched seed regenerates them",
                     "framework": "no GPU layout in its rows (JAX / Paddle / CPU / Ascend init)",
                     "no-rows": "no untouched rows"}.get(kind, kind)
            import re as _re
            mm = _re.search(r"([\d,]+) untouched rows", m.get("reason") or "")
            if kind == "rows-present" and mm:
                rec["pretraining"] = (f"{mm.group(1)} never-trained rows (tokens absent from pretraining); "
                                      "init not regenerated, so no decay reading")
            else:
                rec["pretraining"] = f"no regenerable init: {label}"
        ch = as_child.get(repo, [])
        if ch:
            parts = []
            for r in ch:
                e = r["embed"]
                if r.get("tied"):
                    parts.append(f"vs {r['parent']}: tied (no record)")
                elif e.get("identical") == e.get("rows"):
                    parts.append(f"vs {r['parent']}: embedding unchanged")
                elif e.get("unmoved", 0) > 0:
                    s = f"vs {r['parent']}: {e['unmoved']:,} rows never trained"
                    if e.get("P", 1) < 1:
                        s += f", P={e['P']:.4f}"
                    ss = r.get("special_summary") or {}
                    if isinstance(ss, dict) and ss.get("n"):
                        s += f", control tokens touched {ss.get('moved', 0)}/{ss['n']}"
                    parts.append(s)
                else:
                    parts.append(f"vs {r['parent']}: every row moved")
            rec["finetune"] = "; ".join(parts)
        else:
            rec["finetune"] = "-" if not as_parent.get(repo) else f"parent of {len(as_parent[repo])} analysed fine-tunes"
        rec["readable"] = ((not rec["pretraining"].startswith("no regenerable")) or
                           (rec["finetune"] not in ("-",) and not rec["finetune"].startswith("parent of")))
        rows.append(rec)
    json.dump(rows, open(ROOT / "results" / "roster" / "coverage.json", "w"), indent=1)
    n = len(rows)
    n_pre = sum(1 for r in rows if not r["pretraining"].startswith("no regenerable"))
    n_rowsonly = sum(1 for r in rows if "never-trained rows (tokens absent" in r["pretraining"])
    n_ft = sum(1 for r in rows if r["finetune"] not in ("-",) and not r["finetune"].startswith("parent of"))
    n_par = sum(1 for r in rows if r["finetune"].startswith("parent of"))
    n_read = sum(1 for r in rows if r["readable"])
    n_any = sum(1 for r in rows if r["readable"] or r["finetune"].startswith("parent of"))
    why = Counter(r["pretraining"] for r in rows if r["pretraining"].startswith("no regenerable"))
    lines = ["# Coverage: every roster model", "",
             f"Generated by `scripts/roster/coverage.py`. {n} models. A model is covered when its own weights give a",
             "pretraining reading (never-trained rows regenerated from their initialisation) or it is the child of an analysed",
             "parent -> child pair; a model that is only a parent serves as the reference for its fine-tunes.", "",
             f"- pretraining reading: **{n_pre}** ({n_pre - n_rowsonly} with the decay of never-trained rows vs their regenerated init, {n_rowsonly} with the never-trained rows only)", f"- fine-tune reading (as child): **{n_ft}**",
             f"- models read (pretraining and/or as a fine-tune): **{n_read} / {n}**",
             f"- serving as the reference parent of an analysed fine-tune: {n_par} (of these, {n_any - n_read} are not read themselves)",
             f"- neither: {n - n_any}", "",
             "Models without a regenerable initialisation, by reason: " +
             ", ".join(f"{k.replace('no regenerable init: ', '')}: {v}" for k, v in why.most_common()), "",
             "| model | repo | pretraining (never-trained rows vs init) | fine-tune (vs parent) |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['repo']} | {r['pretraining']} | {r['finetune']} |")
    (ROOT / "results" / "roster" / "COVERAGE.md").write_text("\n".join(lines) + "\n")
    print(n, 'pre', n_pre, 'ft', n_ft, 'read', n_read, 'parent', n_par, 'any', n_any)
    print(why.most_common())


if __name__ == "__main__":
    main()
