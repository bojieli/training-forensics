#!/usr/bin/env python3
"""results/quant_merge/*.json(l) -> results/quant_merge/QUANT_MERGE.md (wording per STYLE.md)."""
from __future__ import annotations

import glob
import json
import pathlib
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results" / "quant_merge"
FRONTIER = {"DeepSeek", "Qwen3", "Llama-4", "gpt-oss", "Kimi", "GLM", "Qwen2.5-72B", "Llama-3.3"}


def load_quant():
    recs = {}
    files = [R / "quant.json"] + sorted(pathlib.Path(p) for p in glob.glob(str(R / "quant_*.jsonl")))
    for f in files:
        if not f.exists():
            continue
        rows = json.load(open(f)) if f.suffix == ".json" else [json.loads(l) for l in open(f) if l.strip()]
        for r in rows:
            recs[(r["family"], r["repo"])] = r
    return list(recs.values())


def decay_only():
    """family name -> (P, never-updated vectors) for official fine-tunes whose never-updated vectors
    only shrank by weight decay (P < 1). For these, 'every vector differs from the base' says nothing
    about data: see results/frontier_ds/RECONCILE.md section 3."""
    out = {}
    for f in sorted(glob.glob(str(ROOT / "results" / "roster" / "sweep*.jsonl"))):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            e = r.get("embed") or {}
            if r.get("status") != "ok" or r.get("tied") or not e:
                continue
            P, n = e.get("P", 1.0), e.get("unmoved", 0)
            if P is not None and 0.5 < P < 1 - 1e-9 and n > 0:
                out.setdefault(r["child"].split("/")[-1], (P, n))
    return out


DECAY_ONLY = None


def found(s):
    """'updated tokens detected / truly updated' (+ false flags) for a tp/fp/fn score."""
    if not s:
        return None
    t = f"{s['tp']:,} / {s['tp'] + s['fn']:,}"
    if s["fp"]:
        t += f", {s['fp']:,} false"
    return t


def stored(t):
    s = t.get("stored", "-")
    return {"absent (tied)": "not in this file", "absent (tied or renamed)": "not in this file"}.get(s, s)


def verdict(r):
    """One plain-language outcome per release, from the input embedding."""
    if r.get("error"):
        return "not checked", "error: " + r["error"][:90]
    e = r["tensors"].get("embed", {})
    st = e.get("stored", "")
    if st in ("F16", "BF16", "F32"):
        if e.get("equals_release_cast"):
            return "identical", f"not quantized ({st}), every number equal to the official release"
        return "changed", f"not quantized ({st}), {e.get('rows_equal_release', 0):,} of {e.get('rows', 0):,} token vectors equal to the official release"
    for k, how in (("read_vs_requantized_bf16_base", "vs the 16-bit base quantized by us"),
                   ("read_vs_same_tool_base", "vs the base quantized with the same tool")):
        if e.get(k):
            s = e[k]
            global DECAY_ONLY
            if DECAY_ONLY is None:
                DECAY_ONLY = decay_only()
            dk = DECAY_ONLY.get(r["family"])
            if dk and s["fn"] == 0 and s["fp"] == 0 and s["tp"] == e.get("rows", s["tp"]):
                P, n = dk
                return "not assessed", (f"quantized ({st}); every token vector differs from the base quantized by us, as in the "
                                        f"16-bit release, where all vectors differ because the {n:,} never-updated ones shrank by "
                                        f"{100 * (1 - P):.2f}% (weight decay). This test compares against the unshrunk base, so it "
                                        "cannot separate those decay-only vectors from trained ones; token record not assessed")
            return ("preserved" if s["fp"] == 0 else "degraded"), f"quantized ({st}); updated tokens detected {found(s)} {how}"
    rq = e.get("requant_from_bf16_rows_exact")
    if rq and rq.get("frac", 0) >= 0.999:
        return "reproducible", f"quantized ({st}), reproduced bit for bit from the official 16-bit release (no public base to score against)"
    if rq:
        return "unscored", f"quantized ({st}); our re-quantization matches {100 * rq['frac']:.1f}% of token vectors; no same-tool base"
    if not st:
        return "not checked", "embedding not found"
    return "unscored", f"quantized ({st}); no base quantized the same way to compare against"


def quant_section():
    Q = load_quant()
    if not Q:
        return []
    cnt = Counter(verdict(r)[0] for r in Q)
    fam = Counter(r["family"] for r in Q)
    out = ["## 1. Quantized releases: is the token record intact in what people download?", "",
           "Every token in a model's vocabulary has its own vector of numbers (its embedding). If a token never appears",
           "in fine-tuning data, its vector is never updated, so a fine-tune's token record is the set of token vectors",
           "that differ from the base model. We checked the quantized copies people actually download (official FP8 /",
           "INT4 / NVFP4 / MXFP4 releases, AWQ / GPTQ, bitsandbytes, GGUF for llama.cpp, MLX) against the official 16-bit",
           "release and, where a public base model exists, against the true record (official 16-bit fine-tune vs base).", "",
           f"**{len(Q)} quantized releases of {len(fam)} models: " + ", ".join(f"{v} {k}" for k, v in cnt.most_common()) + ".**", "",
           "- *identical*: the input embedding is not quantized (16- or 32-bit) and equals the official release number for number.",
           "- *preserved*: the input embedding is quantized, yet comparing it with the base model quantized the same way",
           "  recovers the updated tokens with no false flags (a few tokens with tiny updates are missed; counts in detail).",
           "- *reproducible*: quantized, and we reproduce it bit for bit from the official 16-bit release, so it carries",
           "  exactly the official record (no public base model to score against).",
           "- *unscored*: quantized, and no base model quantized the same way exists to compare against.",
           "- *not assessed*: the official fine-tune applied weight decay, so every vector differs from the unshrunk base",
           "  even where no token was trained; comparing against the unshrunk base cannot show the record.",
           "", "Counts are counts of distinct vocabulary tokens, not amounts of text.", "",
           "| model | quantized release | input embedding / output matrix stored as | outcome | detail |",
           "|---|---|---|---|---|"]
    order = sorted(Q, key=lambda r: (not any(f in r["family"] for f in FRONTIER), r["family"], r["repo"]))
    for r in order:
        e, h = r["tensors"].get("embed", {}), r["tensors"].get("head", {})
        v, d = verdict(r)
        hs = stored(h)
        if h.get("equals_release_cast") is not None:
            hs += " (equal to official)" if h["equals_release_cast"] else " (differs from official)"
        out.append(f"| {r['family']} | {r['repo']} | {stored(e)} / {hs} | {v} | {d} |")
    lp = R / "quant_lineage.json"
    if lp.exists():
        L = json.load(open(lp))
        if L:
            out += ["", "16-bit embeddings that differ from the release they are named after, compared with sibling releases",
                    "(`scripts/quant_merge/lineage_check.py`):", ""]
            for x in L:
                best = max(x["matches"], key=lambda m: m["token_vectors_equal"]) if x["matches"] else None
                if best and best["token_vectors_equal"] == best["of"]:
                    out.append(f"- {x['repo']} (named after {x['named_after']}): its input embedding equals "
                               f"{best['release']} for all {best['of']:,} vocabulary tokens, so it was built from that release.")
                else:
                    out.append(f"- {x['repo']} (named after {x['named_after']}): no checked release matches; best "
                               + (f"{best['release']} ({best['token_vectors_equal']:,} of {best['of']:,} tokens)" if best else "none"))
    # 16-bit releases with a base: the record against the 16-bit base
    sc = [(r, r["tensors"]["embed"]["read_vs_bf16_base"]) for r in Q
          if r["tensors"].get("embed", {}).get("read_vs_bf16_base")]
    if sc:
        out += ["", "16-bit embeddings in quantized releases, record vs the 16-bit base (updated tokens detected / truly updated):", ""]
        out += [f"- {r['repo']}: {found(s)}" for r, s in sc]
        allup = sorted({r["family"] for r, s in sc if s["tp"] + s["fn"] == s["tp"] + s["fn"] + 0 and
                        s.get("true_unmoved", 1) == 0})
        global DECAY_ONLY
        if DECAY_ONLY is None:
            DECAY_ONLY = decay_only()
        shrunk = [f for f in allup if f in DECAY_ONLY]
        allup = [f for f in allup if f not in DECAY_ONLY]
        if allup:
            out += ["", "In " + ", ".join(allup) + " every token vector of the official fine-tune differs from its base and no",
                    "uniform shrink explains the difference, so the record there is 'all tokens'."]
        if shrunk:
            out += ["", "In " + ", ".join(shrunk) + " every vector also differs from the base, but only because the official",
                    "fine-tune applied weight decay: " + "; ".join(f"{f}: {DECAY_ONLY[f][1]:,} never-updated vectors shrank by "
                                                              f"{100 * (1 - DECAY_ONLY[f][0]):.2f}%" for f in shrunk) + ".",
                    "This comparison is against the unshrunk base, so the token record is not assessed there."]
        out += ["", "The informative cases are those with never-updated tokens: " +
                ", ".join(sorted({r["family"] for r, s in sc if s.get("true_unmoved", 0) > 0})) + "."]
        miss = [r for r, s in sc if s["fn"]]
        if miss:
            out += ["", "The tokens float16 copies miss were updated in only one or two of their numbers, each smaller than",
                    "4e-6, where float16 loses precision (checked for Qwen2.5-7B-Instruct-AWQ); bfloat16 copies keep them."]
    return out + [""]


def W(d):
    return ", ".join(f"{k.split('/')[-1]} {v:.3g}" for k, v in d.items()) if d else "-"


def merge_section():
    out = ["", "## 2. Merges (partial: stopped early)", "",
           "Stopped early to focus on frontier models. What was finished:"]
    p = R / "merge_controlled.json"
    if p.exists():
        D = json.load(open(p))
        out += ["", "Controlled merges of our own Pythia-160m fine-tunes made with mergekit (16-bit). Parents: O = base,",
                "A = r01_lin, B = r14_lin_8k (different documents), C = r03_const. *Record* = tokens updated in the merge /",
                "tokens updated in any parent; *untouched kept* = tokens no parent updated whose vectors stay bit-identical",
                "to the base; *recovered weights* = least-squares fit of the merge on the parents (given the parents).", "",
                "| merge | true setting | record | untouched kept | recovered weights | fit error |", "|---|---|---|---|---|---|"]
        for m in D["merges"]:
            if m.get("error"):
                out.append(f"| {m['name']} | {m['truth']['method']} | failed | | | |")
                continue
            wp = m["with_parents"]
            rv = wp["record_vs_union"]
            tr = m["truth"]["method"] + (": " + W(m["truth"]["w"]) if m["truth"].get("w") else "") + (
                f", density {m['truth']['density']}" if "density" in m["truth"] else "")
            out.append("| {} | {} | {:,} / {:,}{} | {:,} / {:,} | {} | {:.2%} |".format(
                m["name"], tr, rv["tp"], rv["tp"] + rv["fn"], f", {rv['fp']:,} false" if rv["fp"] else "",
                wp["untouched_by_all_identical"], wp["untouched_by_all"],
                W({k: v for k, v in wp["lstsq_coef"].items() if abs(v) > 5e-3}), wp["lstsq_rel_resid"]))
        out += ["", "Observed: linear, SLERP, task-arithmetic, TIES and DARE merges keep every token no parent updated",
                "bit-identical, so a merge's token record is the union of its parents' records; given the parents, linear /",
                "SLERP / task-arithmetic weights are recovered to about 0.01 (TIES / DARE / breadcrumbs fits do not equal",
                "the declared weights, because those methods drop and rescale numbers). model_stock changes every token vector by one",
                "smallest 16-bit step, which erases the record. No blind (parent-free) merge detector was finished."]
    p = R / "merge_real.json"
    if p.exists():
        D = json.load(open(p))
        done = [m for m in D["merges"] if m.get("with_parents")]
        if done:
            out += ["", f"Published merges checked before stopping ({len(done)}):", ""]
            for m in done:
                wp = m["with_parents"]
                rv = wp["record_vs_union"]
                out.append(f"- {m['merge']} (declared {m['config']['method']}): updated tokens {rv['tp']:,} / {rv['tp'] + rv['fn']:,} "
                           f"of the parents' union, untouched kept {wp['untouched_by_all_identical']:,} / {wp['untouched_by_all']:,}; "
                           f"fit {W({k: v for k, v in wp['lstsq_coef'].items() if abs(v) > 5e-3})}")
            out += ["", "Caveat found on the way: DeepMount00/Llama-3-8b-Ita differs from Llama-3-8B-Instruct only in tiny",
                    "numbers (0.27% of them, all below 1e-5), the pattern of a float16 round trip, not training. Such a copy",
                    "looks as if it updated 107,065 tokens; the ITA merges inherit that. Fits on parents that identical or",
                    "nearly identical are not meaningful (the anakin87 TIES rows above)."]
    return out


def main():
    lines = ["# Quantized releases and merges", "",
             "Generated by `scripts/quant_merge/report.py` from `quant.json`, `quant_*.jsonl`, `merge_controlled.json`, `merge_real.json`.",
             "Scripts: `scripts/quant_merge/quant.py` (+ `frontier_families.py`, `gguf_remote.py`), `merge_controlled.py`,",
             "`real_merges.py`, `mergesig.py`. Only embedding tensors are downloaded (HTTP range reads).", ""]
    lines += quant_section() + merge_section()
    (R / "QUANT_MERGE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
