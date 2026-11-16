#!/usr/bin/env python3
"""Published-data audits: assumptions, a cutoff x system-prompt sweep, and controls
(tech review M8).

For each published fine-tune (s1, s1.1, Sky-T1, Bespoke-Stratos 32B/7B, OpenThinker-7B):
  - the changed-token set (any number differs from the base, P = 1, as in compare.py);
  - the data's token set re-derived per document for a grid of cutoffs and system
    prompts (TRL whole-text for s1/s1.1; LLaMA-Factory qwen template, slot-wise encoding
    and infer_seqlen truncation for the others, as tokens_in_data.lf_ids); the last
    position of each sequence is excluded (no gradient);
  - the variants under which data set == changed set (the "matching range");
  - controls at the matching variant:
      cross-dataset   : the changed set scored against every other dataset's token set
      subsample power : expected tokens lost, and P(at least one lost), if a random 1% or
                        10% of the documents had been left out of training
                        (hypergeometric per token from its document frequency; tokens
                        treated as independent for P(>=1))
      per-document    : share of documents holding a token no other document has
                        (leaving out that one document alone would be visible)
      extra-document  : share of documents of another dataset that would add at least one
                        token absent from this dataset (an extra such document would be visible)
Writes results/published/controls.json and CONTROLS.md.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "published"))
OUT = ROOT / "results" / "published"
TOK = "Qwen/Qwen2.5-32B-Instruct"
DEFAULT_SYS = "You are a helpful assistant."   # LLaMA-Factory qwen template default

MODELS = [  # child, data, base
    ("simplescaling/s1-32B", "s1", "Qwen/Qwen2.5-32B-Instruct"),
    ("simplescaling/s1.1-32B", "s1.1", "Qwen/Qwen2.5-32B-Instruct"),
    ("NovaSky-AI/Sky-T1-32B-Preview", "skyt1", "Qwen/Qwen2.5-32B-Instruct"),
    ("bespokelabs/Bespoke-Stratos-32B", "stratos", "Qwen/Qwen2.5-32B-Instruct"),
    ("bespokelabs/Bespoke-Stratos-7B", "stratos", "Qwen/Qwen2.5-7B-Instruct"),
    ("open-thoughts/OpenThinker-7B", "ot114k", "Qwen/Qwen2.5-7B-Instruct"),
]
TRL_CUTS = [2048, 4096, 8192, 16384, 20000, 24576, 32768, None]
LF_CUTS = [4096, 8192, 12288, 16384, 20480, 24576, 32768, None]
OT_CUTS = [8192, 16384, 32768, None]
SAMPLE = 500

_tok = None
_rows = None


def _init(which):
    global _tok, _rows
    from transformers import AutoTokenizer
    from datasets import load_dataset
    _tok = AutoTokenizer.from_pretrained(TOK)
    if which in ("s1", "s1.1"):
        ds = load_dataset("simplescaling/s1K_tokenized" if which == "s1" else "simplescaling/s1K-1.1_tokenized", split="train")
        _rows = ds["text"]
    else:
        repo = {"skyt1": "NovaSky-AI/Sky-T1_data_17k", "stratos": "bespokelabs/Bespoke-Stratos-17k",
                "ot114k": "open-thoughts/OpenThoughts-114k"}[which]
        _rows = load_dataset(repo, split="train")


def variants_of(which):
    if which in ("s1", "s1.1"):
        return [("trl", c) for c in TRL_CUTS]
    cuts = OT_CUTS if which == "ot114k" else LF_CUTS
    return [(s, c) for s in ("datasys", "default") for c in cuts]


def lf_pairs(system, turns):
    import tokens_in_data as T
    sys_s = f"<|im_start|>system\n{system}<|im_end|>\n" if system is not None else ""
    msgs = [(t.get("from") or t.get("role"), t.get("value") or t.get("content")) for t in turns]
    pairs, i, first = [], 0, True
    while i + 1 < len(msgs):
        (_, u), (_, a) = msgs[i], msgs[i + 1]
        src = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
        s_ids = (_tok.encode(sys_s, add_special_tokens=False) if (first and sys_s) else []) + _tok.encode(src, add_special_tokens=False)
        t_ids = _tok.encode(f"{a}<|im_end|>\n", add_special_tokens=False)
        pairs.append((s_ids, t_ids)); first = False; i += 2
    return pairs


def lf_truncate(pairs, cutoff):
    import tokens_in_data as T
    ids, total = [], 0
    for s_ids, t_ids in pairs:
        if cutoff and total >= cutoff:
            break
        if cutoff:
            sl, tl = T.infer_seqlen(len(s_ids), len(t_ids), cutoff - total)
            s_ids, t_ids = s_ids[:sl], t_ids[:tl]
        ids += s_ids + t_ids
        total += len(s_ids) + len(t_ids)
    return ids


def _work(args):
    which, lo, hi, sample_ids = args
    import tokens_in_data as T
    var = variants_of(which)
    df = {v: Counter() for v in var}
    first = {v: {} for v in var}
    samp = {v: {} for v in var}
    for k in range(lo, hi):
        if which in ("s1", "s1.1"):
            full = _tok(_rows[k], add_special_tokens=True)["input_ids"]
            per = {}
            for v in var:
                c = v[1]
                ids = full[:c] if c else full
                per[v] = set(ids[:-1])
        else:
            r = _rows[k]
            turns = T.conv_of(r)
            per = {}
            pp = {"datasys": lf_pairs(r.get("system"), turns), "default": lf_pairs(DEFAULT_SYS, turns)}
            for v in var:
                ids = lf_truncate(pp[v[0]], v[1])
                per[v] = set(ids[:-1])
        for v, s in per.items():
            df[v].update(s)
            fv = first[v]
            for t in s:
                if t not in fv:
                    fv[t] = k
            if k in sample_ids:
                samp[v][k] = sorted(s)
    return df, first, samp


def doc_stats(which, n_procs=8):
    _init(which)
    n = len(_rows)
    rng = np.random.default_rng(7)
    sample_ids = set(rng.choice(n, size=min(SAMPLE, n), replace=False).tolist())
    step = max(1, n // (n_procs * 8))
    jobs = [(which, a, min(n, a + step), sample_ids) for a in range(0, n, step)]
    var = variants_of(which)
    df = {v: Counter() for v in var}
    first = {v: {} for v in var}
    samp = {v: {} for v in var}
    with Pool(n_procs, initializer=_init, initargs=(which,)) as P:
        for d, f, s in P.imap_unordered(_work, jobs):
            for v in var:
                df[v].update(d[v])
                for t, k in f[v].items():
                    if t not in first[v] or k < first[v][t]:
                        first[v][t] = k
                samp[v].update(s[v])
    return n, df, first, samp


def changed_mask(child, base):
    import compare as C
    cache = C.PUB / "masks"
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / (child.replace("/", "__") + ".npz")
    if p.exists():
        z = np.load(p)
        return z["changed"], int(z["n"]), int(z["real"])
    from transformers import AutoTokenizer
    W0 = C.get(base, C.EMB)
    W1 = C.get(child, C.PUB)
    n = min(W0.shape[0], W1.shape[0])
    frac = C.frac_changed(W0[:n], W1[:n], 1.0)
    real = len(AutoTokenizer.from_pretrained(base))
    np.savez_compressed(p, changed=frac > 0, n=n, real=real)
    return frac > 0, n, real


def lost_tokens(dfv, n, f):
    """Expected tokens lost and P(>=1 lost) when m = round(f n) random documents are left out."""
    m = int(round(f * n))
    lg = math.lgamma
    exp, logkeep = 0.0, 0.0
    for t, c in dfv.items():
        if c > m:
            continue
        # P(all c documents holding t are among the m left out) = C(n-c, m-c) / C(n, m)
        lp = (lg(n - c + 1) - lg(m - c + 1) - lg(n - m + 1)) - (lg(n + 1) - lg(m + 1) - lg(n - m + 1))
        p = math.exp(lp)
        exp += p
        logkeep += math.log1p(-min(p, 1 - 1e-15))
    return exp, 1 - math.exp(logkeep)


def main():
    only = set(sys.argv[1:])
    datasets = sorted({d for _, d, _ in MODELS if not only or d in only})
    stats = {}
    for d in datasets:
        print("tokenizing", d, flush=True)
        stats[d] = doc_stats(d)
    res = dict(assumptions={}, models={}, validation={})
    REF = {"s1": [(("trl", 32768), "trl_cut32768_nonfinal")], "s1.1": [(("trl", 20000), "trl_cut20000_nonfinal")],
           "skyt1": [(("datasys", 16384), "lf_datasys_16384_nonfinal"), (("default", 16384), "lf_default_16384_nonfinal")],
           "stratos": [(("datasys", 16384), "lf_datasys_16384_nonfinal"), (("default", 16384), "lf_default_16384_nonfinal")],
           "ot114k": [(("datasys", 16384), "lf_datasys_16384_nonfinal"), (("default", 16384), "lf_default_16384_nonfinal")]}
    for d in stats:
        ref = json.load(open(OUT / f"data_{d}.json"))["counts"]
        for v, key in REF.get(d, []):
            if key in ref:
                mine = set(stats[d][1][v]); theirs = {int(k) for k in ref[key]}
                res["validation"][f"{d} {v} vs {key}"] = dict(only_here=len(mine - theirs), only_ref=len(theirs - mine))
    print("validation", res["validation"], flush=True)
    match_var = {}
    for child, d, base in MODELS:
        if d not in stats:
            continue
        changed, n, real = changed_mask(child, base)
        nd, df, first, samp = stats[d]
        rows = {}
        for v in variants_of(d):
            present = np.zeros(n, bool)
            for t in df[v]:
                if t < n:
                    present[t] = True
            pnc = int((present & ~changed).sum())
            cnp = np.nonzero(changed & ~present)[0]
            cnp_real = int((cnp < real).sum())
            rows[f"{v[0]}|{v[1]}"] = dict(present=int(present.sum()), present_not_changed=pnc,
                                          changed_not_present=cnp_real, match=(pnc == 0 and cnp_real == 0))
        ok = [k for k, r in rows.items() if r["match"]]
        best = min(rows, key=lambda k: rows[k]["present_not_changed"] + rows[k]["changed_not_present"])
        match_var[child] = best
        res["models"][child] = dict(data=d, docs=nd, changed=int(changed.sum()), variants=rows,
                                    matching_variants=ok, best_variant=best)
    # controls at the best variant
    for child, d, base in MODELS:
        if child not in res["models"]:
            continue
        M = res["models"][child]
        changed, n, real = changed_mask(child, base)
        nd, df, first, samp = stats[d]
        v = tuple(M["best_variant"].split("|")); v = (v[0], None if v[1] == "None" else int(v[1]))
        dfv = df[v]
        uniq_docs = {first[v][t] for t, c in dfv.items() if c == 1}
        ctl = dict(per_document_share_with_unique_token=len(uniq_docs) / nd)
        for f in (0.01, 0.10):
            e, p1 = lost_tokens(dfv, nd, f)
            ctl[f"leave_out_{int(f * 100)}pct"] = dict(expected_tokens_lost=e, p_at_least_one_lost=p1)
        cross = {}
        present = np.zeros(n, bool)
        for t in dfv:
            if t < n:
                present[t] = True
        for d2 in stats:
            if d2 == d:
                continue
            nd2, df2, first2, samp2 = stats[d2]
            other = [m2 for m2 in res["models"].values() if m2["data"] == d2]
            v2 = tuple(other[0]["best_variant"].split("|")) if other else None
            v2 = (v2[0], None if v2[1] == "None" else int(v2[1])) if v2 else variants_of(d2)[0]
            p2 = np.zeros(n, bool)
            for t in df2[v2]:
                if t < n:
                    p2[t] = True
            cross[d2] = dict(present_not_changed=int((p2 & ~changed).sum()),
                             changed_not_present=int(((changed & ~p2).nonzero()[0] < real).sum()))
            # extra-document power: documents of d2 that add a token absent from d's set
            sets = samp2[v2]
            if sets:
                cross[d2]["extra_doc_detectable_share"] = float(np.mean([any((t < n and not present[t]) for t in s) for s in sets.values()]))
        ctl["cross_dataset"] = cross
        M["controls"] = ctl
    res["assumptions"] = {
        "s1/s1.1": "trl SFTTrainer on the `text` column, add_special_tokens=True, tokenizer Qwen2.5-32B-Instruct; truncation = block size; last position excluded",
        "LLaMA-Factory sets": "qwen template, slot-wise encoding, infer_seqlen truncation (re-implementation in tokens_in_data.lf_ids of LLaMA-Factory 0.9.2.dev0, the copy vendored in GAIR-NLP/LIMO); system prompt from the data ('datasys') or the template default 'You are a helpful assistant.' ('default'); last position excluded",
        "changed": "any number differs from the base in the release format, P = 1 (compare.py)",
        "independence": "P(at least one token lost) treats tokens as independent; the expected count is exact",
    }
    json.dump(res, open(OUT / "controls.json", "w"), indent=1, default=str)
    write_md(res)


def write_md(res):
    L = ["# Published-data audits: assumptions, sweep and controls", "",
         "Generated by `scripts/published/controls.py` (tech review M8). Counts are distinct vocabulary tokens.", "",
         "## Assumptions", ""]
    for k, v in res["assumptions"].items():
        L.append(f"- **{k}:** {v}")
    L += ["", "## Matching range", "",
          "Variant = system prompt | cutoff (`trl` = s1's whole-text tokenization; `None` = no truncation). A variant",
          "matches when every token in the data was updated and no other regular token was.", "",
          "| fine-tune | documents | matching variants | best variant | mismatches at other variants (in data not updated / updated not in data) |",
          "|---|---|---|---|---|"]
    for c, M in res["models"].items():
        others = "; ".join(f"{k}: {r['present_not_changed']}/{r['changed_not_present']}" for k, r in M["variants"].items() if not r["match"])
        L.append(f"| {c} | {M['docs']:,} | {', '.join(M['matching_variants']) or 'none'} | {M['best_variant']} | {others} |")
    L += ["", "## Controls at the best variant", "",
          "| fine-tune | docs with a token no other doc has | leave out 1%: tokens lost (P detect) | leave out 10%: tokens lost (P detect) | cross-dataset (in data not updated / updated not in data) | extra document from another set visible |",
          "|---|---|---|---|---|---|"]
    for c, M in res["models"].items():
        ct = M.get("controls")
        if not ct:
            continue
        cr = "; ".join(f"{d}: {x['present_not_changed']}/{x['changed_not_present']}" for d, x in ct["cross_dataset"].items())
        ex = "; ".join(f"{d}: {100 * x['extra_doc_detectable_share']:.0f}%" for d, x in ct["cross_dataset"].items() if "extra_doc_detectable_share" in x)
        l1, l10 = ct["leave_out_1pct"], ct["leave_out_10pct"]
        L.append(f"| {c} | {100 * ct['per_document_share_with_unique_token']:.1f}% | {l1['expected_tokens_lost']:.1f} ({l1['p_at_least_one_lost']:.2f}) | "
                 f"{l10['expected_tokens_lost']:.0f} ({l10['p_at_least_one_lost']:.2f}) | {cr} | {ex} |")
    L += ["", "Reading: set equality rules out a different dataset and, with the stated probability, a training set missing a",
          "random 1% or 10% of the documents. It cannot rule out that individual documents without a unique token were",
          "absent, or that extra documents whose tokens all occur elsewhere in the data were present."]
    (OUT / "CONTROLS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
