#!/usr/bin/env python3
"""results/tokens/pairs/*.json (+ negative_control.json) -> results/tokens/tokens.json + TOKENS.md.

Gates (editorial review):
  - a record counts only with >= MIN_UNMOVED bit-identical token vectors;
  - special-token lists only where the weight-decay fit reproduces >= 99% of the
    never-updated vectors (sweep P_exact_frac);
  - a model with many barely-changed vectors (> 10% of changed vectors have < 10% of
    their numbers changed) is 'low signal': absence is reported as
    'bit-identical or barely changed', never as 'absent from the data';
  - tokens whose base-model vector is still at its random starting size were never
    trained in pretraining either; they are counted apart from category percentages;
  - unreachable tokens are excluded everywhere.
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results" / "tokens"
sys.path.insert(0, str(ROOT / "scripts" / "tokens"))
from categories import categories  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "frontier"))
import comparability as _cmp  # noqa: E402
_RECS = _cmp.records()


def _class_of(key):
    r = _RECS.get(key)
    return _cmp.classify(r) if r else None

COLS = [("script:LATIN", "Latin"), ("script:CJK", "Han"), ("kana", "Kana"), ("script:HANGUL", "Hangul"),
        ("script:CYRILLIC", "Cyrillic"), ("script:ARABIC", "Arabic"), ("script:HEBREW", "Hebrew"),
        ("script:DEVANAGARI", "Devanagari"), ("script:THAI", "Thai"), ("script:GREEK", "Greek"),
        ("code", "code"), ("code:indent", "indent"), ("digits", "digits"), ("math", "math"), ("emoji", "emoji")]
NAMED = [("chat-template", "chat"), ("tool-call", "tool-call"), ("reasoning-tag", "thinking"), ("multimodal", "image/media")]
MIN_UNMOVED = 10
MIN_PFIT = 0.99
MIN_IDENT_NO_DECAY = 100
LOW_SIGNAL = 0.10
MIN_N = 30
STATES = ("moved", "rewritten", "near", "unmoved")


def kind_of(t):
    cs = categories(t["content"], True, t["special"])
    if "reserved" in cs:
        return "reserved"
    for k, _ in NAMED:
        if k in cs:
            return k
    return "other-special"


def tstate(t, low):
    if t["state"] == "rewritten":
        return "overwritten by hand (" + "+".join(f for f in t["flags"] if f in ("zeroed", "rescaled", "copied")) + ")"
    if t["state"] == "moved":
        if low and t.get("frac_changed", 1) < 0.10:
            return "barely changed"      # low-signal model: a 1-10% change may be rounding drift, not data
        # base vector ~zero (e.g. Qwen2.5 base special tokens, |v| ~ 1e-37): never trained before
        base_zero = t.get("norm_ratio", 1) > 1e3 or "parent_zero" in t.get("flags", [])
        return "first trained here" if (t["at_init_in_parent"] or base_zero) else "in the data"
    if t["state"] == "near":
        return "barely changed"
    if t["at_init_in_parent"]:
        return "never trained (base still at starting size)"
    return "bit-identical or barely changed" if low else "not in the data"


def main():
    rows = []
    for f in sorted(glob.glob(str(R / "pairs" / "*.json"))):
        d = json.load(open(f))
        base = dict(key=d["key"], child=d["child"], parent=d["parent"])
        if d.get("status") != "ok":
            rows.append(dict(base, status=d.get("status")))
            continue
        if d["recomputed"]["unmoved"] < MIN_UNMOVED:
            rows.append(dict(base, status=f"only {d['recomputed']['unmoved']} token vector(s) bit-identical to the base model"))
            continue
        if not d["P"] or d["P"] < 0.5:
            rows.append(dict(base, status=f"decay factor fitted as P={d['P']:.3g}: the bit-identical vectors are zeroed "
                                          "in the fine-tune, not untouched; no token record"))
            continue
        h = d["frac_changed_hist"]
        changed = sum(v for k, v in h.items() if k != "0")
        barely = h["(0,1%]"] + h["(1%,10%]"]
        low = barely > LOW_SIGNAL * max(1, changed)
        pfit = d.get("P_exact_frac")
        # same rule as scripts/frontier/comparability.py: with no decay (P = 1) and >= 100 vectors
        # bit-identical to the base, the 512-vector fit score is capped by how few vectors are
        # untouched and does not indicate a different base (results/frontier_ds/RECONCILE.md)
        # the reviewed comparability classifier (scripts/frontier/comparability.py): special-token lists
        # are trusted only for pairs it classes "token record" (cleaned bit-identical count above its
        # calibrated threshold, and an exact decay fit or no shrink)
        cls = _class_of(d["key"])
        trust = cls == "token record"
        cats = dict(d["categories"])
        kana = {k: cats.get("script:HIRAGANA", {}).get(k, 0) + cats.get("script:KATAKANA", {}).get(k, 0)
                for k in ("moved", "rewritten", "near", "unmoved", "unmoved_at_init", "moved_at_init")}
        cats["kana"] = kana
        a = cats["all"]
        rec = dict(base, status="ok", dtype=d["dtype"], P=d["P"], P_fit_frac=pfit, special_lists_trusted=trust, comparability_class=cls,
                   low_signal=low, barely_changed=barely, changed=changed, frac_changed_hist=h,
                   hand_edits=d["hand_edits"], init_std_source=d["init_std_source"],
                   unreachable=d["unreachable"].get("all"), all=a, frac={}, n={})

        def fr(c):
            n = c.get("moved", 0) + c.get("near", 0) + c.get("unmoved", 0) - c.get("unmoved_at_init", 0) - c.get("moved_at_init", 0)
            return ((c.get("moved", 0) - c.get("moved_at_init", 0)) / n if n > 0 else None), n
        rec["all_frac"], _ = fr(a)
        for k, _ in COLS:
            rec["frac"][k], rec["n"][k] = fr(cats.get(k, {}))
        sp = []
        for t in d["added_tokens"]:
            sp.append(dict(content=t["content"], kind=kind_of(t), state=tstate(t, low), frac_changed=t["frac_changed"]))
        rec["special"] = sp
        rec["first_trained_vocab"] = a.get("moved_at_init", 0)
        rec["never_trained_vocab"] = a.get("unmoved_at_init", 0)
        rec["freq"] = d["freq_rank_buckets"]
        rows.append(rec)
    json.dump(rows, open(R / "tokens.json", "w"), indent=1, ensure_ascii=False)
    write_md(rows)


def pct(x, n):
    return "·" if x is None or n < MIN_N else f"{100 * x:.0f}"


def lst(xs, k=8):
    return ", ".join("`" + c + "`" for c in xs[:k]) + (f" (+{len(xs) - k} more)" if len(xs) > k else "")


def by_state(r, kinds):
    out = defaultdict(list)
    for t in r["special"]:
        if t["kind"] in kinds:
            out[t["state"]].append(t["content"])
    return out


def notable(r):
    out = []
    absent = "bit-identical or barely changed" if r["low_signal"] else "appeared"
    for k, lab in COLS:
        fr, n = r["frac"][k], r["n"][k]
        if fr is not None and n >= 100 and fr < 0.05 and r["all_frac"] and r["all_frac"] > 0.2:
            out.append(f"{lab}: only {100 * fr:.1f}% of {n:,} tokens updated" +
                       (" (the rest bit-identical or barely changed)" if r["low_signal"] else ""))
    if r["special_lists_trusted"]:
        for k, lab in NAMED:
            s = by_state(r, {k})
            if s.get("in the data"):
                out.append(f"{lab} tokens in the data: {lst(s['in the data'])}")
            if s.get("first trained here"):
                out.append(f"{lab} tokens first trained in this fine-tune (still at starting size in the base): "
                           f"{lst(s['first trained here'])}")
            for st in s:
                if st.startswith("overwritten"):
                    out.append(f"{lab} tokens {st}, not trained: {lst(s[st])}")
            if k in ("tool-call", "reasoning-tag") and s.get("not in the data"):
                out.append(f"{lab} tokens not in the data: {lst(s['not in the data'])}")
        rs = by_state(r, {"reserved"})
        tr = rs.get("in the data", []) + rs.get("first trained here", [])
        if tr:
            out.append(f"reserved placeholders updated: {len(tr)} ({lst(tr, 4)})")
    if r["first_trained_vocab"] >= 20:
        out.append(f"{r['first_trained_vocab']:,} ordinary tokens untouched in pretraining were first trained here")
    return out


def fam_key(r):
    org = r["child"].split("/")[0]
    name = r["child"].split("/")[1].lower()
    name = re.sub(r"-?a\d+(\.\d+)?b", "", name)
    name = re.sub(r"\d+(\.\d+)?[bm](?![a-z])", "#", name)
    name = re.sub(r"-(bf16|hf)$", "", name)
    return org, name


def write_md(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    ntrust = sum(r["special_lists_trusted"] for r in ok)
    nlow = sum(r["low_signal"] for r in ok)
    L = ["# What each fine-tune's data contained, detected from its token vectors", "",
         "Generated by `scripts/tokens/report.py` from `results/tokens/pairs/*.json` (`scripts/tokens/mine.py`).", "",
         "Every token in a model's vocabulary has its own vector of numbers (its embedding). During fine-tuning a token's",
         "input vector is updated only when that token occurs in a training input. Comparing a fine-tune with its base",
         "model therefore shows, token by token, whether the token appeared anywhere in the fine-tuning data (every stage",
         "between the two releases: supervised data, preference data, RL rollouts). Counts are counts of **distinct",
         "vocabulary tokens**, never amounts of text.", "",
         f"Covered: **{len(ok)}** base model -> fine-tune pairs (written parent -> child). Each pair has separate input",
         "and output matrices (tied embeddings erase this record), has some token vectors updated, and has at least",
         f"{MIN_UNMOVED} vectors bit-identical to the base model. The earlier sweep counted {len(rows)} such pairs with a",
         f"threshold of one bit-identical vector; the {len(bad)} below this bar are listed at the end.", "",
         "## How a token is classed", "",
         "- **in the data:** the vector differs from the base model's in more than max(2, 1% of its numbers), after",
         "  removing weight decay. Weight decay shrinks every vector by the same factor whether or not its token was",
         "  used. That factor is fitted on the unused tokens and replayed exactly, so shrinkage alone never counts as use.",
         "- **not in the data:** the vector equals the base model's after that shrinkage, in every number.",
         "- **barely changed:** only a few numbers differ, from rounding in the 16-bit release format or a very small",
         "  late update. Never counted as data.",
         "- **never trained:** the base model's vector is still at its random starting size, so the token was not",
         "  trained in pretraining either. The size comes from the base model's starting values, regenerated exactly by",
         "  checkpoint-forensics. Where those aren't available, it comes from byte tokens that cannot occur in valid UTF-8",
         "  text. Such tokens are counted apart from the percentages below. When they are updated in the fine-tune, they",
         "  are reported as *first trained here*.", "",
         "## Checks applied before any claim", "",
         f"- **Is the absence detectable?** In a 16-bit release a small update can round to zero, so 'bit-identical'",
         f"  can fall below the detection floor. Every model has a histogram of the fraction of numbers changed per",
         f"  token vector (`frac_changed_hist` in `tokens.json`). When more than {LOW_SIGNAL:.0%} of the changed vectors",
         f"  have fewer than 10% of their numbers changed, the model is marked **low signal** (**{nlow}** models,",
         "  e.g. DeepSeek-R1-Zero). There, an unchanged token is reported as *bit-identical or barely changed*, not as",
         "  absent from the data.",
         "- **Special-token lists only for a token record:** only where the reviewed comparability classifier",
         "  (`scripts/frontier/comparability.py`) classes the base/fine-tune comparison as *token record*: enough",
         "  bit-identical vectors left after removing unreachable, reserved, at-init, zero and duplicate rows, and",
         f"  an exact decay fit or no shrink (**{ntrust}** of {len(ok)}).",
         "- In a low-signal model a special token counts as *in the data* only if at least 10% of its numbers changed;",
         "  smaller changes are reported as *barely changed*.",
         "- **Named special tokens vs reserved placeholders.** Chat markers, tool-call, thinking and image/media tokens",
         "  are listed by name. Reserved / unused placeholders are counted separately, and the starting-size test shows",
         "  whether the base model ever trained them.",
         "- **Unreachable tokens are excluded.** Some vocabulary entries are never produced by the tokenizer: encoding",
         "  the entry's own text yields other tokens (e.g. 1,578 of Qwen2.5's 2,549 Thai entries), or the entry is a",
         "  single byte that never occurs in valid UTF-8 text. Such entries can never appear in data. Entries reachable",
         "  alone but usually swallowed by a longer token in running text (` 바랍` inside ` 바랍니다`) still count;",
         "  their absence is real, but reflects tokenization as much as content.",
         "- **Hand-edited vectors are flagged, not counted.** A vector set to zero, rescaled (child = c x base with c",
         "  different from the weight-decay factor), or copied from another token is *overwritten by hand*.",
         "- **Across sizes:** where a vendor released the same fine-tune at several sizes, the special-token findings",
         "  are compared below.",
         "- **Negative control:** the same pipeline on models that are not base and fine-tune (end of page).",
         "- 'Han' covers Chinese characters and Japanese kanji alike; 'Kana' is Hiragana + Katakana. A percentage is",
         f"  shown only for categories with at least {MIN_N} tokens (`·` otherwise).",
         "", "## Per model: % of each category's tokens that appeared in the fine-tuning data", "",
         "`low` marks low-signal models (absence may be below the detection floor). Special-token columns count named",
         "tokens *in the data or first trained here / all*, only where the decay fit is exact (`-` otherwise).", "",
         "| child | parent | signal | all | " + " | ".join(l for _, l in COLS) + " | chat | tool-call | thinking | image/media |",
         "|" + "---|" * (4 + len(COLS) + 4)]
    for r in sorted(ok, key=lambda r: r["child"].lower()):
        sp = []
        for k, _ in NAMED:
            s = by_state(r, {k})
            n = sum(len(v) for v in s.values())
            yes = len(s.get("in the data", [])) + len(s.get("first trained here", []))
            sp.append("-" if not r["special_lists_trusted"] else (f"{yes}/{n}" if n else "·"))
        L.append(f"| {r['child']} | {r['parent']} | {'low' if r['low_signal'] else 'ok'} | "
                 f"{pct(r['all_frac'], 10 ** 6)} | " + " | ".join(pct(r["frac"][k], r["n"][k]) for k, _ in COLS) +
                 " | " + " | ".join(sp) + " |")
    L += ["", "## Notable, per vendor", "",
          "Extracted automatically: writing systems with at least 100 tokens of which under 5% were updated (only where",
          "over 20% of the vocabulary was), named special tokens in or not in the data (only where the decay fit is",
          "exact), tokens first trained in the fine-tune, and reserved placeholders that were updated.", ""]
    by = defaultdict(list)
    for r in ok:
        by[r["child"].split("/")[0]].append(r)
    for org in sorted(by, key=str.lower):
        items = []
        for r in sorted(by[org], key=lambda r: r["child"]):
            nb = notable(r)
            if nb:
                items.append(f"- **{r['child'].split('/')[1]}** (from {r['parent']}; "
                             f"{100 * (r['all_frac'] or 0):.0f}% of the vocabulary updated"
                             + ("; low signal" if r["low_signal"] else "") + "): " + "; ".join(nb))
        if items:
            L += [f"### {org}", ""] + items + [""]
    # across sizes
    groups = defaultdict(list)
    for r in ok:
        groups[fam_key(r)].append(r)
    L += ["## Same fine-tune at several sizes", "",
          "Named special tokens in the data (or first trained here), compared across sizes of one release. Only models",
          "whose decay fit is exact are compared. A difference between two low-signal models (gpt-oss-safeguard) can come",
          "from the 10% threshold for barely-changed tokens rather than from different data.", "",
          "| release | sizes | named special tokens in the data | same at every size? |", "|---|---|---|---|"]
    for (org, name), rs in sorted(groups.items()):
        seen = {}
        for r in rs:
            if r["special_lists_trusted"]:
                seen.setdefault(r["child"], r)
        rs = list(seen.values())
        if len(rs) < 2:
            continue
        sets = []
        for r in rs:
            s = by_state(r, {k for k, _ in NAMED})
            sets.append(tuple(sorted(s.get("in the data", []) + s.get("first trained here", []))))
        same = len(set(sets)) == 1
        sizes = ", ".join(r["child"].split("/")[1] for r in rs)
        desc = lst(list(sets[0]), 10) if same else "; ".join(
            f"{r['child'].split('/')[1]}: {lst(list(st), 6) or 'none'}" for r, st in zip(rs, sets))
        L.append(f"| {org}/{name.replace('#', 'N')} | {sizes} | {desc or 'none'} | {'yes' if same else '**no**'} |")
    L += ["", "## How broad was the data", "",
          "Tokenizer vocabularies are roughly ordered by how common each token is. The share of the 1,000 most common",
          "tokens versus the rarest ones (rank 50,000 and beyond) that were updated separates narrow fine-tunes (common",
          "tokens only) from very large or multilingual ones (most of the rare tail too).", "",
          "| child | top 1k | 1k-10k | 10k-50k | 50k+ |", "|---|---|---|---|---|"]
    for r in sorted(ok, key=lambda r: r["all_frac"] or 0):
        cells = []
        for b in ("0-1000", "1000-10000", "10000-50000", "50000-end"):
            c = r["freq"].get(b, {})
            n = sum(c.get(s, 0) for s in STATES) - c.get("unmoved_at_init", 0)
            cells.append(f"{100 * c.get('moved', 0) / n:.0f}%" if n > 0 else "·")
        L.append(f"| {r['child']} | " + " | ".join(cells) + " |")
    nc = R / "negative_control.json"
    if nc.exists():
        L += ["", "## Negative control: models that are not base and fine-tune", "",
              "The same pipeline on unrelated models that share a tokenizer and shape. A sound method finds no vectors",
              "bit-identical between them. Two runs that started from the same random values may share vectors, but only",
              "for tokens neither run ever trained: they must all be unreachable or at starting size.", "",
              "| model A | model B | tokens | bit-identical | of which unreachable | of which at starting size | left over | note |",
              "|---|---|---|---|---|---|---|---|"]
        for c in json.load(open(nc)):
            L.append(f"| {c['parent']} | {c['child']} | {c['rows']:,} | {c['unmoved']:,} | {c.get('unmoved_unreachable', '-')} | "
                     f"{c.get('unmoved_at_init', '-')} | {c.get('unmoved_left', '-')} | {c.get('note', '')} |")
    if bad:
        L += ["", "## Not covered", ""] + [f"- {r['child']} (from {r['parent']}): {r['status']}" for r in bad]
    (R / "TOKENS.md").write_text("\n".join(L) + "\n")
    print(len(ok), "ok;", len(bad), "not covered; trusted", ntrust, "low", nlow)


if __name__ == "__main__":
    main()
