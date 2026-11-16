#!/usr/bin/env python3
"""results/large/{sweep_large.jsonl, tokens/*.json, tulu_audit.json} -> results/large/LARGE.md + large.json.

Wording follows STYLE.md: a token is "bit-identical" when every number of its vector equals the
base (after removing any uniform shrink); "barely changed" when at most 1% of its numbers
differ; "updated" otherwise. In 16-bit releases a small update can round to zero, so
"bit-identical" is an observation, not proof the token was absent from the data.
"""
import glob, json, math, pathlib, sys
from collections import OrderedDict
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "frontier"))
from comparability import classify  # noqa: E402

L = ROOT / "results" / "large"
REC = OrderedDict()
for l in open(L / "sweep_large.jsonl"):
    r = json.loads(l)
    if r.get("status") == "ok":
        REC[r["key"]] = r
ORDER = [p["key"] for p in json.load(open(L / "pairs_large.json"))]
TOK = {}
for f in glob.glob(str(L / "tokens" / "*.json")):
    d = json.load(open(f))
    TOK[d["key"]] = d
TULU = json.load(open(L / "tulu_audit.json")) if (L / "tulu_audit.json").exists() else {}
SKIP = json.load(open(L / "skipped_large.json"))
_done = {(k.split("@ -> ")[0], k.split("@ -> ")[1]) for k in [p["key"] for p in json.load(open(L / "pairs_large.json"))]}
SKIP = [x for x in SKIP if (x["parent"], x["child"]) not in _done]
NICK = {"mlx-community/perplexity-ai-r1-1776-bf16": "perplexity-ai/r1-1776 (bf16 mirror)",
        "NousResearch/Meta-Llama-3.1-405B-FP8": "Llama-3.1-405B (bf16 embedding of the FP8 release)",
        "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8": "Llama-3.1-405B-Instruct (bf16 embedding of the FP8 release)",
        "unsloth/Meta-Llama-3.1-70B": "Llama-3.1-70B (unsloth mirror)"}
nm = lambda r: NICK.get(r, r)
SCRIPTS = ["script:LATIN", "script:CJK", "script:CYRILLIC", "script:ARABIC", "script:HANGUL", "script:HEBREW",
           "script:THAI", "script:DEVANAGARI", "script:HIRAGANA", "code", "emoji", "math"]


def reserved(c):
    c = c.lower()
    return "reserved" in c or "place▁holder" in c or "placeholder" in c or "pad" in c


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def summary(k, r):
    e = r["embed"]
    t = TOK.get(k)
    h = (t or {}).get("frac_changed_hist")
    rows = e["rows"]
    s = dict(key=k, parent=r["parent"], child=r["child"], relation=r.get("relation"), cls=classify(r),
             rows=rows, bit_identical=e["unmoved"], barely=e["near"], updated=e["moved"], P=e["P"],
             fit=e.get("P_exact_frac"), shrink_pct=round(100 * (1 - e["P"]), 4), dtype=r.get("dtype_child"))
    if h:
        ch = rows - h["0"]
        small = h["(0,1%]"] + h["(1%,10%]"]
        s["share_changed_under_10pct"] = round(small / ch, 4) if ch else None
        s["low_signal"] = bool(ch and small / ch > 0.10)
        s["hand_edits"] = t["hand_edits"]
    else:
        ch = r.get("change_hist", {}).get("counts")
        if ch:
            tot = sum(ch[1:])
            s["share_changed_under_10pct"] = round((ch[1] + ch[2]) / tot, 4) if tot else None
            s["low_signal"] = bool(tot and (ch[1] + ch[2]) / tot > 0.10)
    sp = [x for x in r.get("special", []) if x.get("special")]
    named = [x for x in sp if not reserved(x["content"])]
    s["named_special"] = dict(n=len(named), updated=sum(x["state"] == "moved" for x in named),
                              barely=sum(x["state"] == "near" for x in named),
                              bit_identical=[x["content"] for x in named if x["state"] == "unmoved"])
    s["reserved_special"] = dict(n=len(sp) - len(named), updated=sum(x["state"] == "moved" for x in sp if x not in named))
    hd = r.get("head") or {}
    if "rows" in hd:
        s["head"] = dict(bit_identical=hd["unmoved"], barely=hd["near"], updated=hd["moved"], P=hd["P"])
    if t and t.get("status") == "ok":
        cats = {}
        for c in SCRIPTS:
            v = t["categories"].get(c)
            if v:
                n = sum(v[x] for x in ("moved", "rewritten", "near", "unmoved"))
                if n >= 50:
                    cats[c.replace("script:", "")] = dict(n=n, bit_identical=v["unmoved"], barely=v["near"],
                                                          share_bit_identical=round(v["unmoved"] / n, 4))
        s["categories"] = cats
        s["special_detail"] = [dict(content=x["content"], state=x["state"], at_init_in_parent=x["at_init_in_parent"],
                                    frac_changed=x["frac_changed"], flags=x["flags"]) for x in t["added_tokens"]
                               if not reserved(x["content"])][:80]
        s["parent_at_init"] = t.get("parent_at_init")
    return s


def findings(S):
    by = {(s["parent"], s["child"]): s for s in S}
    g = lambda a, b: by.get((a, b), {})
    DI = json.load(open(L / "decay_intervals.json")) if (L / "decay_intervals.json").exists() else []
    KZ = json.load(open(L / "kimidev_zeroed.json")) if (L / "kimidev_zeroed.json").exists() else {}
    R1, R1776, MAI = "deepseek-ai/DeepSeek-R1", "mlx-community/perplexity-ai-r1-1776-bf16", "microsoft/MAI-DS-R1"
    L405, L405I = "NousResearch/Meta-Llama-3.1-405B-FP8", "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8"
    o = ["", "## Lineage: which published model is the closer parent", "",
         "More bit-identical tokens against a candidate parent means fewer of its token vectors were touched since that parent.", ""]
    for child, cands in (("Qwen/QwQ-32B", ["Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct", "Qwen/QwQ-32B-Preview"]),
                         ("a-m-team/AM-Thinking-v1", ["Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct"]),
                         ("moonshotai/Kimi-Dev-72B", ["Qwen/Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct"]),
                         ("nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", [L405, L405I]),
                         ("NousResearch/Hermes-4-405B", [L405, "NousResearch/Hermes-3-Llama-3.1-405B"]),
                         ("deepseek-ai/DeepSeek-R1-0528", [R1, "deepseek-ai/DeepSeek-V3-Base"])):
        cells = [f"{nm(c)}: {g(c, child).get('bit_identical', 0):,} bit-identical" for c in cands if g(c, child)]
        o.append(f"- {nm(child)}: " + "; ".join(cells))
    o += ["", "Kimi-Dev-72B has no bit-identical vectors against either Qwen2.5-72B release; its 2,171 'bit-identical' vectors are",
          "vectors set to exactly zero (below). Its median cosine to the base is 0.99900 vs 0.99883 to the Instruct release",
          "(results/large/kimidev_check.json).", ""]
    if DI:
        o += ["## Total weight decay on the token vectors (exact ranges, results/large/decay_intervals.json)", ""]
        for x in DI:
            a, b = x["key"].split("@ -> ")
            o.append(f"- {nm(a)} -> {nm(b)}: vectors shrank by {x['shrink_pct_lo']:.3f}-{x['shrink_pct_hi']:.3f}% "
                     f"(sum lr x wd in [{x['sum_lr_wd_lo']:.5f}, {x['sum_lr_wd_hi']:.5f}], from {x['rows']:,} never-updated vectors)")
        o += ["", "The Llama-3.1-405B -> Instruct range is as wide as one 16-bit step: those vectors moved by exactly one smallest step,",
              "so it is a bound, not a value (the same effect as the corrected Llama-3.1/3.3 artefact in results/roster/NOTES.md).",
              "Every other comparison here shows no uniform shrink (P = 1): no detectable weight decay on the token vectors.", ""]
    if KZ:
        o += ["## Zeroed vectors in Kimi-Dev-72B (results/large/kimidev_zeroed.json)", "",
              f"Kimi-Dev-72B sets {KZ['kimi_dev_zero']:,} token vectors to exactly zero: {KZ['padding_slots']} unused vocabulary slots past the",
              f"tokenizer and {KZ['real_tokens']:,} tokens, of which only {KZ['real_reachable']} can be produced by the tokenizer (the rest are",
              "invalid UTF-8 bytes and byte sequences the merges never emit). KAT-Dev-72B-Exp (Kwaipilot), another SWE model on",
              f"Qwen2.5-72B, zeroes exactly the same {KZ['overlap']:,} vectors. These are hand edits, not training; the two releases",
              "share the step, consistent with a common tool that zeroes never-trained tokens.", ""]
    rr, mm = g(R1, R1776), g(R1, MAI)
    t = TULU
    o += ["## Findings and article-ready sentences", "",
          "1. **Two third-party post-trainings of DeepSeek-R1 left different footprints.** (holds, with wording) "
          f"r1-1776 leaves {rr.get('bit_identical', 0):,} of 129,280 token vectors bit-identical to R1 and {rr.get('barely', 0):,} barely changed; "
          f"MAI-DS-R1 leaves {mm.get('bit_identical', 0):,} bit-identical and updates {mm.get('updated', 0):,}. Both are low-signal releases "
          "(most changed vectors have under 10% of their numbers changed), so token lists read as \"bit-identical or barely changed\", "
          "not \"absent from the data\". Article: *\"Perplexity's r1-1776 changed DeepSeek-R1's token vectors only lightly: "
          f"{rr.get('bit_identical', 0):,} are bit-identical and most of the rest barely changed. Microsoft's MAI-DS-R1 updated "
          f"{mm.get('updated', 0):,} of them.\"* Category shares are in the table above; no content interpretation is made.",
          "2. **Hermes 4 leaves nearly the same tokens untouched at 70B and 405B; most of them Hermes 3 left untouched too.** (holds with wording) "
          "After removing unreachable, reserved, at-init, zero and duplicate rows, the 405B release keeps 2,461 vectors bit-identical to the "
          "Llama-3.1 base and the 70B release 2,482; all 2,461 are in the 70B set (Jaccard 0.99, results/frontier/sets.json). As a baseline, "
          "97% of those 2,461 are also bit-identical in Hermes-3-Llama-3.1-405B, 48% in Tulu-3-405B-SFT and 9-18% in Meta's own 8B/70B Instruct "
          "tunes; only 84 are left untouched by every one of them. Article: *\"Hermes 4 70B and 405B leave almost the same 2,500 tokens "
          "untouched, consistent with the same post-training data at both sizes; 97% of them Hermes 3 left untouched too.\"* "
          "(Before the review: 'the same 2,739 tokens, identical sets'; 2,739 was the raw count and the sets were not compared.)",
          "3. **Tulu 3 405B's SFT stage matches its published SFT mixture, with 245 exceptions; the DPO stage does not match its mixture.** (holds with wording) "
          + (f"All {t['sft']['cut']['present']:,} tokens in the published SFT mixture (as rendered and truncated for training) except "
             f"{t['sft']['cut']['present_not_changed']} were updated, and no regular token outside the data was updated. The {t['sft']['cut']['present_not_changed']} "
             "exceptions occur 1-2 times in most cases (39 occur more than 5 times). 'Nothing outside the data' is tested on the 3,684 "
             "vocabulary tokens the mixture lacks. Scored against the DPO mixture instead (wrong data), 1,598 in-data tokens are "
             "bit-identical and 1,008 changed tokens are outside it. The DPO stage against its own mixture: 2,188 in-data tokens "
             "bit-identical and 399 changed outside it, so that stage is not a match (results/large/tulu_controls.json, "
             "results/fixc/FIXC.md section 5). " if 'sft' in t and 'cut' in t['sft'] else "")
          + "Article: *\"Tulu 3 405B's SFT stage updated no token outside its published data, and all but 245 of the tokens in it.\"*",
          "4. **The end-of-sequence token is often bit-identical after DPO or RL.** (holds, mechanism) `<|end_of_text|>` appears in "
          "every Tulu 3 DPO response yet is bit-identical after the 405B DPO stage; DeepSeek's end-of-sentence token is bit-identical "
          "in MAI-DS-R1, AceReason-Nemotron-7B and both Skywork-OR1 releases. In a causal model the input vector of a sequence's final token "
          "gets no gradient unless another sequence follows it in the same row (packing), so this is consistent with unpacked training and needs no other explanation.",
          "5. **Tulu 3 405B's final RL stage touched few tokens.** (holds) After DPO, the RLVR stage leaves "
          f"{g('allenai/Llama-3.1-Tulu-3-405B-DPO', 'allenai/Llama-3.1-Tulu-3-405B').get('bit_identical', 0):,} of 128,264 vectors bit-identical, "
          "including every non-Latin-script token in the table; most changed vectors changed by under 10% of their numbers (low signal), "
          "so read this as a small, narrow update, not as a list of what the RL data lacked.",
          "6. **Lineage, re-derived with a margin rule (results/frontier/lineage.md).** Counts are bit-identical vectors after removing "
          "unreachable, reserved, at-init, zero and duplicate rows; a candidate is named only if it leads every other by 1.5x and 200 vectors. "
          "AM-Thinking-v1: Qwen2.5-32B base (9,337 vs 4,232 for Instruct). Nemotron-Ultra-253B: Llama-3.1-405B-Instruct (420 vs 47 for the base; "
          "its vectors are Instruct's shrunk by 0.765-0.769%). Not decided by the weights: QwQ-32B shares about equally with the Qwen2.5-32B base "
          "(2,579), QwQ-32B-Preview (2,566) and Instruct (1,947); Hermes-4-405B with the Llama-3.1-405B base (2,461) and Hermes 3 (2,400). "
          "(Before the review this item read: QwQ-32B closer to the base than to Instruct; Hermes-4-405B closer to the base than to Hermes 3.)",
          "7. **Two SWE releases on Qwen2.5-72B zero the same 2,171 vectors** (Kimi-Dev-72B, KAT-Dev-72B-Exp): a shared hand edit of "
          "never-trainable tokens. Report as an observation; no claim about how.",
          "8. **Not readable:** Skywork-OR1-7B (7 bit-identical; essentially every vector changed), DeepSeek-V3 -> V3.1 (429, poor fit), "
          "Kimi-Dev-72B (zeroed vectors only).",
          "", "Sources and caveats:",
          "- perplexity-ai/r1-1776 is no longer on Hugging Face; the bf16 MLX mirror is used. 20,876 of its vectors are bit-identical to R1,",
          "  which a lossy conversion could not produce, so its embedding is an exact copy of the release.",
          "- Llama-3.1-405B (base and Instruct) are gated; the bf16 token vectors inside public FP8 releases are used (NousResearch, RedHatAI).",
          "  Both are bit-identical in all 2,101,346,304 values to copies whose files match Meta's official releases by sha256 (results/gated/GATED.md).",
          "  Tulu and Hermes keep thousands of vectors bit-identical to them, so those embeddings are the original 16-bit ones.",
          "- AM-Thinking-v1's index file points at the wrong shards; its tensors were located by scanning shard headers (fix_am.py).",
          "- Output heads are compared up to 2.6 GB; the 405B heads (4.2 GB) were skipped to stay within the download budget.",
          "- All releases are 16-bit: a token updated by less than half a 16-bit step reads as bit-identical."]
    return o


def main():
    S = [summary(k, REC[k]) for k in ORDER if k in REC]
    missing = [k for k in ORDER if k not in REC]
    json.dump(dict(pairs=S, missing=missing, skipped=SKIP, tulu=TULU), open(L / "large.json", "w"), indent=1, ensure_ascii=False)
    o = ["# Large fine-tunes of frontier bases: token-level footprint", "",
         "Generated by `scripts/large/report.py` from `results/large/sweep_large.jsonl` (scripts/large/run.py, the roster",
         "sweep with per-vector change histograms), `results/large/tokens/` (scripts/tokens/mine.py categories) and",
         "`results/large/tulu_audit.json`. Embeddings (input token vectors) are compared; output heads up to 2.6 GB too.", "",
         "Terms: every token in the vocabulary has its own vector. **Bit-identical**: every number equals the base after removing",
         "any uniform shrink. **Barely changed**: at most 1% of the numbers differ. **Updated**: more than 1% differ. All releases",
         "here are 16-bit, so a small update can round to zero: bit-identical is what the weights show, not proof that a token was",
         "absent from the training data. **Low signal**: more than 10% of the changed vectors have under 10% of their numbers",
         "changed (the R1-Zero pattern); read token lists from those releases as \"bit-identical or barely changed\".", "",
         "Classes (scripts/frontier/comparability.py): *token record* = some bit-identical tokens and the shrink fit reproduces",
         ">=99% of the closest vectors (and >= T_LEFT trainable ones, results/frontier/calibration.json); *untrainable only* = the "
         "bit-identical vectors are all tokens no run could train; *record, poor fit* = some bit-identical but fit <99%; "
         "*changed throughout* = none",
         "bit-identical, same base; *not comparable* = not a shrunk copy of that base.", "",
         "| base -> release | relation | class | bit-identical | barely | updated | changed vectors with <10% of numbers changed | shrink | named special tokens updated | output head bit-identical / updated |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        ns = s["named_special"]
        hd = s.get("head")
        sh = s.get("share_changed_under_10pct")
        shs = "-" if sh is None else "%.1f%%" % (100 * sh)
        if s.get("low_signal"):
            shs += " (low signal)"
        hds = "-" if not hd else "{:,} / {:,}".format(hd["bit_identical"], hd["updated"])
        o.append("| {} -> {} | {} | {} | {:,} | {:,} | {:,} | {} | {:g}% | {}/{} | {} |".format(
            nm(s["parent"]), nm(s["child"]), s["relation"], s["cls"], s["bit_identical"], s["barely"], s["updated"],
            shs, s["shrink_pct"], ns["updated"], ns["n"], hds))
    if missing:
        o += ["", "Not finished: " + "; ".join(missing)]
    if SKIP:
        o += ["", "Skipped: " + "; ".join(f"{x['parent']} -> {x['child']} ({x['why']})" for x in SKIP)]
    o += ["", "## Share of vocabulary tokens bit-identical, by script and kind (reachable tokens only)", "",
          "| release (vs base) | " + " | ".join(c.replace("script:", "") for c in SCRIPTS) + " |",
          "|---|" + "---|" * len(SCRIPTS)]
    for s in S:
        if "categories" not in s:
            continue
        cells = []
        for c in SCRIPTS:
            v = s["categories"].get(c.replace("script:", ""))
            cells.append("-" if not v else f"{100 * v['share_bit_identical']:.0f}% of {v['n']:,}")
        o.append(f"| {nm(s['child'])} (vs {nm(s['parent']).split(' (')[0]}) | " + " | ".join(cells) + " |")
    o += ["", "## Named special tokens left bit-identical", ""]
    for s in S:
        b = s["named_special"]["bit_identical"]
        if b:
            o.append(f"- {nm(s['child'])} vs {nm(s['parent'])}: " + ", ".join(f"`{x}`" for x in b[:30]) + (" ..." if len(b) > 30 else ""))
    if TULU:
        o += ["", "## Tulu 3 405B: updated tokens vs the published data", ""]
        for w, res in TULU.items():
            if "cut" not in res:
                o.append(f"- {w}: {res.get('status')}"); continue
            for v in ("cut", "nocut"):
                x = res[v]
                o.append(f"- **{w}** ({'truncated as in training' if v == 'cut' else 'no truncation'}): {x['present']:,} distinct tokens in the data; "
                         f"{x['present_and_changed']:,} of them changed; {x['present_not_changed']:,} in the data but bit-identical "
                         f"(occurrences 1-5: {x['present_not_changed_occ_hist']}); {x['changed_not_present_regular']:,} regular tokens updated but absent from the data.")
            o.append(f"  - vectors: {res['bit_identical']:,} bit-identical, {res['barely']:,} barely changed, {res['updated']:,} updated; shrink fit {res.get('fit')}")
            if res["cut"]["changed_not_present_examples"]:
                o.append("  - updated but absent (examples): " + ", ".join(f"`{t}`" for _, t, _ in res["cut"]["changed_not_present_examples"][:15]))
            if res["cut"]["present_not_changed_examples"]:
                o.append("  - in the data but bit-identical (examples, count): " + ", ".join(f"`{t}` ({n})" for _, t, n in res["cut"]["present_not_changed_examples"][:15]))
    o += findings(S)
    (L / "LARGE.md").write_text("\n".join(o) + "\n")
    print("\n".join(o[:60]))


if __name__ == "__main__":
    main()
