#!/usr/bin/env python3
"""DeepSeek V4 task special tokens: documented vs undocumented.

DeepSeek's V4 repos document DS_TASK_SP_TOKENS ("task special tokens for internal classification
tasks") in encoding/encoding_dsv4.py and encoding/README.md: action, query, authority, domain,
title, read_url (README also lists extracted_url with read_url). political, safety, entity, task,
answer and search are in tokenizer.json but not in that documentation.

For each DeepSeek V4 release vs its base: per-token change statistics, and the similarity between
the changes (release - base) of every pair of tokens. Question: do political/safety change like the
documented classifier tokens, and do they correlate more with them than with other tokens?
Writes results/frontier_ds/v4_task_tokens.json.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ds_tokens import load, OUT  # noqa: E402

DOC = {128829: "action", 128830: "query", 128831: "authority", 128832: "domain", 128836: "title",
       128845: "read_url", 128844: "extracted_url"}
UNDOC = {128834: "political", 128837: "safety", 128835: "entity", 128833: "task", 128838: "answer",
         128839: "search"}
OTHER = {128826: "begin_sys", 128827: "end_sys", 128828: "latest_reminder", 128842: "search_begin",
         128843: "search_end", 128846: "end_of_query", 129279: "image", 129276: "box", 129278: "ref"}
DOC_CLS = ["query", "authority", "domain"]     # the documented prompt classifiers appended after the user turn

PAIRS = [("deepseek-ai/DeepSeek-V4-Pro-Base", "deepseek-ai/DeepSeek-V4-Pro"),
         ("deepseek-ai/DeepSeek-V4-Flash-Base", "deepseek-ai/DeepSeek-V4-Flash"),
         ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro-0813"),
         ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-0731")]


def main():
    allt = {**DOC, **UNDOC, **OTHER}
    ids = list(allt)
    out = dict(documentation=dict(
        url_code="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/encoding_dsv4.py",
        url_readme="https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md",
        documented=sorted(DOC.values()), undocumented=sorted(UNDOC.values())), comparisons={})
    for p, c in PAIRS:
        A, _ = load(p)
        B, _ = load(c)
        Af, Bf = A.float(), B.float()
        D = Bf - Af
        tok = {}
        for i in ids:
            tok[allt[i]] = dict(group="documented" if i in DOC else "undocumented" if i in UNDOC else "other",
                                frac_changed=round(float((A[i] != B[i]).float().mean()), 4),
                                size_base=round(float(Af[i].norm()), 4), change_size=round(float(D[i].norm()), 4),
                                sim_to_base=round(float(torch.nn.functional.cosine_similarity(Af[i], Bf[i], dim=0)), 4))
        X = D[ids]
        X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-30)
        S = (X @ X.T).numpy()
        name = [allt[i] for i in ids]
        sim = {name[a]: {name[b]: round(float(S[a, b]), 3) for b in range(len(ids)) if b != a} for a in range(len(ids))}
        # reference: random ordinary-token pairs
        g = torch.Generator().manual_seed(0)
        r = torch.randint(3, 128000, (400,), generator=g)
        R_ = D[r] / D[r].norm(dim=1, keepdim=True).clamp_min(1e-30)
        Sr = (R_ @ R_.T)[torch.triu(torch.ones(400, 400, dtype=bool), 1)]
        ref95 = float(torch.quantile(Sr, 0.95))

        def mean_sim(t, group):
            v = [sim[t][u] for u in group if u != t and u in sim[t]]
            return round(statistics.mean(v), 3) if v else None

        undoc_wo = lambda t: [u for u in UNDOC.values() if u != t]
        other = list(OTHER.values())
        focus = {}
        for t in ("political", "safety"):
            focus[t] = dict(with_documented_classifiers=mean_sim(t, DOC_CLS),
                            with_all_documented_except_action=mean_sim(t, [x for x in DOC.values() if x != "action"]),
                            with_other_undocumented=mean_sim(t, undoc_wo(t)),
                            with_other_named=mean_sim(t, other),
                            top3=sorted(sim[t].items(), key=lambda kv: -kv[1])[:3])
        # groups: documented (excluding action, whose vector was replaced/erased) vs undocumented
        dl = [x for x in DOC.values() if x != "action"]
        ul = list(UNDOC.values())
        within = lambda L: [sim[a][b] for i, a in enumerate(L) for b in L[i + 1:]]
        out["comparisons"][f"{p.split('/')[1]} -> {c.split('/')[1]}"] = dict(
            tokens=tok, ref_random_pairs_p95=round(ref95, 3), focus=focus,
            within_documented_mean=round(statistics.mean(within(dl)), 3),
            within_undocumented_mean=round(statistics.mean(within(ul)), 3),
            documented_x_undocumented_mean=round(statistics.mean([sim[a][b] for a in dl for b in ul]), 3),
            within_other_named_mean=round(statistics.mean(within(other)), 3),
            similarity=sim)
        print(c, json.dumps(focus), flush=True)
    json.dump(out, open(OUT / "v4_task_tokens.json", "w"), indent=1)


if __name__ == "__main__":
    main()
