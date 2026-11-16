#!/usr/bin/env python3
"""Step 5: published examples against the rows that received a gradient (nonzero Adam second moment).
Cutoff dependence, per-example evidence, the size of the unexplained set, evaluation-set coverage with base rates.
-> out/deeper.json, out/sets.json"""
import json, glob, os, collections, numpy as np
from tokenizers import Tokenizer
from common import D, save, gh, OUT

tk = Tokenizer.from_file(D + "limo_files/tokenizer.json"); ids = lambda t: tk.encode(t, add_special_tokens=False).ids
C = set(np.nonzero(np.load(D + "v_nz.npy"))[0].tolist())
C2 = set(np.nonzero(np.load(D + "changed_mask_v2.npy"))[0].tolist())
Cs = set(np.nonzero(np.load(D + "changed_mask_s1.npy"))[0].tolist())
ghd = json.load(open(gh("train/data/limo.json")))
render = lambda s, q, a: f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
seqs = [ids(render(g["system"], g["instruction"], g["output"])) for g in ghd]
out = {"gradient_rows": len(C)}

# --- cutoff dependence: the run's cutoff is not in the released files (LLaMA-Factory data argument).
# GitHub train_limo.yaml said 32768 on the release day and 16384 from 2025-02-14; the paper says 16384.
cut = {}
for K in (1024, 2048, 4096, 8192, 16384, 32768):
    P = set(t for x in seqs for t in x[:K])
    cut[str(K)] = {"distinct_tokens_in_published_set": len(P), "in_set_with_gradient": len(P & C), "in_set_without_gradient": len(P - C),
                   "gradient_rows_absent_from_set": len(C - P),
                   "examples_with_a_token_without_gradient": sum(1 for x in seqs if any(t not in C for t in x[:K]))}
out["by_cutoff"] = cut
out["max_published_length"] = max(len(x) for x in seqs)

# --- first position of each published token that never got a gradient
df = collections.Counter(); first = {}
for i, x in enumerate(seqs):
    for p, t in enumerate(x):
        if t not in first or p < first[t]: first[t] = p
    df.update(set(x))
P = set(df); unch = sorted(P - C)
pos = np.array([first[t] for t in unch])
out["no_gradient_published_tokens"] = {"n": len(unch), "first_position_quantiles": dict(zip(("0", "10", "25", "50", "75", "90", "100"), np.percentile(pos, [0, 10, 25, 50, 75, 90, 100]).tolist())),
                                       "doc_freq_hist": dict(collections.Counter(df[t] for t in unch).most_common(6))}

# --- per example: tokens unique to that example in the published set (window 16384)
W = 16384
per = []
for i, x in enumerate(seqs):
    u = set(t for t in x[:W] if df[t] == 1)
    per.append((i, len(u), sum(t not in C for t in u)))
with_u = [p for p in per if p[1] > 0]
yields = np.array([p[1] for p in with_u])
A = sorted(C - set(t for x in seqs for t in x[:W]))
out["per_example_unique_tokens"] = {
    "window": W, "examples": len(seqs), "examples_with_unique_tokens": len(with_u),
    "all_unique_tokens_have_gradient": sum(1 for p in with_u if p[2] == 0),
    "some_unique_token_without_gradient": sum(1 for p in with_u if p[2] > 0),
    "all_unique_tokens_without_gradient": sum(1 for p in with_u if p[2] == p[1]),
    "examples_without_unique_tokens (cannot be flagged)": len(seqs) - len(with_u),
    "unique_tokens_per_example": {"mean": float(yields.mean()), "median": float(np.median(yields)), "p90": float(np.percentile(yields, 90)), "max": int(yields.max())},
    "gradient_rows_absent_from_set": len(A),
    # if each differing example's trained text adds new distinct tokens at the rate published examples add unique ones,
    # the absent set needs about |A| / yield differing examples; |A| / max is a loose floor.
    "differing_examples_needed_at_mean_yield": round(len(A) / float(yields.mean())),
    "differing_examples_needed_at_max_yield": round(len(A) / int(yields.max())),
}

# --- evaluation-set coverage, with base rates from other token sets of known origin
def cover(S, qsets): return sum(1 for s in qsets if s <= S)
sets = {"LIMO gradient rows": C, "LIMO-v2 changed rows (other data)": C2, "s1-32B changed rows (other data)": Cs, "published LIMO tokens": P}
ev = {"set_sizes": {k: len(v) for k, v in sets.items()}, "files": {}}
for fpath in sorted(glob.glob(gh("eval/data/*/test.jsonl"))):
    rows = [json.loads(l) for l in open(fpath)]
    qs = []
    for rw in rows:
        q = next((rw[k] for k in ("question", "problem", "Question", "prompt", "input") if isinstance(rw.get(k), str)), None)
        if q is not None: qs.append(set(ids(q)))
    if qs:
        ev["files"][fpath.split("eval/data/")[1].split("/")[0]] = {"questions": len(qs), **{k: cover(S, qs) for k, S in sets.items()}}
# evaluation-set coverage goes to a local, git-ignored file: the report deliberately does not publish it
json.dump(ev, open(OUT / "eval_coverage.local.json", "w"), indent=1)
out["eval_questions_fully_covered"] = 'not committed: written to results/limo_check/out/eval_coverage.local.json by scripts/limo_check/deeper.py; token coverage cannot show that a question was trained on (see PUBLISHED_LIMO.md)'
json.dump({"absent_cut16384": A, "published_without_gradient": unch}, open(OUT / "sets.json", "w"))
print(json.dumps(out, indent=1), flush=True)
save("deeper", out)
