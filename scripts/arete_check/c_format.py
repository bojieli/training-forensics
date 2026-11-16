#!/usr/bin/env python3
"""Which rows / which format did the trainer use? Match Nous blog data.js (sorted per-example lengths, per-topic counts)."""
import glob, json, sys, collections, numpy as np
from transformers import AutoTokenizer
D = sys.argv[1]
s = open(f"{D}/nous_data.js").read(); N = json.loads(s[s.index("{"):s.rindex("}")+1])
SL = collections.Counter(N["sorted_lengths"]); L = N["lengths"]
tok = AutoTokenizer.from_pretrained(f"{D}/tok_base")
rows = []
for f in sorted(glob.glob(f"{D}/data/sft_accepted/*.jsonl")):
    rows += [json.loads(l) for l in open(f)]
topics = [r["metadata"]["topic"] for r in rows]
print("published per topic", collections.Counter(topics)); print("nous per topic", N["topics"])
def lens(docs):
    out = []
    for k in range(0, len(docs), 2048): out += [len(x) for x in tok(docs[k:k+2048], add_special_tokens=False)["input_ids"]]
    return np.array(out)
P = [r["messages"][1]["content"] for r in rows]; A = [r["messages"][2]["content"] for r in rows]; S = rows[0]["messages"][0]["content"]
lp, la = lens(P), lens(A)
print("nous: total", L["total_tokens"], "answer", L["answer_tokens"], "non-answer per ex", (L["total_tokens"]-L["answer_tokens"])/L["n"], "qwen extra", L["qwen_template_extra_per_example"])
print("mine all rows: prompt mean", lp.mean(), "answer mean", la.mean())
IM, IE = "<|im_start|>", "<|im_end|>"
fm = {
 "P+A": [p + a for p, a in zip(P, A)],
 "chatml_nosys_nothink": [f"{IM}user\n{p}{IE}\n{IM}assistant\n{a}{IE}\n" for p, a in zip(P, A)],
 "chatml_sys_nothink": [f"{IM}system\n{S}{IE}\n{IM}user\n{p}{IE}\n{IM}assistant\n{a}{IE}\n" for p, a in zip(P, A)],
 "chatml_sys_emptythink": [f"{IM}system\n{S}{IE}\n{IM}user\n{p}{IE}\n{IM}assistant\n<think>\n\n</think>\n\n{a}{IE}\n" for p, a in zip(P, A)],
 "chatml_nosys_emptythink": [f"{IM}user\n{p}{IE}\n{IM}assistant\n<think>\n\n</think>\n\n{a}{IE}\n" for p, a in zip(P, A)],
}
res = {}
for k, docs in fm.items():
    l = lens(docs); res[k] = l
    c = collections.Counter(l.tolist())
    # constant shift scan: sorted_lengths must be a sub-multiset of {l+c}
    best = []
    for sh in range(-40, 41):
        miss = sum(max(0, n - c.get(v - sh, 0)) for v, n in SL.items())
        best.append((miss, sh))
    best.sort()
    print(f"{k}: mean {l.mean():.2f} max {l.max()} | best constant shifts (nous lengths not coverable, shift): {best[:3]}")
np.savez(f"{D}/fmt_lens.npz", lp=lp, la=la, **res)
