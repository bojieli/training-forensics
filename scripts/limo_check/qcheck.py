#!/usr/bin/env python3
"""Step 6: questions vs solutions, and CJK base rates. -> out/qcheck.json"""
import json, collections, numpy as np
from tokenizers import Tokenizer
from common import D, save, gh

tk = Tokenizer.from_file(D + "limo_files/tokenizer.json"); ids = lambda t: tk.encode(t, add_special_tokens=False).ids
C = set(np.nonzero(np.load(D + "v_nz.npy"))[0].tolist())
ghd = json.load(open(gh("train/data/limo.json")))
Q = [set(ids("user\n" + g["instruction"] + "<|im_end|>")) for g in ghd]
S = [set(ids("assistant\n" + g["output"] + "<|im_end|>")) for g in ghd]
allS = set().union(*S); allQ = set().union(*Q)
dfq = collections.Counter(t for q in Q for t in q); dfs = collections.Counter(t for s in S for t in s)
qonly = [t for t in dfq if t not in allS]
flag = [i for i, s in enumerate(S) if any(t not in C for t in s)]
u = [t for i in flag for t in S[i] if dfs[t] == 1 and t not in dfq]
voc = tk.get_vocab()
cjk = [i for s, i in voc.items() if any("一" <= c <= "鿿" for c in tk.decode([i]))]
out = {
    "question_tokens": len(allQ), "question_tokens_without_gradient": len(allQ - C),
    "question_only_tokens": len(qonly), "question_only_with_gradient": sum(t in C for t in qonly),
    "question_only_in_one_question": sum(dfq[t] == 1 for t in qonly),
    "published_tokens_with_gradient_share": len((allQ | allS) & C) / len(allQ | allS),
    "examples_with_solution_token_without_gradient (full text)": len(flag),
    "their_solution_unique_tokens": len(u), "their_solution_unique_tokens_with_gradient_share": sum(t in C for t in u) / max(1, len(u)),
    "cjk_vocab_rows": len(cjk), "cjk_rows_with_gradient": sum(i in C for i in cjk),
    "all_rows_with_gradient_share": len(C) / 152064,
}
print(json.dumps(out, indent=1), flush=True)
save("qcheck", out)
