import json,os,numpy as np,collections
from tokenizers import Tokenizer
D=os.path.expanduser("~/training-forensics/results/limo_check/dl/")
tk=Tokenizer.from_file(D+"limo_small/tokenizer.json"); ids=lambda t: tk.encode(t,add_special_tokens=False).ids
C=set(np.nonzero(np.load(D+"v_nz.npy"))[0].tolist())
gh=json.load(open(D+"gh_limo/train/data/limo.json"))
Q=[set(ids("user\n"+g["instruction"]+"<|im_end|>")) for g in gh]; S=[set(ids("assistant\n"+g["output"]+"<|im_end|>")) for g in gh]
allS=set().union(*S); dfq=collections.Counter(t for q in Q for t in q)
qonly=[t for t in dfq if t not in allS]; print("question-only tokens:",len(qonly),"changed:",sum(t in C for t in qonly),"| with df==1:",sum(dfq[t]==1 for t in qonly),"examples covered:",len({i for i,q in enumerate(Q) for t in q if t not in allS}))
# chance rate: solution-unique tokens in examples flagged as differing
dfs=collections.Counter(t for s in S for t in s)
flag=[i for i,s in enumerate(S) if any(t not in C for t in s)]
u=[t for i in flag for t in S[i] if dfs[t]==1 and t not in dfq]; print("flagged examples",len(flag),"their solution-unique tokens",len(u),"changed frac %.3f"%(sum(t in C for t in u)/len(u)))
# vocabulary-wide base rates
voc=tk.get_vocab(); n=152064
cjk=[i for s,i in voc.items() if any("一"<=c<="鿿" for c in tk.decode([i]))]
print("CJK-containing vocab rows:",len(cjk),"changed:",sum(i in C for i in cjk),"| overall changed frac %.3f"%(len(C)/n))
