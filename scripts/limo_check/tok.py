# Independent tokenization of the published LIMO data (GitHub limo.json and HF limo.jsonl), no LLaMA-Factory code.
import json,os,numpy as np,collections
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
D=os.path.expanduser("~/training-forensics/results/limo_check/dl/")
tk=Tokenizer.from_file(hf_hub_download("GAIR/LIMO","tokenizer.json",local_dir=D+"limo_small"))
tb=Tokenizer.from_file(hf_hub_download("Qwen/Qwen2.5-32B-Instruct","tokenizer.json",local_dir=D+"q_small"))
va,vb=tk.get_vocab(with_added_tokens=True),tb.get_vocab(with_added_tokens=True); print("vocab equal:",va==vb,len(va))
gh=json.load(open(D+"gh_limo/train/data/limo.json"))
hf=[json.loads(l) for l in open(hf_hub_download("GAIR/LIMO","limo.jsonl",repo_type="dataset",local_dir=D+"ds_limo"))]
print(len(gh),len(hf),hf[0].keys())
print("gh==hf question/solution:",sum(g["instruction"]==h["question"] for g,h in zip(gh,hf)),sum(g["output"]==h["solution"] for g,h in zip(gh,hf)))
SYS=gh[0]["system"]
def render(s,q,a): return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
def ids(t): return tk.encode(t,add_special_tokens=False).ids
cnt=collections.Counter(); cntq=collections.Counter(); lens=[]; cut=collections.Counter()
for g in gh:
    x=ids(render(g["system"],g["instruction"]+("\n"+g["input"] if g["input"] else ""),g["output"])); lens.append(len(x))
    cnt.update(set(x)); cut.update(set(x[:16384])); cntq.update(set(ids(g["instruction"])))
hfset=set()
for h in hf:
    for k,v in h.items(): hfset.update(ids(str(v))); hfset.update(ids(" "+str(v))); hfset.update(ids("\n"+str(v)))
print("max len",max(lens),"n>16384",sum(l>16384 for l in lens),"total tokens",sum(lens))
ch=np.load(D+"changed_mask.npy"); C=set(np.nonzero(ch)[0].tolist())
for name,S in (("full",set(cnt)),("cut16384",set(cut)),("full+all HF fields",set(cnt)|hfset),("questions only",set(cntq))):
    print(name,"tokens",len(S),"changed",len(C),"in&changed",len(S&C),"in,unchanged",len(S-C),"changed,absent",len(C-S))
json.dump({"doc_freq":cnt,"q_doc_freq":cntq,"lens":lens},open(D+"../tok_limo.json","w"))
# brute-force: any changed-absent token string appearing as substring in raw data text?
raw="\n".join(json.dumps(h,ensure_ascii=False) for h in hf)+"\n".join(g["output"]+g["instruction"] for g in gh)
ab=sorted(C-set(cnt)-hfset); hit=0; cjk=0
for t in ab:
    s=tk.decode([t])
    if any("一"<=c<="鿿" for c in s):
        cjk+=1
        if s in raw: hit+=1
print("absent CJK tokens",cjk,"whose decoded string occurs anywhere in raw published text:",hit, "| CJK chars in raw text:",sum("一"<=c<="鿿" for c in raw))
