import json,os,numpy as np,collections,glob
from tokenizers import Tokenizer
D=os.path.expanduser("~/training-forensics/results/limo_check/dl/")
tk=Tokenizer.from_file(D+"limo_small/tokenizer.json"); ids=lambda t: tk.encode(t,add_special_tokens=False).ids
vnz=np.load(D+"v_nz.npy"); C=set(np.nonzero(vnz)[0].tolist()); st=np.load(D+"row_stats.npy")
gh=json.load(open(D+"gh_limo/train/data/limo.json"))
def render(s,q,a): return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
seqs=[ids(render(g["system"],g["instruction"],g["output"])) for g in gh]
df=collections.Counter(); first={}; 
for i,x in enumerate(seqs):
    for p,t in enumerate(x):
        if t not in first or p<first[t][0]: first[t]=(p,i)
    df.update(set(x))
P=set(df); unch=sorted(P-C)
print("published tokens",len(P),"never-got-gradient (v==0):",len(unch))
pos=np.array([first[t][0] for t in unch]); print("min-position quantiles of never-updated published tokens:",np.percentile(pos,[0,10,25,50,75,90,100]).tolist(),"n with minpos<16384:",int((pos<16384).sum()),"<8192:",int((pos<8192).sum()),"<2000:",int((pos<2000).sum()))
print("df of never-updated:",collections.Counter(df[t] for t in unch).most_common(6))
# per-example: df==1 tokens (unique to that example), fraction never updated
per=[]
for i,x in enumerate(seqs):
    u=set(t for t in x[:16384] if df[t]==1); 
    if u: per.append((i,len(u),sum(t not in C for t in u),len(x)))
tot=sum(p[1] for p in per); bad=sum(p[2] for p in per)
print("examples with >=1 unique token:",len(per),"unique tokens total",tot,"of which never updated",bad)
exs=[p for p in per if p[2]>0]; print("examples having >=1 never-updated unique token (<16384):",len(exs))
print("  all-unique-unchanged examples:",sum(1 for p in per if p[2]==p[1]),"| mixed:",sum(1 for p in per if 0<p[2]<p[1]))
print("  sample (idx,nuniq,nunchanged,len):",sorted(exs,key=lambda p:-p[2])[:15])
# examples containing any never-updated token within first 16384
ex_any=set(i for i,x in enumerate(seqs) if any(t not in C for t in x[:16384])); print("examples with any never-updated token in first 16384:",len(ex_any))
ex_any2=set(i for i,x in enumerate(seqs) if any(t not in C for t in x[:4096])); print("... in first 4096:",len(ex_any2))
# decode some with context
for i,nu,nb,L in sorted(exs,key=lambda p:-p[2])[:4]:
    x=seqs[i]; ps=[p for p,t in enumerate(x[:16384]) if t not in C][:3]
    for p in ps: print("  ex",i,"pos",p,"/",L,repr(tk.decode(x[max(0,p-12):p]))," [[",repr(tk.decode([x[p]])),"]] ",repr(tk.decode(x[p+1:p+8])))
# total tokens vs flos
lens=np.array([len(x) for x in seqs]); f=368952499765248.0; n=17408*64+5120
print("flos-implied tokens/epoch (832 slots):",f/6/n/9,"| published sum len full:",int(lens.sum()),"cut16384:",int(np.minimum(lens,16384).sum()),"cut32768:",int(np.minimum(lens,32768).sum()),"mean",lens.mean())
# absent tokens: stats vs in-data by df
A=sorted(C-P); vs=st[:,0]; dwm=st[:,2]; r=st[:,1]/np.maximum(st[:,0],1e-300)
def q(S): S=list(S); return "n=%d v_sum med %.3g | mean|dw| med %.3g | m2/v med %.3g"%(len(S),np.median(vs[S]),np.median(dwm[S]),np.median(r[S]))
print("absent      ",q(A))
for lo,hi in ((1,1),(2,3),(4,10),(11,100),(101,817)): print("in-data df %d-%d"%(lo,hi),q([t for t in P&C if lo<=df[t]<=hi]))
k=lambda S: np.clip(np.round(np.log(np.maximum(r[list(S)],1e-30)/10*9)/np.log(0.81/0.999)),-5,60).astype(int)
print("k-est hist absent:",np.bincount(k(A)+5,minlength=66)[:45].tolist()); print("k-est hist df1   :",np.bincount(k([t for t in P&C if df[t]==1])+5,minlength=66)[:45].tolist())
# eval sets coverage
for fpath in sorted(glob.glob(D+"gh_limo/eval/data/*/*.jsonl")+glob.glob(D+"gh_limo/eval/**/*.json",recursive=True)):
    try:
        rows=[json.loads(l) for l in open(fpath)] if fpath.endswith("jsonl") else json.load(open(fpath))
    except Exception as e: print(fpath,"ERR",e); continue
    if isinstance(rows,dict): rows=list(rows.values())
    T=set(); full=0; nq=0
    for rw in rows:
        if not isinstance(rw,dict): continue
        qtxt=next((rw[kk] for kk in ("question","problem","Question","prompt","input") if kk in rw and isinstance(rw[kk],str)),None)
        if qtxt is None: continue
        s=set(ids(qtxt)); T|=s; nq+=1; full+= s<=C
    if nq: print(os.path.relpath(fpath,D+"gh_limo"),"n",nq,"tokens",len(T),"in changed %.3f"%(len(T&C)/len(T)),"new tokens (not in published) %d of which changed %.3f"%(len(T-P),len((T-P)&C)/max(1,len(T-P))),"questions fully covered by changed set: %d (by published set: %d)"%(full,sum(1 for rw in rows if isinstance(rw,dict) and any(kk in rw for kk in ("question","problem")) and set(ids(str(rw.get("question",rw.get("problem","")))))<=P)))
json.dump({"absent":A,"never_updated_published":unch},open(D+"../sets.json","w"))
