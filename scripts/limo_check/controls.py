import json,os,numpy as np,collections,torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from sympy import factorint
D=os.path.expanduser("~/training-forensics/results/limo_check/dl/")
print("flos/6 factors:",factorint(61492083294208))
tk=Tokenizer.from_file(D+"limo_small/tokenizer.json"); ids=lambda t: tk.encode(t,add_special_tokens=False).ids
base=np.fromfile(D+"q32i.bin",dtype=np.uint16).reshape(152064,5120)
# LIMO-v2
c2=np.load(D+"changed_mask_v2.npy"); C2=set(np.nonzero(c2)[0].tolist())
v2=[json.loads(l) for l in open(hf_hub_download("GAIR/LIMO-v2","limo-v2.jsonl",repo_type="dataset",local_dir=D+"ds_limo2"))]
print("v2 n",len(v2),v2[0].keys())
SYS="Please reason step by step, and put your final answer within \\boxed{}."
def render(s,q,a): return f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
ks=list(v2[0].keys()); qk=[k for k in ks if k in("question","instruction","problem")][0]; ak=[k for k in ks if k in("solution","output","response")][0]
S=set(); Sc=set()
for h in v2:
    x=ids(render(SYS,h[qk],h[ak])); S|=set(x); Sc|=set(x[:16384])
for n,s in (("full",S),("cut16384",Sc)): print("LIMO-v2",n,"tokens",len(s),"changed",len(C2),"in,unchanged",len(s-C2),"changed,absent",len(C2-s))
# s1 control (fp32 release vs bf16 base)
s1=np.fromfile(D+"s1.bin",dtype=np.float32).reshape(152064,5120)
bf=torch.from_numpy(base.view(np.int16)).view(torch.bfloat16).float().numpy()
cs=np.zeros(152064,bool)
for a in range(0,152064,8192): cs[a:a+8192]=(s1[a:a+8192]!=bf[a:a+8192]).any(1)
Cs=set(np.nonzero(cs)[0].tolist())
from datasets import load_dataset
ds=load_dataset("simplescaling/s1K_tokenized",split="train"); T=set(); Tl=set()
for t in ds["text"]:
    x=ids(t); T|=set(x); Tl|=set(x[:-1])
print("s1 control: changed",len(Cs),"data tokens",len(T),"excl last pos",len(Tl),"in,unchanged",len(Tl-Cs),"changed,absent",len(Cs-T))
