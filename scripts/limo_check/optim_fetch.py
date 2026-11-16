# Fetch the embedding slice (first 152064*5120/32 floats) of exp_avg, exp_avg_sq, fp32 master weights for each ZeRO-3 rank.
import sys,os,struct,requests,numpy as np,json
from concurrent.futures import ThreadPoolExecutor
from optim_probe import layout
OUT=os.path.expanduser("~/training-forensics/results/limo_check/dl/optim"); os.makedirs(OUT,exist_ok=True)
N=152064*5120//32*4
def rng(url,a,b):
    for t in range(8):
        try:
            r=requests.get(url,headers={"Range":f"bytes={a}-{b-1}"},timeout=300); r.raise_for_status(); assert len(r.content)==b-a; return r.content
        except Exception as e: err=e
    raise err
def one(rank):
    if os.path.exists(f"{OUT}/r{rank}.done"): return
    url,ents,d=layout(rank)
    g={k.split("/",1)[1]:v for k,v in ents.items()}
    step=struct.unpack("<f",rng(url,g["data/0"][0],g["data/0"][0]+4))[0]
    sizes={k:v[1] for k,v in g.items()}
    for name,key in (("m","data/1"),("v","data/2"),("w","data/6")):
        off=g[key][0]; assert g[key][1]==4095427200
        parts=[(s,min(s+(32<<20),off+N)) for s in range(off,off+N,32<<20)]
        open(f"{OUT}/r{rank}_{name}.bin","wb").write(b"".join(rng(url,*p) for p in parts))
    json.dump({"rank":rank,"step":step,"sizes":sizes},open(f"{OUT}/r{rank}.done","w")); print(rank,step,flush=True)
with ThreadPoolExecutor(8) as ex: list(ex.map(one,range(32)))
