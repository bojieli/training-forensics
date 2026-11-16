# Independent fetch of model.embed_tokens.weight via ranged reads of safetensors shards.
import sys,json,struct,requests,numpy as np,os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_url
OUT=os.path.expanduser("~/training-forensics/results/limo_check/dl")
def rng(url,a,b):
    for t in range(8):
        try:
            r=requests.get(url,headers={"Range":f"bytes={a}-{b-1}"},timeout=120); r.raise_for_status()
            assert len(r.content)==b-a; return r.content
        except Exception as e: err=e
    raise err
def fetch(repo,rev,tag):
    idx=requests.get(hf_hub_url(repo,"model.safetensors.index.json",revision=rev)).json()["weight_map"]
    names=[k for k in idx if "embed_tokens" in k]; assert names==["model.embed_tokens.weight"],names
    tied="lm_head.weight" not in idx
    url=hf_hub_url(repo,idx[names[0]],revision=rev)
    n=struct.unpack("<Q",rng(url,0,8))[0]; hdr=json.loads(rng(url,8,8+n)); m=hdr[names[0]]
    a,b=m["data_offsets"]; a+=8+n; b+=8+n
    print(repo,rev,idx[names[0]],m["dtype"],m["shape"],"lm_head_in_index",not tied,flush=True)
    step=32<<20; parts=[(s,min(s+step,b)) for s in range(a,b,step)]
    with ThreadPoolExecutor(12) as ex: bufs=list(ex.map(lambda p:rng(url,*p),parts))
    raw=b"".join(bufs)
    open(f"{OUT}/{tag}.bin","wb").write(raw)
    json.dump({"repo":repo,"rev":rev,"dtype":m["dtype"],"shape":m["shape"],"shard":idx[names[0]]},open(f"{OUT}/{tag}.json","w"))
for spec in sys.argv[1:]:
    repo,rev,tag=spec.split(","); fetch(repo,rev,tag)
