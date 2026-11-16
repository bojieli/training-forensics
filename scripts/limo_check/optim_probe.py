# Probe an old-revision DeepSpeed ZeRO-3 optimizer-state file by ranged reads (no unpickling: pickletools only).
import sys,zipfile,pickletools,requests,io,struct,json
from huggingface_hub import hf_hub_url
REV="1833babb94"
class Remote:
    def __init__(s,url):
        s.url=url; s.pos=0
        r=requests.head(url,allow_redirects=True); s.size=int(r.headers["Content-Length"]); s.url=r.url
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.pos
    def seek(s,o,w=0):
        s.pos = o if w==0 else (s.pos+o if w==1 else s.size+o); return s.pos
    def read(s,n=-1):
        if n<0: n=s.size-s.pos
        if n==0 or s.pos>=s.size: return b""
        r=requests.get(s.url,headers={"Range":f"bytes={s.pos}-{min(s.pos+n,s.size)-1}"},timeout=120); r.raise_for_status()
        s.pos+=len(r.content); return r.content
def layout(rank):
    url=hf_hub_url("GAIR/LIMO",f"global_step234/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt",revision=REV)
    f=Remote(url); z=zipfile.ZipFile(f)
    ents={}
    for i in z.infolist():
        # data offset = header_offset + 30 + len(name) + len(extra) ; read local header
        f.seek(i.header_offset); h=f.read(30); nl,el=struct.unpack("<HH",h[26:30])
        ents[i.filename]=(i.header_offset+30+nl+el,i.file_size,i.compress_type)
    pk=[n for n in ents if n.endswith("data.pkl")][0]
    f.seek(ents[pk][0]); d=f.read(ents[pk][1])
    return f.url,ents,d
if __name__=="__main__":
    url,ents,d=layout(int(sys.argv[1]))
    for k,v in ents.items(): print(k,v)
    print(" | ".join(f"{op.name}:{a!r}" if a is not None else op.name for op,a,p in pickletools.genops(d) if op.name not in ("BINPUT","LONG_BINPUT","MEMOIZE","PROTO","FRAME"))[:9000])
