"""Per-mirror: are the files actually used (embedding/lm_head shards, safetensors) byte-identical to the official repo's?"""
import json, sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
prev = json.load(open(sys.argv[1]))
out = []
for r in prev:
    mir, off = r["mirror"], r.get("official")
    if not off: out.append(r); continue
    mi = api.model_info(mir, files_metadata=True); oi = api.model_info(off, files_metadata=True)
    m = {s.rfilename: s.lfs.sha256 for s in mi.siblings if s.lfs and s.rfilename.endswith(".safetensors")}
    o = {s.rfilename: s.lfs.sha256 for s in oi.siblings if s.lfs and s.rfilename.endswith(".safetensors")}
    # which mirror files hold the embedding / output matrix
    emb_files = set()
    idx = [s.rfilename for s in mi.siblings if s.rfilename.endswith("model.safetensors.index.json")]
    if idx:
        wm = json.load(open(hf_hub_download(mir, idx[0], revision=mi.sha)))["weight_map"]
        emb_files = {f for k, f in wm.items() if any(t in k for t in ("embed_tokens", "lm_head", "tok_embeddings", "output.weight"))}
    elif len(m) == 1:
        emb_files = set(m)
    oset = set(o.values())
    rec = dict(mirror=mir, mirror_rev=mi.sha, official=off, official_rev=oi.sha,
               safetensors_mirror=len(m), safetensors_official=len(o),
               safetensors_identical=sum(v in oset for v in m.values()),
               embedding_files=sorted(emb_files),
               embedding_files_identical=all(m[f] in oset for f in emb_files) if emb_files else None)
    out.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if not k.endswith("_rev")}))
json.dump(out, open(sys.argv[2], "w"), indent=1)
