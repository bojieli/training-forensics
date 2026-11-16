import json, sys
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
cands = set()
for q in ["Llama-3.1-405B", "Meta-Llama-3.1-405B"]:
    for m in api.list_models(search=q, limit=300):
        if not m.id.startswith("meta-llama/"): cands.add(m.id)
def shas(r):
    i = api.model_info(r, files_metadata=True)
    return i, {s.rfilename: s.lfs.sha256 for s in i.siblings if s.lfs and s.rfilename.endswith(".safetensors")}
offs = {k: set(shas(o)[1].values()) for k, o in {"base": "meta-llama/Llama-3.1-405B", "inst": "meta-llama/Llama-3.1-405B-Instruct"}.items()}
out = []
for c in sorted(cands):
    try:
        ci, cd = shas(c)
        if ci.gated or not cd: continue
        idx = [f for f in cd if False] or [s.rfilename for s in ci.siblings if s.rfilename.endswith("model.safetensors.index.json")]
        if not idx: continue
        wm = json.load(open(hf_hub_download(c, idx[0], revision=ci.sha)))["weight_map"]
        ef = {f for n, f in wm.items() if "embed_tokens" in n}
        for k, oset in offs.items():
            hit = [f for f in ef if cd.get(f) in oset]
            if hit:
                rec = dict(mirror=c, rev=ci.sha, official=k, embed_file=hit[0], shared=len(oset & set(cd.values())), official_files=len(oset))
                out.append(rec); print("MATCH", json.dumps(rec), flush=True)
    except Exception as e:
        pass
json.dump(out, open(sys.argv[1], "w"), indent=1)
