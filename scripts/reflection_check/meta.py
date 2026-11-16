#!/usr/bin/env python3
"""Step 1: LFS sha256 of every safetensors shard, mirrors vs official meta-llama (HF API, no token)."""
import json, sys, urllib.request
def info(repo, rev=None):
    u = f"https://huggingface.co/api/models/{repo}" + (f"/revision/{rev}" if rev else "") + "?blobs=true"
    try:
        return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "python"}), timeout=60))
    except Exception as e:
        return {"error": str(e)}
def shards(i):
    return {s["rfilename"]: (s.get("lfs") or {}).get("sha256") or s.get("blobId") for s in i.get("siblings", [])}
PAIRS = [("NousResearch/Meta-Llama-3-70B-Instruct", "meta-llama/Meta-Llama-3-70B-Instruct"),
         ("NousResearch/Meta-Llama-3-70B", "meta-llama/Meta-Llama-3-70B"),
         ("unsloth/Meta-Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
         ("unsloth/Meta-Llama-3.1-70B", "meta-llama/Meta-Llama-3.1-70B"),
         ("unsloth/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.3-70B-Instruct")]
out = {}
for m, o in PAIRS:
    im, io = info(m), info(o)
    sm, so = shards(im), shards(io)
    st = sorted(k for k in sm if k.endswith(".safetensors"))
    ot = sorted(k for k in so if k.endswith(".safetensors"))
    same = [k for k in st if so.get(k) == sm[k]]
    diff = [k for k in st if k in so and so[k] != sm[k]]
    other = {k: (sm.get(k) == so.get(k)) for k in ("tokenizer.json", "config.json", "tokenizer_config.json", "generation_config.json") if k in sm or k in so}
    out[m] = dict(official=o, mirror_sha=im.get("sha"), official_sha=io.get("sha"), mirror_err=im.get("error"), official_err=io.get("error"),
                  gated=io.get("gated"), n_mirror=len(st), n_official=len(ot), sha256_equal=len(same), sha256_differ=diff, small_files_equal=other,
                  mirror_created=im.get("createdAt"), official_created=io.get("createdAt"), mirror_shards={k: sm[k] for k in st})
    print(m, "vs", o, {k: v for k, v in out[m].items() if k != "mirror_shards"}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
