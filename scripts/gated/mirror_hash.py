"""Compare LFS sha256 of weight files between community mirrors and official meta-llama repos (metadata only)."""
import json, os, sys
from huggingface_hub import HfApi
api = HfApi()
PAIRS = {
 "unsloth/Llama-4-Maverick-17B-128E": ["meta-llama/Llama-4-Maverick-17B-128E"],
 "unsloth/Llama-4-Maverick-17B-128E-Instruct": ["meta-llama/Llama-4-Maverick-17B-128E-Instruct"],
 "unsloth/Llama-4-Scout-17B-16E": ["meta-llama/Llama-4-Scout-17B-16E"],
 "unsloth/Llama-4-Scout-17B-16E-Instruct": ["meta-llama/Llama-4-Scout-17B-16E-Instruct"],
 "unsloth/Llama-3.3-70B-Instruct": ["meta-llama/Llama-3.3-70B-Instruct"],
 "unsloth/Meta-Llama-3.1-70B-Instruct": ["meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct"],
 "unsloth/Meta-Llama-3.1-70B": ["meta-llama/Llama-3.1-70B", "meta-llama/Meta-Llama-3.1-70B"],
 "unsloth/Meta-Llama-3.1-8B": ["meta-llama/Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B"],
 "unsloth/Meta-Llama-3.1-8B-Instruct": ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"],
 "unsloth/llama-3-8b": ["meta-llama/Meta-Llama-3-8B"],
 "unsloth/llama-3-8b-Instruct": ["meta-llama/Meta-Llama-3-8B-Instruct"],
 "unsloth/Llama-3.2-1B": ["meta-llama/Llama-3.2-1B"],
 "unsloth/Llama-3.2-1B-Instruct": ["meta-llama/Llama-3.2-1B-Instruct"],
 "unsloth/Llama-3.2-3B": ["meta-llama/Llama-3.2-3B"],
 "unsloth/Llama-3.2-3B-Instruct": ["meta-llama/Llama-3.2-3B-Instruct"],
 "NousResearch/Llama-2-7b-hf": ["meta-llama/Llama-2-7b-hf"],
 "NousResearch/Llama-2-7b-chat-hf": ["meta-llama/Llama-2-7b-chat-hf"],
 "NousResearch/Llama-2-13b-hf": ["meta-llama/Llama-2-13b-hf"],
 "NousResearch/Llama-2-13b-chat-hf": ["meta-llama/Llama-2-13b-chat-hf"],
 "NousResearch/Llama-2-70b-hf": ["meta-llama/Llama-2-70b-hf"],
 "NousResearch/Llama-2-70b-chat-hf": ["meta-llama/Llama-2-70b-chat-hf"],
 "NousResearch/Meta-Llama-3.1-405B-FP8": ["meta-llama/Llama-3.1-405B-FP8", "meta-llama/Meta-Llama-3.1-405B-FP8"],
 "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8": ["meta-llama/Llama-3.1-405B-Instruct-FP8", "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"],
}

def shas(repo):
    info = api.model_info(repo, files_metadata=True)
    return info.sha, {s.rfilename: s.lfs.sha256 for s in info.siblings
                      if s.lfs and s.rfilename.endswith((".safetensors", ".bin", ".pth"))}

out = []
for mir, offs in PAIRS.items():
    rec = dict(mirror=mir)
    try:
        rec["mirror_rev"], m = shas(mir)
    except Exception as e:
        rec["error"] = f"mirror: {type(e).__name__}"; out.append(rec); print(rec); continue
    for off in offs:
        try:
            rev, o = shas(off)
        except Exception as e:
            rec.setdefault("tried", []).append(f"{off}: {type(e).__name__}"); continue
        mset, oset = set(m.values()), set(o.values())
        rec.update(official=off, official_rev=rev, mirror_files=len(m), official_files=len(o),
                   shared_sha=len(mset & oset), mirror_all_in_official=mset <= oset and len(mset) > 0,
                   identical_set=mset == oset and len(mset) > 0)
        break
    out.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("mirror_rev", "official_rev")}))
json.dump(out, open(sys.argv[1], "w"), indent=1)
