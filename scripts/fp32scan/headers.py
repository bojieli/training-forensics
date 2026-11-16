import json, sys, requests
sys.path.insert(0, ".")
from trainforensics import remote
from huggingface_hub import hf_hub_url
C = [("simplescaling/s1.1-32B","Qwen/Qwen2.5-32B-Instruct"),("simplescaling/s1-32B","Qwen/Qwen2.5-32B-Instruct"),
     ("agentica-org/DeepScaleR-1.5B-Preview","deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
     ("agentica-org/DeepCoder-14B-Preview","deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
     ("microsoft/UserLM-8b","meta-llama/Llama-3.1-8B"),
     ("marin-community/marin-8b-instruct","marin-community/marin-8b-base"),
     ("nvidia/Nemotron-Research-Reasoning-Qwen-1.5B","deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
     ("agentica-org/DeepSWE-Preview","Qwen/Qwen3-32B"),("open-thoughts/OpenThinker-Agent-v1","Qwen/Qwen3-8B"),
     ("microsoft/NextCoder-7B","Qwen/Qwen2.5-Coder-7B-Instruct"),("Zyphra/ZR1-1.5B","deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
     ("princeton-nlp/Llama-3-8B-ProLong-64k-Instruct","princeton-nlp/Llama-3-8B-ProLong-64k-Base"),
     ("allenai/Olmo-3.1-32B-Instruct-DPO","allenai/Olmo-3.1-32B-Instruct-SFT"),
     ("Felladrin/Llama-160M-Chat-v1","JackFram/llama-160m"),
     ("saraprice/llama2-7B-backdoor-DEPLOYMENT","meta-llama/Llama-2-7b-hf"),
     ("bosonai/Higgs-Llama-3-70B","meta-llama/Meta-Llama-3-70B"),
     ("facebook/MobileLLM-R1-950M","facebook/MobileLLM-R1-950M-base"),
     ("ngxson/MiniThinky-v2-1B-Llama-3.2","meta-llama/Llama-3.2-1B-Instruct"),
     ("HuggingFaceTB/nanowhale-100m","HuggingFaceTB/nanowhale-100m-base"),
     ("Nikity/lille-130m-instruct","Nikity/lille-130m-base"),
     ("simplescaling/s1.1-7B","Qwen/Qwen2.5-7B-Instruct"),("simplescaling/s1.1-14B","Qwen/Qwen2.5-14B-Instruct"),
     ("simplescaling/s1.1-3B","Qwen/Qwen2.5-3B-Instruct"),("simplescaling/s1.1-1.5B","Qwen/Qwen2.5-1.5B-Instruct"),
     ("agentica-org/DeepCoder-1.5B-Preview","deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
     ("microsoft/NextCoder-14B","Qwen/Qwen2.5-Coder-14B-Instruct"),("microsoft/NextCoder-32B","Qwen/Qwen2.5-Coder-32B-Instruct"),
     ("Zyphra/Zamba2-1.2B-instruct","Zyphra/Zamba2-1.2B"),
     ]
try:  # optional local, unpublished additions
    from private_pairs import PAIRS as _P
    C += _P
except ImportError:
    pass
def emb(repo):
    try:
        ix = remote.index(repo)
    except Exception as e:
        return None, f"ERR {str(e)[:60]}"
    names = [k for k in ix if k.endswith(("embed_tokens.weight","wte.weight","word_embeddings.weight","tok_embeddings.weight"))]
    heads = [k for k in ix if k.endswith(("lm_head.weight","output.weight","embed_out.weight"))]
    out = {}
    for k in names[:1] + heads[:1]:
        try:
            out[k] = remote.tensor_info(repo, k)
        except Exception as e:
            out[k] = f"ERR {str(e)[:40]}"
    try:
        cfg = requests.get(hf_hub_url(repo, "config.json"), timeout=60).json()
        tie = cfg.get("tie_word_embeddings", cfg.get("text_config", {}).get("tie_word_embeddings"))
    except Exception:
        tie = "?"
    return out, tie
res = []
for c, p in C:
    ce, ct = emb(c); pe, pt = emb(p)
    r = dict(child=c, parent=p, child_emb=ce, child_tie=ct, parent_emb=pe, parent_tie=pt)
    res.append(r)
    print(json.dumps(r), flush=True)
json.dump(res, open("results/fp32scan/headers.json", "w"), indent=1)
