#!/usr/bin/env python3
"""List candidate checkpoint series: tie flag, embedding tensor name/dtype/shape, branch count."""
import json, re, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from huggingface_hub import HfApi, hf_hub_url
import requests
from trainforensics import remote
REPOS = sys.argv[1:]
api = HfApi()
EMBN = ("embed_tokens.weight", "wte.weight", "embed_in.weight", "word_embeddings.weight", "tok_embeddings.weight")
for r in REPOS:
    try:
        br = sorted(b.name for b in api.list_repo_refs(r).branches)
        rev = [b for b in br if b != "main"][-1] if len(br) > 1 else "main"
        cfg = requests.get(hf_hub_url(r, "config.json", revision="main"), timeout=60).json()
        tie = cfg.get("tie_word_embeddings", cfg.get("weight_tying"))
        idx = remote.index(r, "main")
        en = [k for k in idx if k.endswith(EMBN)]
        info = remote.tensor_info(r, en[0], "main") if en else None
        print(json.dumps(dict(repo=r, branches=len(br), tie=tie, emb=en, info=info, vocab=cfg.get("vocab_size"),
              wd=cfg.get("weight_decay"), ex=br[:2] + br[-2:])), flush=True)
    except Exception as e:
        print(json.dumps(dict(repo=r, error=repr(e)[:200])), flush=True)
