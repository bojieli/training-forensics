#!/usr/bin/env python3
"""Second, independent lineage check on other tensors: per tensor, share of
values bit-identical to each candidate (after upcasting it to fp32)."""
import json
import sys
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "reflection_check"))
from pins import PIN  # noqa: E402  pinned revisions for children and references

out, child = sys.argv[1:3]
crepo, _, crev = child.partition("@")
CANDS = {
    "llama-3.1-70b": "unsloth/Meta-Llama-3.1-70B",
    "llama-3.1-70b-instruct": "unsloth/Meta-Llama-3.1-70B-Instruct",
    "llama-3-70b": "NousResearch/Meta-Llama-3-70B",
    "llama-3-70b-instruct": "NousResearch/Meta-Llama-3-70B-Instruct",
}
NAMES = ["model.norm.weight"] + [f"model.layers.{i}.{n}" for i in (0, 40, 79) for n in
         ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.k_proj.weight",
          "self_attn.v_proj.weight")] + ["lm_head.weight"]
res = dict(child=child, tensors={})
for name in NAMES:
    try:
        W = remote.tensor(crepo, name, crev or PIN.get(crepo))
    except Exception as e:
        res["tensors"][name] = dict(error=str(e)[:100])
        continue
    W = W.float()
    rows = min(W.shape[0], 128256) if name == "lm_head.weight" else W.shape[0]
    W = W[:rows]
    r = dict(shape=list(W.shape), on_bf16_grid=float(((W.view(torch.int32) & 0xFFFF) == 0).float().mean()))
    for k, repo in CANDS.items():
        C = remote.tensor(repo, name, PIN.get(repo)).float()[:rows]
        eq = (W == C)
        r[k] = dict(values_identical=float(eq.float().mean()),
                    rows_identical=int(eq.all(-1).sum()) if W.dim() == 2 else int(eq.all())),
        r[k] = r[k][0]
        r[k]["rel_dist"] = float((W - C).norm() / C.norm())
    res["tensors"][name] = r
    print(name, json.dumps(r), flush=True)
json.dump(res, open(out, "w"), indent=1)
