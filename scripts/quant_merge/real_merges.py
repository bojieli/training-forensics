#!/usr/bin/env python3
"""Published mergekit merges on roster parents: read the merge from the embedding rows,
then check it against the merge config the author published.

Only embedding tensors are fetched (HTTP range reads). Gated meta-llama repos are
read through their public mirrors (unsloth / NousResearch). Writes
results/quant_merge/merge_real.json.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

import requests
import torch
import yaml
from huggingface_hub import HfApi, hf_hub_url

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from trainforensics import remote  # noqa: E402
from mergesig import analyse  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "quant_merge" / "merge_real.json"
KEY = "model.embed_tokens.weight"
MIRROR = {
    "meta-llama/Meta-Llama-3-8B": "unsloth/llama-3-8b",
    "meta-llama/Meta-Llama-3-8B-Instruct": "unsloth/llama-3-8b-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "unsloth/Llama-3.1-8B-Instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B": "unsloth/Meta-Llama-3.1-8B",
}
L3I, L3B, L31I = "unsloth/llama-3-8b-Instruct", "unsloth/llama-3-8b", "unsloth/Meta-Llama-3.1-8B-Instruct"
# merge repo -> reference base O for the row diff (declared base_model if any, else the parents' common ancestor)
MERGES = {
    "grimjim/Llama-3-Instruct-8B-SimPO-SPPO-Iter3-merge": L3I,
    "grimjim/Llama-3-Instruct-8B-SPPO-Iter3-SimPO-merge": L3I,
    "anakin87/Llama-3-8b-ita-slerp": L3I,
    "anakin87/Llama-3-8b-ita-ties": None,
    "anakin87/Llama-3-8b-ita-ties-pro": None,
    "mlabonne/ChimeraLlama-3-8B-v3": None,
    "MergekitCloud/mergekit-150": None,
    "MergekitCloud/mergekit-154": None,
    "voidful/Llama-3.1-TAIDE-R1-8B-Chat": None,
    "shanchen/llama3-8B-slerp-med-chinese": L3B,
    "mlabonne/ChimeraLlama-3-8B-v2": None,
    "Kukedlc/NeuralLLaMa-3-8b-DT-v0.1": None,
}
# natural (non-merged) fine-tunes, same families: the null for the blind detector
NATURAL = [("unsloth/llama-3-8b-Instruct", L3B), ("unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B"),
           ("princeton-nlp/Llama-3-Instruct-8B-SimPO", L3I), ("UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3", L3I),
           ("swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA", L3I), ("DeepMount00/Llama-3-8b-Ita", L3I),
           ("NousResearch/Hermes-3-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B"),
           ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "unsloth/Meta-Llama-3.1-8B"),
           ("Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B"), ("allenai/Llama-3.1-Tulu-3-8B-SFT", "unsloth/Meta-Llama-3.1-8B"),
           ("allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B-SFT"),
           ("Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-14B")]


def mirror(r):
    return MIRROR.get(r, r)


def config(repo):
    api = HfApi()
    fs = api.list_repo_files(repo)
    h = remote._headers()
    if "mergekit_config.yml" in fs:
        t = requests.get(hf_hub_url(repo, "mergekit_config.yml"), headers=h, timeout=60).text
    else:
        md = requests.get(hf_hub_url(repo, "README.md"), headers=h, timeout=60).text
        m = [x for x in re.findall(r"```ya?ml\n(.*?)```", md, re.S) if "merge_method" in x]
        t = m[0] if m else ""
    return yaml.safe_load(t) if t else None


def parents_of(cfg):
    ps = []
    for m in cfg.get("models", []) or []:
        ps.append((m["model"], (m.get("parameters") or {})))
    for s in cfg.get("slices", []) or []:
        for src in s.get("sources", []):
            ps.append((src["model"], {}))
    seen, out = set(), []
    for p, par in ps:
        if p not in seen:
            seen.add(p)
            out.append((p, par))
    return out


def emb(repo):
    return remote.tensor(mirror(repo), KEY)


def main():
    res = []
    for repo, O_name in MERGES.items():
        rec = dict(merge=repo)
        try:
            cfg = config(repo)
            rec["config"] = dict(method=cfg.get("merge_method"), base_model=cfg.get("base_model"),
                                 parameters=cfg.get("parameters"), dtype=cfg.get("dtype"),
                                 tokenizer=cfg.get("tokenizer_source") or cfg.get("tokenizer"),
                                 models=[dict(model=p, parameters=par) for p, par in parents_of(cfg)])
            O_name = O_name or cfg.get("base_model")
            rec["reference_base"] = mirror(O_name)
            M = emb(repo)
            O = emb(O_name)
            parents = {}
            for p, _ in parents_of(cfg):
                if mirror(p) == mirror(O_name):
                    continue
                try:
                    parents[p] = emb(p)
                except Exception as e:
                    rec.setdefault("parent_errors", {})[p] = f"{type(e).__name__}: {str(e)[:120]}"
            rec["shapes"] = dict(merge=list(M.shape), base=list(O.shape), **{p: list(v.shape) for p, v in parents.items()})
            rec.update(analyse(M, O, parents or None))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(json.dumps(rec)[:2500], flush=True)
        res.append(rec)
        json.dump(dict(merges=res), open(OUT, "w"), indent=1)
    nat = []
    for c, b in NATURAL:
        try:
            nat.append(dict(child=c, base=b, **analyse(emb(c), emb(b), None)))
        except Exception as e:
            nat.append(dict(child=c, base=b, error=f"{type(e).__name__}: {str(e)[:200]}"))
        print(json.dumps(nat[-1])[:800], flush=True)
    json.dump(dict(merges=res, natural=nat), open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
