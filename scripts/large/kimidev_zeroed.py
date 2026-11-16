#!/usr/bin/env python3
"""Kimi-Dev-72B: which token vectors are exactly zero, and are they the same set KAT-Dev-72B-Exp zeroed?"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import json, sys, os, glob, pathlib, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "roster"))
import sweep
from transformers import AutoTokenizer
P = {p["key"]: p for p in json.load(open(ROOT / "results/large/pairs_large.json"))}
p = P["Qwen/Qwen2.5-72B@ -> moonshotai/Kimi-Dev-72B"]
B = sweep.fetch(p["child"]).float(); zd = set(torch.nonzero(B.norm(dim=1) == 0).flatten().tolist())
f = glob.glob(str(paths.cache("trainforensics") / "emb") + "/Kwaipilot__KAT-Dev-72B-Exp*embed*")[0]
K = torch.frombuffer(bytearray(open(f, "rb").read()), dtype=torch.bfloat16).view(152064, 8192).float()
zk = set(torch.nonzero(K.norm(dim=1) == 0).flatten().tolist())
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B")
V = len(tok); pad = sorted(i for i in zd if i >= V); real = sorted(i for i in zd if i < V)
reach = [i for i in real if tok.encode(tok.decode([i]), add_special_tokens=False) == [i]]
res = dict(kimi_dev_zero=len(zd), kat_dev_72b_exp_zero=len(zk), overlap=len(zk & zd), tokenizer_len=V,
           padding_slots=len(pad), real_tokens=len(real), real_reachable=len(reach),
           examples=tok.convert_ids_to_tokens(real[:30]), reachable_examples=tok.convert_ids_to_tokens(reach[:30]))
json.dump(res, open(ROOT / "results/large/kimidev_zeroed.json", "w"), ensure_ascii=False, indent=1)
print({k: v for k, v in res.items() if "examples" not in k}); print(res["reachable_examples"])
