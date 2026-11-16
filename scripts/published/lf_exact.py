#!/usr/bin/env python3
"""Tokenize a published SFT set with LLaMA-Factory's own encoder (llamafactory 0.9.2.dev0,
template.py unchanged since 2025-02-05), template `qwen`, exactly as
`_encode_supervised_example` builds input_ids.

Variants: system prompt from the dataset vs LLaMA-Factory's default (used when the dataset's
system column is not mapped in dataset_info.json); cutoff 16384 vs none. For each variant two
counts are kept: all positions, and all but the last position of each sequence (the last input
token of a sequence gets no gradient: nothing attends to it and it has no next-token target).

Usage: PYTHONPATH=<llamafactory-0.9.2.dev0>/src python3 lf_exact.py <dataset> <tokenizer>
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys
from collections import Counter
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "published"
TOKREPO = None
_tpl = _tok = None


def _init(tokrepo):
    global _tpl, _tok
    from transformers import AutoTokenizer
    from llamafactory.data.template import TEMPLATES
    _tok = AutoTokenizer.from_pretrained(tokrepo)
    _tpl = TEMPLATES["qwen"]


def _encode(args):
    from llamafactory.data.processors.supervised import _encode_supervised_example
    rows, variants = args
    out = {v: [Counter(), Counter()] for v in variants}
    for system, msgs in rows:
        prompt, response = msgs[:-1], msgs[-1:]
        for v in variants:
            sysv, cut = v.split("|")
            s = system if sysv == "datasys" else None
            ids, _ = _encode_supervised_example(prompt, response, s, None, [], [], [], _tpl, _tok, None,
                                                cutoff_len=10 ** 9 if cut == "nocut" else int(cut),
                                                train_on_prompt=False, mask_history=False)
            out[v][0].update(ids)
            out[v][1].update(ids[:-1])
    return out


def norm_msgs(turns):
    role = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant"}
    return [{"role": role[t.get("from") or t.get("role")], "content": t.get("value") or t.get("content")}
            for t in turns]


def load_rows(which):
    from datasets import load_dataset
    try:  # optional local loaders for datasets not covered here
        from private_sets import LF_ROWS
        if which in LF_ROWS:
            return LF_ROWS[which](OUT)
    except ImportError:
        pass
    repo = {"skyt1": "NovaSky-AI/Sky-T1_data_17k", "stratos": "bespokelabs/Bespoke-Stratos-17k",
            "ot114k": "open-thoughts/OpenThoughts-114k"}[which]
    ds = load_dataset(repo, split="train")
    rows = []
    for r in ds:
        c = r["conversations"]
        if isinstance(c, str):
            c = ast.literal_eval(c)
        rows.append((r.get("system"), norm_msgs(c)))
    return rows


def main():
    which, tokrepo = sys.argv[1], sys.argv[2]
    rows = load_rows(which)
    variants = ["datasys|16384", "default|16384", "datasys|nocut"]
    if which == "ot114k":
        variants = ["datasys|16384", "default|16384"]
    chunks = [(rows[i:i + 200], variants) for i in range(0, len(rows), 200)]
    tot = {v: [Counter(), Counter()] for v in variants}
    with Pool(12, initializer=_init, initargs=(tokrepo,)) as pool:
        for part in pool.imap_unordered(_encode, chunks):
            for v in variants:
                tot[v][0].update(part[v][0])
                tot[v][1].update(part[v][1])
    counts = {}
    for v in variants:
        name = v.replace("|", "_")
        counts[f"lf_{name}"] = {str(k): int(n) for k, n in tot[v][0].items()}
        counts[f"lf_{name}_nonfinal"] = {str(k): int(n) for k, n in tot[v][1].items()}
    counts["main"] = counts["lf_datasys_16384_nonfinal"]
    # keep earlier approximate variants if present
    p = OUT / f"data_{which}.json"
    if p.exists():
        old = json.load(open(p))["counts"]
        for k, v in old.items():
            if k != "main":
                counts.setdefault(f"approx_{k}", v)
    from transformers import AutoTokenizer
    json.dump(dict(dataset=which, tokenizer=tokrepo, vocab=len(AutoTokenizer.from_pretrained(tokrepo)),
                   encoder="llamafactory 0.9.2.dev0 template qwen", counts=counts),
              open(p, "w"))
    for k, c in counts.items():
        print(which, k, "distinct", len(c), flush=True)


if __name__ == "__main__":
    main()
