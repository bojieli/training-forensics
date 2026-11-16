#!/usr/bin/env python3
"""Token occurrence counts of a published fine-tuning set, tokenized the way its
training code did (plus variants). Writes results/published/data_<name>.npz-like JSON.

Formats
  s1 / s1.1   : the `text` column of simplescaling/s1K(-1.1)_tokenized, tokenized whole,
                truncated at 32,768 (train/sft.py: block_size=32768, trl SFTTrainer).
  llamafactory: LLaMA-Factory `qwen` template; every formatted slot is encoded on its own
                (system, user, assistant), then source/target truncated with infer_seqlen.
Variants
  whole       : the same conversation rendered as one string and tokenized whole
  nocut       : no truncation
  defsys      : LLaMA-Factory's default system prompt instead of the dataset's
"""
from __future__ import annotations

import ast
import json
import pathlib
import sys
from collections import Counter

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "published"


def infer_seqlen(source_len, target_len, cutoff_len):
    # LLaMA-Factory src/llamafactory/data/processor/processor_utils.py
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def conv_of(row):
    c = row["conversations"]
    if isinstance(c, str):
        c = ast.literal_eval(c)
    return c


def lf_ids(tok, system, turns, cutoff, whole=False):
    """LLaMA-Factory qwen template, multi-turn: list of (source_ids, target_ids)."""
    sys_s = f"<|im_start|>system\n{system}<|im_end|>\n" if system is not None else ""
    pairs = []
    msgs = [(t.get("from") or t.get("role"), t.get("value") or t.get("content")) for t in turns]
    i = 0
    first = True
    while i + 1 < len(msgs):
        (ru, u), (ra, a) = msgs[i], msgs[i + 1]
        src = f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"
        tgt = f"{a}<|im_end|>\n"
        if whole:
            s_ids = tok.encode((sys_s if first else "") + src, add_special_tokens=False)
        else:
            s_ids = (tok.encode(sys_s, add_special_tokens=False) if (first and sys_s) else []) + \
                tok.encode(src, add_special_tokens=False)
        t_ids = tok.encode(tgt, add_special_tokens=False)
        pairs.append((s_ids, t_ids))
        first = False
        i += 2
    ids = []
    total = 0
    for s_ids, t_ids in pairs:
        if cutoff and total >= cutoff:
            break
        if cutoff:
            sl, tl = infer_seqlen(len(s_ids), len(t_ids), cutoff - total)
            s_ids, t_ids = s_ids[:sl], t_ids[:tl]
        ids += s_ids + t_ids
        total += len(s_ids) + len(t_ids)
    return ids


def count(name, base_tok, rows_fn, n_rows):
    c = Counter()
    for k in range(n_rows):
        c.update(rows_fn(k))
    return c


def main():
    which = sys.argv[1]
    tok_repo = sys.argv[2]
    tok = AutoTokenizer.from_pretrained(tok_repo)
    V = len(tok)
    out = {}
    if which in ("s1", "s1.1"):
        ds = load_dataset("simplescaling/s1K_tokenized" if which == "s1" else "simplescaling/s1K-1.1_tokenized",
                          split="train")
        texts = ds["text"]
        variants = {"main": 32768, "nocut": None}
        for v, cut in variants.items():
            c = Counter()
            for t in texts:
                ids = tok(t, add_special_tokens=True)["input_ids"]
                c.update(ids[:cut] if cut else ids)
            out[v] = c
        # trl versions that append EOS
        c = Counter(out["main"])
        c[tok.convert_tokens_to_ids("<|im_end|>")] += 0
        c[tok.eos_token_id] += len(texts)
        out["main+eos"] = c
    else:
        try:  # loaders for LIMO / LIMO-v2 (extra_sets.py)
            from extra_sets import TID_ROWS
        except ImportError:
            TID_ROWS = {}
        if which in TID_ROWS:
            rows, cut = TID_ROWS[which](ROOT), 16384
        else:
            repo = {"skyt1": "NovaSky-AI/Sky-T1_data_17k", "stratos": "bespokelabs/Bespoke-Stratos-17k",
                    "ot114k": "open-thoughts/OpenThoughts-114k"}[which]
            ds = load_dataset(repo, split="train")
            rows = ds
            cut = 16384
        n = len(rows)

        def run(variant):
            c = Counter()
            for k in range(n):
                r = rows[k]
                system = r.get("system")
                if variant == "defsys":
                    system = "You are a helpful assistant."
                ids = lf_ids(tok, system, conv_of(r), None if variant == "nocut" else cut, whole=(variant == "whole"))
                c.update(ids)
            return c
        for v in ("main", "whole", "nocut", "defsys"):
            if which == "ot114k" and v in ("whole", "defsys"):
                continue
            out[v] = run(v)
            print(which, v, len(out[v]), flush=True)
    res = {v: {str(k): int(n) for k, n in c.items()} for v, c in out.items()}
    json.dump(dict(dataset=which, tokenizer=tok_repo, vocab=V, counts=res),
              open(OUT / f"data_{which}.json", "w"))
    for v, c in out.items():
        print(which, v, "distinct tokens", len(c), "total", sum(c.values()))


if __name__ == "__main__":
    main()
