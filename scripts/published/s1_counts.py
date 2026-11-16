#!/usr/bin/env python3
"""s1 / s1.1: the `text` column tokenized whole (trl SFTTrainer, add_special_tokens) and truncated at the
block size; counts over all positions and over all but the last position of each sequence."""
import json, pathlib, sys
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "published"
which = sys.argv[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
ds = load_dataset("simplescaling/s1K_tokenized" if which == "s1" else "simplescaling/s1K-1.1_tokenized", split="train")
enc = [tok(t, add_special_tokens=True)["input_ids"] for t in ds["text"]]
counts = {}
for cut in (32768, 20000):
    a, nf = Counter(), Counter()
    for ids in enc:
        ids = ids[:cut]
        a.update(ids); nf.update(ids[:-1])
    counts[f"trl_cut{cut}"] = {str(k): v for k, v in a.items()}
    counts[f"trl_cut{cut}_nonfinal"] = {str(k): v for k, v in nf.items()}
# documented block sizes: s1 32768 (train/sft.sh), s1.1 20000 (README)
counts["main"] = counts["trl_cut32768_nonfinal" if which == "s1" else "trl_cut20000_nonfinal"]
json.dump(dict(dataset=which, tokenizer="Qwen/Qwen2.5-32B-Instruct", vocab=len(tok),
               encoder="trl SFTTrainer text field, add_special_tokens=True", counts=counts), open(OUT / f"data_{which}.json", "w"))
for k, c in counts.items():
    print(which, k, len(c))
