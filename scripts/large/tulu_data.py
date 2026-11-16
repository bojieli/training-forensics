#!/usr/bin/env python3
"""Token occurrence counts of the published Tulu 3 data, tokenized the way open-instruct did.

SFT  (Llama-3.1-Tulu-3-405B-SFT): allenai/tulu-3-sft-mixture, rendered with the released
     tokenizer's chat template (apply_chat_template), max_seq_length 4096 (open-instruct
     Tulu 3 SFT recipe); variant `nocut` without truncation.
DPO  (Llama-3.1-Tulu-3-405B-DPO): allenai/llama-3.1-tulu-3-405b-preference-mixture,
     chosen and rejected conversations, max_seq_length 2048; variant `nocut`.
Writes results/large/tulu_<which>_counts.json: {variant: {token_id: count}}.
"""
import json, pathlib, sys
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOK = "allenai/Llama-3.1-Tulu-3-405B-SFT"
which = sys.argv[1]
tok = AutoTokenizer.from_pretrained(TOK)


def ids_of(msgs):
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)


def fn(batch):
    # count inside each batch so no token lists are kept in memory
    cut, nocut = Counter(), Counter()
    convs = batch["messages"] if which == "sft" else list(batch["chosen"]) + list(batch["rejected"])
    for m in convs:
        ids = ids_of(m)
        nocut.update(ids)
        cut.update(ids[:CUT])
    return {"c": [json.dumps([dict(cut), dict(nocut)])]}


if which == "sft":
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train"); CUT = 4096
else:
    ds = load_dataset("allenai/llama-3.1-tulu-3-405b-preference-mixture", split="train"); CUT = 2048
cut = CUT
n_rows = len(ds)
agg = ds.map(fn, batched=True, batch_size=2000, num_proc=6, remove_columns=ds.column_names)
res = {"cut": Counter(), "nocut": Counter()}
for s_ in agg["c"]:
    a, b = json.loads(s_)
    res["cut"].update({int(k): v for k, v in a.items()}); res["nocut"].update({int(k): v for k, v in b.items()})
ds = agg
out = {v: {str(k): n for k, n in c.items()} for v, c in res.items()}
out["meta"] = dict(dataset=ds.info.dataset_name if hasattr(ds, "info") else None, rows=n_rows, cut=cut, tokenizer=TOK)
json.dump(out, open(ROOT / "results" / "large" / f"tulu_{which}_counts.json", "w"))
print(which, "rows", n_rows, "distinct", len(res["cut"]), len(res["nocut"]))
