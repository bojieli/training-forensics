#!/usr/bin/env python3
"""Sanity control: the published-data pipeline (tokenize the data independently, find updated token
vectors with the same decay fit, compare) applied to our own controlled runs, where the trained
documents are known exactly. Expected: 0 tokens updated but absent from the data."""
import json, pathlib, sys
import numpy as np, torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "published"))
from compare import frac_changed, fit_P  # noqa: E402

OUT = ROOT / "results" / "published"


def run(name, release):
    d = ROOT / "runs" / "ctl" / name
    t = json.load(open(d / "truth.json"))
    a = t["args"]
    tok = AutoTokenizer.from_pretrained(a["model"])
    ds = load_dataset("json", data_files=str(ROOT / a["data_jsonl"]), split="train") if a.get("data_jsonl") else \
        load_dataset("tatsu-lab/alpaca", split="train")

    def fmt(r):
        s = f"### Instruction:\n{r['instruction']}\n"
        if r["input"]:
            s += f"### Input:\n{r['input']}\n"
        return s + f"### Response:\n{r['output']}"
    present, present_nf = set(), set()
    for i in t["trained"]:
        ids = tok(fmt(ds[i]), truncation=True, max_length=a["max_len"])["input_ids"]
        present.update(ids)
        present_nf.update(ids[:-1])
    E = load_file(str(d / "emb.safetensors"))
    W0 = E["W0"]
    W1 = E[{"fp32": "W1", "bf16": "W1_bf16", "fp16": "W1_fp16"}[release]]
    W0c = W0
    f = frac_changed(W0c, W1, 1.0)
    P = 1.0
    if (f == 0).sum() < 0.001 * len(f):
        P = fit_P(W0c, W1, f)
        f = frac_changed(W0c, W1, P)
    changed = set(np.nonzero(f > 0)[0].tolist())
    real = set(range(len(tok)))
    res = dict(run=name, release=release, wd=a["wd"], P=P, present=len(present), changed=len(changed),
               changed_not_present=len((changed - present) & real), changed_not_present_any=len(changed - present),
               present_not_changed=len(present - changed),
               nonfinal_present=len(present_nf), nonfinal_changed_not_present=len((changed - present_nf) & real),
               nonfinal_present_not_changed=len(present_nf - changed))
    print(res, flush=True)
    return res


if __name__ == "__main__":
    out = [run(n, r) for n, r in (("r01_lin", "fp32"), ("r01_lin", "bf16"), ("s06_bf16", "bf16"),
                                  ("r07_cos_wd", "bf16"), ("r12_cosmin_wd", "bf16"), ("r07_cos_wd", "fp16"))]
    json.dump(out, open(OUT / "control.json", "w"), indent=1)
