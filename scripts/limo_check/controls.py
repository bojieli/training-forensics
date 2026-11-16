#!/usr/bin/env python3
"""Step 7: the same counting applied to LIMO-v2 and to s1-32B (whose published data matches). -> out/controls.json"""
import json, numpy as np
from tokenizers import Tokenizer
from common import D, save
import pins

tk = Tokenizer.from_file(D + "limo_files/tokenizer.json"); ids = lambda t: tk.encode(t, add_special_tokens=False).ids
out = {}
C2 = set(np.nonzero(np.load(D + "changed_mask_v2.npy"))[0].tolist())
v2 = [json.loads(l) for l in open(D + "ds_limo2_hf/limo-v2.jsonl")]
SYS = "Please reason step by step, and put your final answer within \\boxed{}."
render = lambda s, q, a: f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
ks = list(v2[0].keys()); qk = [k for k in ks if k in ("question", "instruction", "problem")][0]; ak = [k for k in ks if k in ("solution", "output", "response")][0]
S = set(); Sc = set()
for h in v2:
    x = ids(render(SYS, h[qk], h[ak])); S |= set(x); Sc |= set(x[:16384])
out["limo2"] = {"source": "{}@{} {}".format(*pins.DATASETS["limo2_hf"]), "examples": len(v2), "changed_rows": len(C2)}
for n, s in (("full", S), ("cut16384", Sc)):
    out["limo2"][n] = {"distinct_tokens": len(s), "in_set_unchanged": len(s - C2), "changed_absent": len(C2 - s)}
Cs = set(np.nonzero(np.load(D + "changed_mask_s1.npy"))[0].tolist())
from datasets import load_dataset
repo, rev, _ = pins.DATASETS["s1k_tok"]
ds = load_dataset(repo, split="train", revision=rev, cache_dir=D + "hf_datasets")
T = set(); Tl = set()
for t in ds["text"]:
    x = ids(t); T |= set(x); Tl |= set(x[:-1])
out["s1"] = {"source": f"{repo}@{rev}", "changed_rows": len(Cs), "distinct_tokens": len(T), "excluding_last_position": len(Tl),
             "in_set_unchanged": len(Tl - Cs), "changed_absent": len(Cs - T)}
print(json.dumps(out, indent=1), flush=True)
save("controls", out)
