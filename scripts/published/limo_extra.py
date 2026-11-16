"""LIMO: do the updated tokens absent from the published 817 examples come from somewhere identifiable?
Checks (1) the HF dataset copy vs the GitHub copy, (2) overlap of the extra tokens with other public reasoning sets."""
import json, pathlib, sys
from collections import Counter
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote
OUT = ROOT / "results" / "published"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")
remote.CACHE = pathlib.Path.home() / ".cache" / "trainforensics" / "emb"
W0 = remote.tensor("Qwen/Qwen2.5-32B-Instruct", "model.embed_tokens.weight")
remote.CACHE = pathlib.Path.home() / ".cache" / "tf_published"
W1 = remote.tensor("GAIR/LIMO", "model.embed_tokens.weight")
changed = set(np.nonzero((W0 != W1).any(1).numpy())[0].tolist())
gh = set(int(k) for k in json.load(open(OUT / "data_limo.json"))["counts"]["main"])
hf = load_dataset("GAIR/LIMO", split="train")
hfset = set()
for r in hf:
    hfset.update(tok.encode(r["question"] + "\n" + r["solution"] + "\n" + str(r["answer"]), add_special_tokens=False))
extra = changed - gh
res = dict(changed=len(changed), github_set=len(gh), hf_set=len(hfset), extra_vs_github=len(extra),
           extra_vs_hf=len(changed - hfset - gh), hf_minus_github=len(hfset - gh))
others = {}
for name in ("s1", "s1.1", "skyt1", "stratos", "ot114k"):
    p = OUT / f"data_{name}.json"
    if p.exists():
        s = set(int(k) for k in json.load(open(p))["counts"]["main"])
        others[name] = dict(size=len(s), covers_extra=len(extra & s), extra_not_covered=len(extra - s),
                            changed_minus_set=len(changed - s), set_minus_changed=len(s - changed))
res["others"] = others
# categories of the extra tokens
def cat(t):
    s = tok.convert_tokens_to_string([t])
    if any("一" <= ch <= "鿿" for ch in s): return "CJK"
    if any(ord(ch) > 127 for ch in s): return "other non-ASCII"
    if any(k in s for k in ("import", "def", "class", "return", "public", "{", "}", "();", "::", "self", "#include")): return "code-like"
    return "ASCII other"
res["extra_categories"] = Counter(cat(tok.convert_ids_to_tokens(i)) for i in extra)
json.dump(res, open(OUT / "limo_extra.json", "w"), indent=1)
print(json.dumps(res, indent=1))
