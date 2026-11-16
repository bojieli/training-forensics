#!/usr/bin/env python3
"""Step 4: independent tokenization of the published LIMO set (GitHub limo.json and HF limo.jsonl), qwen chat
template written out by hand, no LLaMA-Factory code. Compares with the changed rows. -> out/tok.json"""
import json, collections, numpy as np
from tokenizers import Tokenizer
from common import D, save, gh
import pins

tk = Tokenizer.from_file(D + "limo_files/tokenizer.json")
tb = Tokenizer.from_file(D + "q32i_files/tokenizer.json")
va, vb = tk.get_vocab(with_added_tokens=True), tb.get_vocab(with_added_tokens=True)
ghd = json.load(open(gh("train/data/limo.json")))
hf = [json.loads(l) for l in open(D + "ds_limo_hf/limo.jsonl")]
render = lambda s, q, a: f"<|im_start|>system\n{s}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
ids = lambda t: tk.encode(t, add_special_tokens=False).ids
cnt = collections.Counter(); cut = collections.Counter(); lens = []
for g in ghd:
    x = ids(render(g["system"], g["instruction"] + ("\n" + g["input"] if g["input"] else ""), g["output"])); lens.append(len(x))
    cnt.update(set(x)); cut.update(set(x[:16384]))
hfset = set()
for h in hf:
    for k, v in h.items(): hfset.update(ids(str(v))); hfset.update(ids(" " + str(v))); hfset.update(ids("\n" + str(v)))
C = set(np.nonzero(np.load(D + "changed_mask.npy"))[0].tolist())
var = {}
for name, S in (("full", set(cnt)), ("cut16384", set(cut)), ("full+all HF fields", set(cnt) | hfset)):
    var[name] = {"distinct_tokens_in_published_set": len(S), "changed_rows": len(C), "in_set_and_changed": len(S & C),
                 "in_set_unchanged": len(S - C), "changed_absent_from_set": len(C - S)}
raw = "\n".join(json.dumps(h, ensure_ascii=False) for h in hf) + "\n".join(g["output"] + g["instruction"] for g in ghd)
is_cjk = lambda s: any("一" <= c <= "鿿" for c in s)
absent = sorted(C - set(cut))
out = {
    "source": {"github": f"{pins.GITHUB}@{pins.GITHUB_REV} train/data/limo.json", "hf": "{}@{} {}".format(*pins.DATASETS["limo_hf"])},
    "vocab_equal_limo_vs_base": va == vb, "vocab_size": len(va),
    "examples": {"github": len(ghd), "hf": len(hf),
                 "github_eq_hf_questions": sum(g["instruction"] == h["question"] for g, h in zip(ghd, hf)),
                 "github_eq_hf_solutions": sum(g["output"] == h["solution"] for g, h in zip(ghd, hf))},
    "lengths": {"max": max(lens), "n_over_16384": sum(l > 16384 for l in lens), "n_over_32768": sum(l > 32768 for l in lens),
                "total_full": sum(lens), "total_cut16384": int(np.minimum(lens, 16384).sum())},
    "variants": var,
    "absent_cut16384_cjk": sum(is_cjk(tk.decode([t])) for t in absent),
    "absent_cut16384_cjk_found_as_substring_in_raw_published_text": sum(1 for t in absent if is_cjk(tk.decode([t])) and tk.decode([t]) in raw),
    "cjk_chars_in_raw_published_text": sum("一" <= c <= "鿿" for c in raw),
}
json.dump({"doc_freq": cnt, "lens": lens}, open(D + "tok_limo.json", "w"))
print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)
save("tok", out)
