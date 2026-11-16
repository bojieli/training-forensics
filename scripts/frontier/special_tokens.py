#!/usr/bin/env python3
"""Claims 1-4 and 6: per-token change of named special tokens and reserved placeholder slots
for DeepSeek, Qwen3/3.5 and Llama-4 Scout; per-vector changed-fraction histograms.

For each base -> release comparison:
  frac[i] = fraction of the numbers in token vector i that differ from round(P * base[i]),
  with P the shrink factor fitted by the sweep (weight decay). 0 = bit-identical after
  removing decay; <=1% = barely changed (can be rounding of a tiny update in 16-bit).
Writes results/frontier/special_tokens.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import requests
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import OUT, DT, changed_fraction, is_placeholder, norms, pair, records, state  # noqa: E402

DS = "deepseek-ai/"
PAIRS = [
    (DS + "DeepSeek-V3-Base", DS + "DeepSeek-V3"),
    (DS + "DeepSeek-V3-Base", DS + "DeepSeek-V3-0324"),
    (DS + "DeepSeek-V3-Base", DS + "DeepSeek-R1"),
    (DS + "DeepSeek-V3-Base", DS + "DeepSeek-R1-Zero"),
    (DS + "DeepSeek-V3-Base", "nex-agi/DeepSeek-V3.1-Nex-N1"),
    (DS + "DeepSeek-V3.2-Exp-Base", DS + "DeepSeek-V3.2"),
    (DS + "DeepSeek-V4-Flash-Base", DS + "DeepSeek-V4-Flash"),
    (DS + "DeepSeek-V4-Flash", DS + "DeepSeek-V4-Flash-0731"),
    (DS + "DeepSeek-V4-Pro-Base", DS + "DeepSeek-V4-Pro"),
    (DS + "DeepSeek-V4-Pro", DS + "DeepSeek-V4-Pro-0813"),
    (DS + "DeepSeek-V4-Pro-Base", "nex-agi/Nex-N2.5-Max"),
    (DS + "DeepSeek-V4.1-Flash", DS + "DeepSeek-V4.1-Flash-1017"),
    ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"),
    ("Qwen/Qwen3-14B-Base", "Qwen/Qwen3-14B"),
    ("Qwen/Qwen3-30B-A3B-Base", "Qwen/Qwen3-30B-A3B"),
    ("Qwen/Qwen3.5-35B-A3B-Base", "Qwen/Qwen3.5-35B-A3B"),
    ("unsloth/Llama-4-Scout-17B-16E", "unsloth/Llama-4-Scout-17B-16E-Instruct"),
]
EDGES = [0, 1e-9, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0001]


def chat_template(repo):
    for fn in ("tokenizer_config.json", "chat_template.jinja", "chat_template.json"):
        try:
            r = requests.get(f"https://huggingface.co/{repo}/resolve/main/{fn}", timeout=60)
            if r.status_code == 200:
                if fn.endswith(".json"):
                    t = r.json().get("chat_template")
                    if isinstance(t, list):
                        t = " ".join(x.get("template", "") for x in t)
                    if t:
                        return t
                else:
                    return r.text
        except Exception:
            pass
    return ""


def ranges(ix):
    ix = sorted(int(i) for i in ix)
    out, s = [], None
    for i in ix:
        if s is None:
            s = p = i
        elif i == p + 1:
            p = i
        else:
            out.append([s, p]); s = p = i
    if s is not None:
        out.append([s, p])
    return out


def main():
    recs = records()
    only = sys.argv[1:]
    res = {}
    p = OUT / "special_tokens.json"
    if p.exists():
        res = json.load(open(p))
    for par, ch in PAIRS:
        key = f"{par} -> {ch}"
        if only and not any(o in key for o in only):
            continue
        try:
            W0, W1, r = pair(par, ch, recs)
        except Exception as e:
            res[key] = dict(error=str(e)); print(key, "ERR", e, flush=True); continue
        dt = DT[r["dtype_child"]]
        P = r["embed"]["P"]
        d = W0.shape[1]
        frac = changed_fraction(W0, W1, P, dt)
        n0, n1 = norms(W0), norms(W1)
        tmpl = chat_template(ch)
        sp = r.get("special", [])
        spid = {t["id"] for t in sp}
        ordinary = np.array([i not in spid for i in range(W0.shape[0])])
        named, ph = [], []
        for t in sp:
            i = t["id"]
            if i >= W0.shape[0]:
                continue
            row = dict(id=i, content=t["content"], frac=float(frac[i]), state=state(frac[i], d),
                       base_norm=float(n0[i]), release_norm=float(n1[i]),
                       in_chat_template=bool(t["content"]) and t["content"] in tmpl)
            (ph if is_placeholder(t["content"]) else named).append(row)
        hist = np.histogram(frac, bins=EDGES)[0].tolist()
        st = {s: int(sum(state(f, d) == s for f in frac)) for s in ("bit-identical", "barely changed", "partly changed", "updated")}
        ph_upd = [x["id"] for x in ph if x["state"] in ("updated", "partly changed")]
        bi_ord = ordinary & (frac == 0)
        res[key] = dict(
            parent=par, child=ch, P=P, fit_reproduced=r["embed"]["P_exact_frac"], dim=d, vocab=int(W0.shape[0]),
            states=st, hist_edges=EDGES, hist=hist,
            frac_quantiles={q: float(np.quantile(frac, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            base_norm_ordinary_median=float(np.median(n0[ordinary])),
            base_norm_bitidentical_ordinary_median=(float(np.median(n0[bi_ord])) if bi_ord.any() else None),
            named=named,
            placeholders=dict(n=len(ph), states={s: sum(x["state"] == s for x in ph) for s in st},
                              updated_id_ranges=ranges(ph_upd),
                              base_norm_median=(float(np.median([x["base_norm"] for x in ph])) if ph else None),
                              release_norm_median=(float(np.median([x["release_norm"] for x in ph])) if ph else None)),
            chat_template_found=bool(tmpl))
        print(key, st, "named", len(named), "ph", res[key]["placeholders"]["states"], flush=True)
        json.dump(res, open(p, "w"), indent=1)
        del W0, W1


if __name__ == "__main__":
    main()
