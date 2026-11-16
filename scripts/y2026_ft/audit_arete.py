#!/usr/bin/env python3
"""Published-data audit: NousResearch/Arete-27B vs Qwen3.8-27B against NousResearch/arete-data-pairs.

Updated token vectors (results/y2026_ft/pairs/arete27b.json, mask ~/.cache/tf_y2026/masks/arete27b.npz)
vs the tokens of the published SFT set (sft_accepted: 29,236 chat examples; reasoning: Inkling traces
for the same rows), tokenized with the release's own chat template. Variants:
  msgs        chat template, assistant answer only
  msgs+think  chat template with the reasoning trace as reasoning_content
  pieces      every message content encoded on its own (no template), plus the reasoning text
  union       any of the above
Control: the same comparison against the 5,633 teacher-rejected and the Qwen-answer texts in
`paired` (prompts not in the SFT set) is reported as tokens unique to them.
Writes results/y2026_ft/audit_arete27b.json
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import glob
import json
import pathlib
import sys
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = paths.cache("tf_y2026") / "data" / "arete"
M = np.load(paths.cache("tf_y2026") / "masks" / "arete27b.npz")
nd, d = M["ndiff"], int(M["d"])
thr = max(2, int(0.01 * d))
V = len(nd)
upd = nd > thr
barely = (nd > 0) & (nd <= thr)
same = nd == 0
tok = AutoTokenizer.from_pretrained("NousResearch/Arete-27B")
pair = json.load(open(ROOT / "results" / "y2026_ft" / "pairs" / "arete27b.json"))
special_ids = {s["id"] for s in pair.get("special_tokens", [])}

rows = []
for f in sorted(glob.glob(str(D / "sft_accepted" / "*.jsonl"))):
    for line in open(f):
        rows.append(json.loads(line))
reason = {}
for f in sorted(glob.glob(str(D / "reasoning" / "*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        reason[r["id"]] = r.get("reasoning") or ""
print("rows", len(rows), "reasoning", len(reason), flush=True)


def rid(r):
    m = r.get("metadata") or {}
    return m.get("id")


def counts(texts_ids):
    c = np.zeros(V, np.int64)
    for ids in texts_ids:
        ids = np.asarray([i for i in ids if i < V], dtype=np.int64)
        c += np.bincount(ids, minlength=V)[:V]
    return c


var = {}
B = 512
# 1. chat template, answer only
docs = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows]
enc = []
for k in range(0, len(docs), B):
    enc += tok(docs[k:k + B], add_special_tokens=False)["input_ids"]
var["msgs"] = counts(enc)
# 2. with reasoning_content
docs2 = []
miss = 0
for r in rows:
    msgs = [dict(m) for m in r["messages"]]
    rs = reason.get(rid(r))
    if rs is None:
        miss += 1
    else:
        msgs[-1]["reasoning_content"] = rs
    docs2.append(tok.apply_chat_template(msgs, tokenize=False))
enc2 = []
for k in range(0, len(docs2), B):
    enc2 += tok(docs2[k:k + B], add_special_tokens=False)["input_ids"]
var["msgs+think"] = counts(enc2)
lens2 = np.array([len(x) for x in enc2])
# 3. pieces
pieces = []
for r in rows:
    for m in r["messages"]:
        pieces.append(m["content"])
    pieces.append(reason.get(rid(r), ""))
enc3 = []
for k in range(0, len(pieces), B):
    enc3 += tok(pieces[k:k + B], add_special_tokens=False)["input_ids"]
var["pieces"] = counts(enc3)
var["union"] = var["msgs"] + var["msgs+think"] + var["pieces"]


def dec(i):
    return tok.decode([int(i)])


out = dict(child="NousResearch/Arete-27B", base="Qwen/Qwen3.8-27B", data="NousResearch/arete-data-pairs (sft_accepted + reasoning)",
           rows=len(rows), rows_without_reasoning=miss, tokens_per_row_msgs_think=dict(median=float(np.median(lens2)), max=int(lens2.max()),
                                                                                            over_65536=int((lens2 > 65536).sum())),
           vector_states=dict(updated=int(upd.sum()), barely=int(barely.sum()), bit_identical=int(same.sum())), variants={})
for k, c in var.items():
    present = c > 0
    pu = present & upd
    pn = present & ~upd
    up_abs = upd & ~present
    up_abs_ns = np.array([i for i in np.nonzero(up_abs)[0] if i not in special_ids])
    pn_ns = np.array([i for i in np.nonzero(pn)[0] if i not in special_ids])
    occ = c[pn_ns] if len(pn_ns) else np.array([])
    rec = dict(present=int(present.sum()), present_and_updated=int(pu.sum()), present_not_updated=int(pn.sum()),
               present_not_updated_barely=int((present & barely).sum()),
               present_not_updated_occ_hist={str(x): int((occ == x).sum()) for x in (1, 2, 3, 4, 5)} | {">5": int((occ > 5).sum())},
               updated_not_present=int(up_abs.sum()), updated_not_present_nonspecial=int(len(up_abs_ns)),
               updated_not_present_examples=[[int(i), dec(i), round(float(nd[i] / d), 3)] for i in up_abs_ns[:40]],
               present_not_updated_examples=[[int(i), dec(i), int(c[i])] for i in pn_ns[np.argsort(-occ)][:40]] if len(pn_ns) else [])
    out["variants"][k] = rec
    print(k, {x: rec[x] for x in ("present", "present_and_updated", "present_not_updated", "updated_not_present",
                                  "updated_not_present_nonspecial")}, flush=True)
json.dump(out, open(ROOT / "results" / "y2026_ft" / "audit_arete27b.json", "w"), indent=1, ensure_ascii=False)
