#!/usr/bin/env python3
"""Arete audit, step 2.
 (a) special tokens: which appear in the chat-formatted SFT rows, and their states;
 (b) rows that hold a token seen once in the whole SFT set whose vector is bit-identical to the
     base (not 'barely'): candidate rows not used in training. The card's 601 steps x 48 =
     28,848 examples per epoch vs 29,236 published rows leaves 388 unaccounted for;
 (c) the 240 updated-but-absent tokens: do they occur in the other published texts
     (paired: teacher-rejected answers, Qwen starting-checkpoint answers, excluded prompts)?
Writes results/y2026_ft/audit_arete27b_step2.json
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import glob
import json
import pathlib

import numpy as np
from transformers import AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[2]
D = paths.cache("tf_y2026") / "data" / "arete"
M = np.load(paths.cache("tf_y2026") / "masks" / "arete27b.npz")
nd, d = M["ndiff"], int(M["d"])
thr = max(2, int(0.01 * d))
V = len(nd)
tok = AutoTokenizer.from_pretrained("NousResearch/Arete-27B")
pair = json.load(open(ROOT / "results" / "y2026_ft" / "pairs" / "arete27b.json"))
spec = {s["id"]: s for s in pair["special_tokens"]}
rows = []
for f in sorted(glob.glob(str(D / "sft_accepted" / "*.jsonl"))):
    rows += [json.loads(l) for l in open(f)]
enc = []
for k in range(0, len(rows), 512):
    docs = [tok.apply_chat_template(r["messages"], tokenize=False) for r in rows[k:k + 512]]
    enc += tok(docs, add_special_tokens=False)["input_ids"]
cnt = np.zeros(V, np.int64)
for ids in enc:
    cnt += np.bincount(np.asarray(ids), minlength=V)[:V]
out = {}
out["special_in_data"] = [dict(id=i, content=spec[i]["content"], rows_with_it=int(sum(1 for ids in enc if i in set(ids))),
                               state=spec[i]["state"]) for i in spec if cnt[i] > 0]
out["example_rendering_head"] = tok.apply_chat_template(rows[0]["messages"], tokenize=False)[:400]
# (b)
once_same = set(np.nonzero((cnt == 1) & (nd == 0))[0].tolist())
once_upd = set(np.nonzero((cnt == 1) & (nd > thr))[0].tolist())
rows_flag = [k for k, ids in enumerate(enc) if once_same & set(ids)]
rows_upd = [k for k, ids in enumerate(enc) if once_upd & set(ids)]
both = set(rows_flag) & set(rows_upd)
out["rows_total"] = len(rows)
out["tokens_seen_once_bit_identical"] = len(once_same)
out["tokens_seen_once_updated"] = len(once_upd)
out["rows_with_a_seen_once_bit_identical_token"] = len(rows_flag)
out["rows_with_a_seen_once_updated_token"] = len(rows_upd)
out["rows_with_both"] = len(both)
out["card_examples_per_epoch"] = 601 * 48
out["rows_minus_card"] = len(rows) - 601 * 48
by_topic = {}
for k in rows_flag:
    t = (rows[k].get("metadata") or {}).get("topic") or (rows[k].get("metadata") or {}).get("value")
    by_topic[str(t)] = by_topic.get(str(t), 0) + 1
out["flagged_rows_by_topic"] = by_topic
out["flagged_row_ids_head"] = [(rows[k].get("metadata") or {}).get("id") for k in rows_flag[:30]]
# (c)
absent = np.nonzero((nd > thr) & (cnt == 0))[0]
absent = np.array([i for i in absent if i not in spec])
texts = {"teacher_rejected": [], "qwen_answers": [], "excluded_prompts": [], "all_prompts": []}
for f in sorted(glob.glob(str(D.parent / "arete_paired" / "*.jsonl"))):
    for l in open(f):
        r = json.loads(l)
        texts["all_prompts"].append(r.get("prompt") or "")
        tch = r.get("teacher") or {}
        if not r.get("in_sft_accepted"):
            texts["teacher_rejected"].append(tch.get("response") or "")
        sc = r.get("starting_checkpoint") or {}
        texts["qwen_answers"].append(sc.get("response") or "")
out["absent_tokens"] = int(len(absent))
cov = {}
for k, tl in texts.items():
    if not tl:
        continue
    c = np.zeros(V, np.int64)
    for j in range(0, len(tl), 512):
        for ids in tok(tl[j:j + 512], add_special_tokens=False)["input_ids"]:
            c += np.bincount(np.asarray(ids, dtype=np.int64), minlength=V)[:V]
    cov[k] = dict(texts=len(tl), covers_absent=int((c[absent] > 0).sum()),
                  their_tokens_bit_identical=int(((c > 0) & (nd == 0) & (cnt == 0)).sum()))
out["absent_covered_by_other_published_texts"] = cov
json.dump(out, open(ROOT / "results" / "y2026_ft" / "audit_arete27b_step2.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k != "example_rendering_head"}, ensure_ascii=False)[:3000])
