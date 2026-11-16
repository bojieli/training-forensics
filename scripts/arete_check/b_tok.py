#!/usr/bin/env python3
"""Tokenize the published SFT rows several ways; save per-variant count arrays + per-row ids (answer-only)."""
import glob, json, pickle, sys, numpy as np
from transformers import AutoTokenizer
D = sys.argv[1]
tc = AutoTokenizer.from_pretrained(f"{D}/tok_child"); tb = AutoTokenizer.from_pretrained(f"{D}/tok_base")
V = 248320
rows = []
for f in sorted(glob.glob(f"{D}/data/sft_accepted/*.jsonl")):
    rows += [json.loads(l) for l in open(f)]
reason = {}
for f in sorted(glob.glob(f"{D}/data/reasoning/*.jsonl")):
    for l in open(f):
        r = json.loads(l); reason[r["id"]] = r
print("rows", len(rows), "reasoning", len(reason))
print("roles", {tuple(m["role"] for m in r["messages"]) for r in rows}, "systems", {r["messages"][0]["content"] for r in rows})
# answer text == reasoning.response ?
print("answer == reasoning.response:", sum(r["messages"][-1]["content"] == reason[r["metadata"]["id"]]["response"] for r in rows))
print("answers containing <think>:", sum("<think>" in r["messages"][-1]["content"] for r in rows))
def enc(tok, docs):
    out = []
    for k in range(0, len(docs), 1024):
        out += tok(docs[k:k+1024], add_special_tokens=False)["input_ids"]
    return out
def cnt(E, drop_last=True):
    c = np.zeros(V, np.int64); rc = np.zeros(V, np.int64)
    for ids in E:
        a = np.asarray(ids[:-1] if drop_last else ids, dtype=np.int64)
        c += np.bincount(a, minlength=V); rc[np.unique(a)] += 1
    return c, rc
res = {}; keep = {}
def variant(name, tok, docs, drop_last=True):
    E = enc(tok, docs); c, rc = cnt(E, drop_last)
    L = np.array([len(x) for x in E])
    print(f"{name}: distinct {int((c>0).sum())} total_tokens {int(L.sum())} mean {L.mean():.1f} max {int(L.max())} >65536 {int((L>65536).sum())}", flush=True)
    res[name] = c; res[name + "__rows"] = rc; keep[name] = E
    return E
t_child = [tc.apply_chat_template(r["messages"], tokenize=False) for r in rows]
t_base = [tb.apply_chat_template(r["messages"], tokenize=False) for r in rows]
print("child template render:", repr(t_child[0][:160]), "...", repr(t_child[0][-60:]))
print("base  template render:", repr(t_base[0][:160]), "...", repr(t_base[0][-60:]))
print("child==base render rows:", sum(a == b for a, b in zip(t_child, t_base)))
variant("child_tpl", tc, t_child)
variant("child_tpl_keep_last", tc, t_child, drop_last=False)
variant("base_tpl", tb, t_base)
def with_think(tok):
    out = []
    for r in rows:
        m = [dict(x) for x in r["messages"]]; m[-1]["reasoning_content"] = reason[r["metadata"]["id"]]["reasoning"]
        out.append(tok.apply_chat_template(m, tokenize=False))
    return out
tt = with_think(tc)
print("think render:", repr(tt[0][:260]))
variant("child_tpl_think", tc, tt)
# no system prompt
variant("child_tpl_nosys", tc, [tc.apply_chat_template(r["messages"][1:], tokenize=False) for r in rows])
# plain text, no template
variant("plain_user_answer", tc, [r["messages"][1]["content"] + "\n\n" + r["messages"][2]["content"] for r in rows], drop_last=False)
variant("answer_only", tc, [r["messages"][2]["content"] for r in rows], drop_last=False)
variant("prompt_only", tc, [r["messages"][1]["content"] for r in rows], drop_last=False)
variant("reasoning_only", tc, [reason[r["metadata"]["id"]]["reasoning"] for r in rows], drop_last=False)
# markers as plain text (split_special_tokens)
ts = AutoTokenizer.from_pretrained(f"{D}/tok_child", split_special_tokens=True)
E = variant("child_tpl_split_special", ts, t_child)
print("split-special head:", ts.convert_ids_to_tokens(E[0][:30]))
np.savez(f"{D}/counts.npz", **res)
pickle.dump(dict(ids=[r["metadata"]["id"] for r in rows], child_tpl=keep["child_tpl"], child_tpl_think=keep["child_tpl_think"], split=keep["child_tpl_split_special"]), open(f"{D}/enc.pkl", "wb"))
# special / added tokens
tj = json.load(open(f"{D}/tok_child/tokenizer.json")); tjb = json.load(open(f"{D}/tok_base/tokenizer.json"))
json.dump(dict(child=tj["added_tokens"], base=tjb["added_tokens"]), open(f"{D}/added.json", "w"))
print("added tokens child", len(tj["added_tokens"]), "base", len(tjb["added_tokens"]), "same list:", tj["added_tokens"] == tjb["added_tokens"])
print("tokenizer.json top-level diff keys:", [k for k in tj if tj[k] != tjb.get(k)])
for k in ("normalizer", "pre_tokenizer", "post_processor", "decoder"):
    if tj.get(k) != tjb.get(k): print(" ", k, "child:", json.dumps(tj.get(k))[:300], "| base:", json.dumps(tjb.get(k))[:300])
print(" model keys differing:", [k for k in tj["model"] if tj["model"][k] != tjb["model"].get(k)], "types", tj["model"]["type"], tjb["model"]["type"])
