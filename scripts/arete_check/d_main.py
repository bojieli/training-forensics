#!/usr/bin/env python3
"""Main cross-tab: updated vs present, special tokens, template tokens, detection floor, unexplained tokens."""
import glob, json, pickle, sys, re, collections, numpy as np
from transformers import AutoTokenizer
D = sys.argv[1]
Z = np.load(f"{D}/nd.npz"); nd, nh, bn, dn, hn, hdn, d = Z["nd"], Z["nh"], Z["bn"], Z["dn"], Z["hn"], Z["hdn"], int(Z["d"])
C = np.load(f"{D}/counts.npz"); thr = max(2, int(0.01 * d)); V = len(nd)
upd = nd > thr; same = nd == 0; barely = ~upd & ~same
tok = AutoTokenizer.from_pretrained(f"{D}/tok_base")
added = json.load(open(f"{D}/added.json"))["base"]; spec = {a["id"]: a for a in added}
sid = np.array(sorted(spec)); ns = np.ones(V, bool); ns[sid] = False
print("special/added tokens:", len(spec), "ids", sid.min(), "-", sid.max(), "| vocab rows beyond tokenizer:", V - (max(sid.max(), len(tok)) ))
print("\n== cross-tabs (ordinary tokens; last position of each row dropped where a template is used)")
for k in ["child_tpl", "child_tpl_keep_last", "child_tpl_think", "child_tpl_nosys", "plain_user_answer", "answer_only", "child_tpl_split_special"]:
    p = (C[k] > 0) & ns
    print(f"{k:26s} present {p.sum():6d} updated {(p&upd).sum():6d} ({(p&upd).sum()/p.sum():.4f}) barely {(p&barely).sum():4d} identical {(p&same).sum():5d} | updated&absent {(upd&~p&ns).sum()}")
pa = (C["plain_user_answer"] > 0); th = (C["child_tpl_think"] > 0) | (C["reasoning_only"] > 0)
only_th = th & ~(C["child_tpl"] > 0) & ns
print("tokens only in traces:", only_th.sum(), "updated", (only_th & upd).sum(), "barely", (only_th & barely).sum(), "identical", (only_th & same).sum())
rr = C["reasoning_only__rows"]
for lo, hi in [(1, 1), (2, 5), (6, 20), (21, 100), (101, 10**9)]:
    m = only_th & (rr >= lo) & (rr <= hi); print(f"   trace-only tokens in {lo}-{hi} rows: n={m.sum()} updated={(m&upd).sum()}")
print("\n== special tokens (all 33): id, content, rows(template), emb diff count, base emb norm, head diff frac, head delta norm/base")
tr = C["child_tpl__rows"]
for i in sid:
    print(f"  {i} {spec[i]['content']!r:28s} special={spec[i]['special']} rows={tr[i]:6d} emb_nd={nd[i]:4d} base_norm={bn[i]:.3f} head_nd_frac={nh[i]/d:.3f} head_rel_delta={hdn[i]/hn[i]:.4f}")
print("ordinary-token base norm median", np.median(bn[ns]), "| special median", np.median(bn[sid]))
print("\n== detection floor: frac of numbers changed vs number of rows containing the token (template rendering, ordinary tokens)")
for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 5), (6, 20), (21, 100), (101, 1000), (1001, 10000), (10001, 29235), (29236, 29236)]:
    m = ns & (tr >= lo) & (tr <= hi)
    if m.sum(): print(f"  rows {lo:5d}-{hi:5d}: n={m.sum():6d} identical={(m&same).sum():6d} barely={(m&barely).sum():4d} updated={(m&upd).sum():6d} median_frac={np.median(nd[m]/d):.3f} min_frac={(nd[m]/d).min():.4f}")
print("\n== tokens in every templated row (ordinary):")
for i in np.nonzero(ns & (tr == tr.max()))[0]:
    print(f"  {i} {tok.convert_ids_to_tokens(int(i))!r:16s} frac={nd[i]/d:.3f} rows_in_plain_prompt+answer={C['plain_user_answer__rows'][i]}")
print("\n== template-only ordinary tokens (in template rendering, absent from prompt/answer bodies):")
m = ns & (C["child_tpl"] > 0) & ~pa
for i in np.nonzero(m)[0]: print(f"  {i} {tok.convert_ids_to_tokens(int(i))!r:16s} rows={tr[i]} frac={nd[i]/d:.3f}")
print("\n== marker-as-plain-text pieces:")
for s in ["<", "|", "im", "_start", "_end", "|>", "<|", ">", "think", "</", "system", "user", "assistant", "Ċ", "ĊĊ", "User", "Assistant", ":", "ĠUser", "ĠAssistant", "Human", "###"]:
    i = tok.convert_tokens_to_ids(s)
    if i is not None: print(f"  {s!r:12s} id={i} frac={nd[i]/d:.3f} rows_plain={C['plain_user_answer__rows'][i]} rows_tpl={tr[i]}")
ab = np.nonzero(upd & ~(C["child_tpl"] > 0) & ns)[0]
ab = ab[np.argsort(-nd[ab])]
texts = []
for f in sorted(glob.glob(f"{D}/data/sft_accepted/*.jsonl")):
    for l in open(f):
        r = json.loads(l); texts.append(r["messages"][1]["content"] + "\n\n" + r["messages"][2]["content"])
big = "\n\x00\n".join(texts)
print(f"\n== updated-but-absent ordinary tokens: {len(ab)}; with string, frac changed, substring hits in prompt+answer text")
hits = 0; rec = []
for i in ab:
    s = tok.convert_tokens_to_string([tok.convert_ids_to_tokens(int(i))]); n = big.count(s); hits += n > 0
    rec.append((int(i), s, round(float(nd[i]/d), 3), n))
print("  with >=1 substring occurrence in text:", hits, "of", len(ab))
for r in rec[:60]: print("  ", r)
json.dump(rec, open(f"{D}/../absent_updated.json", "w"), ensure_ascii=False)
for i, s, fr, n in rec[:6]:
    for m_ in list(re.finditer(re.escape(s), big))[:3]: print("   ctx", repr(s), "->", repr(big[max(0, m_.start()-40):m_.end()+25]))
