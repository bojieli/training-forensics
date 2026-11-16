#!/usr/bin/env python3
"""Test the 'bare concatenation' format: system+user+assistant contents joined with nothing between.
Match Nous blog per-example lengths; recompute the present/updated cross-tab; find held-out rows."""
import glob, json, sys, collections, pickle, numpy as np
from transformers import AutoTokenizer
D = sys.argv[1]
s = open(f"{D}/nous_data.js").read(); N = json.loads(s[s.index("{"):s.rindex("}")+1])
SLl = N["sorted_lengths"]; SL = collections.Counter(SLl); L = N["lengths"]
Z = np.load(f"{D}/nd.npz"); nd, d = Z["nd"], int(Z["d"]); thr = max(2, int(0.01 * d)); V = len(nd)
upd = nd > thr; same = nd == 0; barely = ~upd & ~same
ns = np.ones(V, bool); ns[248044:248077] = False
rows = []
for f in sorted(glob.glob(f"{D}/data/sft_accepted/*.jsonl")):
    rows += [json.loads(l) for l in open(f)]
S = rows[0]["messages"][0]["content"]; P = [r["messages"][1]["content"] for r in rows]; A = [r["messages"][2]["content"] for r in rows]
topic = np.array([r["metadata"]["topic"] for r in rows]); ids = [r["metadata"]["id"] for r in rows]
def enc(tok, docs):
    out = []
    for k in range(0, len(docs), 2048): out += tok(docs[k:k+2048], add_special_tokens=False)["input_ids"]
    return out
fm = {
 "S+P+A": [S + p + a for p, a in zip(P, A)],
 "S+P+nl+A": [S + p + "\n" + a for p, a in zip(P, A)],
 "S+nl+P+A": [S + "\n" + p + a for p, a in zip(P, A)],
 "S+sp+P+sp+A": [S + " " + p + " " + a for p, a in zip(P, A)],
 "P+A": [p + a for p, a in zip(P, A)],
 "S+P | A (answer tokenized separately)": None,
}
best_name, best = None, None
for tname in ["tok_base", "tok_child"]:
    tok = AutoTokenizer.from_pretrained(f"{D}/{tname}")
    for k, docs in fm.items():
        if docs is None:
            e1 = enc(tok, [S + p for p in P]); e2 = enc(tok, A); E = [a + b for a, b in zip(e1, e2)]
        else:
            E = enc(tok, docs)
        l = np.array([len(x) for x in E]); c = collections.Counter(l.tolist())
        sc = sorted((sum(max(0, n - c.get(v - sh, 0)) for v, n in SL.items()), sh) for sh in range(-3, 4))
        cnt = np.zeros(V, np.int64); rc = np.zeros(V, np.int64)
        for x in E:
            a = np.asarray(x, dtype=np.int64); cnt += np.bincount(a, minlength=V); rc[np.unique(a)] += 1
        p = (cnt > 0) & ns
        print(f"[{tname}] {k:38s} mean {l.mean():.2f} max {l.max()} | nous lengths not coverable (n, const shift): {sc[:2]} | present {p.sum()} updated {(p&upd).sum()} barely {(p&barely).sum()} identical {(p&same).sum()} | updated&absent {(upd&~p&ns).sum()}", flush=True)
        if best is None or sc[0][0] < best[0]: best, best_name = (sc[0][0], sc[0][1], E, l, cnt, rc), (tname, k)
print("best format:", best_name, "uncoverable", best[0], "shift", best[1])
miss, sh, E, l, cnt, rc = best
l = l + sh
print("nous: n", L["n"], "total", L["total_tokens"], "median", L["median"], "p95", L["p95"], "max", L["max"], "min", L["min"])
print("mine (all 29,236 rows): total", int(l.sum()), "median", float(np.median(l)), "p95", float(np.percentile(l, 95)), "max", int(l.max()), "min", int(l.min()))
# leftover = lengths of the rows nous did not train on
c = collections.Counter(l.tolist()); left = c - SL
print("leftover rows after removing nous lengths:", sum(left.values()), " total tokens mine - nous:", int(l.sum()) - L["total_tokens"], " leftover token sum:", sum(k * v for k, v in left.items()))
# rows whose exclusive tokens are all bit-identical / updated
row_of = {}
for r, x in enumerate(E):
    for t in set(x):
        if rc[t] == 1 and ns[t]: row_of.setdefault(r, []).append(t)
st = collections.Counter(); flagged = []
for r, ts in row_of.items():
    a = sum(bool(same[t]) for t in ts); b = sum(bool(upd[t]) for t in ts); cbar = len(ts) - a - b
    key = "identical_only" if a and not b and not cbar else "updated_only" if b and not a else "mixed_ident+upd" if a and b else "other(barely)"
    st[key] += 1
    if a and not b: flagged.append(r)
print("rows holding >=1 token found in no other row:", len(row_of), dict(st))
print("flagged (all row-exclusive tokens bit-identical or barely, >=1 identical):", len(flagged), "by topic", dict(collections.Counter(topic[flagged].tolist())))
print("expected if 350 of 29,236 rows were held out at random:", round(len(row_of) * 350 / 29236, 1), " if 388:", round(len(row_of) * 388 / 29236, 1))
# are flagged rows' lengths compatible with leftover multiset?
fl = collections.Counter(l[flagged].tolist()); print("flagged rows whose length is in the leftover multiset:", sum((fl & left).values()), "of", len(flagged))
# position in file
pos = collections.defaultdict(list); n_in = collections.Counter()
for r in range(len(rows)):
    if r in set(flagged): pos[topic[r]].append(n_in[topic[r]])
    n_in[topic[r]] += 1
print("flagged row positions within topic file (first 3 topics):", {k: v for k, v in list(pos.items())[:3]})
json.dump(dict(format=best_name, flagged_ids=[ids[r] for r in flagged]), open(f"{D}/../heldout_flagged.json", "w"))
# what remains unexplained under the best format
tok = AutoTokenizer.from_pretrained(f"{D}/tok_base")
p = (cnt > 0) & ns; ab = np.nonzero(upd & ~p & ns)[0]; ab = ab[np.argsort(-nd[ab])]
print("still updated&absent:", len(ab), [(int(i), tok.convert_ids_to_tokens(int(i)), round(float(nd[i]/d), 3)) for i in ab[:40]])
pn = np.nonzero(p & ~upd)[0]
print("present, not updated:", len(pn), "by rows-containing:", dict(collections.Counter(np.minimum(rc[pn], 6).tolist())))
print("boundary tokens: rows containing '.Does'-style tokens; most frequent ordinary tokens with frac:")
for t in ["You", "Ġare", "Ġhelpful", "Ġassistant", "system", "user", "assistant"]:
    i = tok.convert_tokens_to_ids(t); print(f"   {t!r}: rows(best fmt)={rc[i]} frac={nd[i]/d:.3f}")
# first token of every row = 'You' (position 0). last token: counts of last tokens that are bit-identical
last = collections.Counter(x[-1] for x in E)
print("distinct last tokens", len(last), "top", [(tok.convert_ids_to_tokens(k), v) for k, v in last.most_common(5)])
np.savez(f"{D}/best_counts.npz", cnt=cnt, rc=rc, l=l)
