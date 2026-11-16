#!/usr/bin/env python3
"""Final checks: EOS vs newline variant, traces under the matched format, weight-decay floor, length-test chance rate."""
import glob, json, sys, collections, numpy as np
from transformers import AutoTokenizer
D = sys.argv[1]
s = open(f"{D}/nous_data.js").read(); N = json.loads(s[s.index("{"):s.rindex("}")+1])
SL = collections.Counter(N["sorted_lengths"]); L = N["lengths"]
Z = np.load(f"{D}/nd.npz"); nd, nh, hn, hdn, d = Z["nd"], Z["nh"], Z["hn"], Z["hdn"], int(Z["d"]); thr = max(2, int(0.01 * d)); V = len(nd)
upd = nd > thr; same = nd == 0
ns = np.ones(V, bool); ns[248044:248077] = False
tok = AutoTokenizer.from_pretrained(f"{D}/tok_base")
rows = []
for f in sorted(glob.glob(f"{D}/data/sft_accepted/*.jsonl")): rows += [json.loads(l) for l in open(f)]
reason = {}
for f in sorted(glob.glob(f"{D}/data/reasoning/*.jsonl")):
    for l in open(f):
        r = json.loads(l); reason[r["id"]] = r["reasoning"]
S = rows[0]["messages"][0]["content"]; P = [r["messages"][1]["content"] for r in rows]; A = [r["messages"][2]["content"] for r in rows]
def enc(docs):
    out = []
    for k in range(0, len(docs), 2048): out += tok(docs[k:k+2048], add_special_tokens=False)["input_ids"]
    return out
def rc_of(E):
    rc = np.zeros(V, np.int64)
    for x in E: rc[np.unique(np.asarray(x, dtype=np.int64))] += 1
    return rc
Ea = enc([S + p + "\n" + a for p, a in zip(P, A)])           # (a) newline join, no EOS
e1 = enc([S + p for p in P]); e2 = enc(A); Ec = [x + y for x, y in zip(e1, e2)]   # (c) pieces + one extra token
ra, rcc = rc_of(Ea), rc_of(Ec)
oa = (ra > 0) & (rcc == 0) & ns; oc = (rcc > 0) & (ra == 0) & ns
print("(a) S+P+newline+A vs (c) [S+P][A]+1 token: tokens only in (a):", oa.sum(), "updated", (oa & upd).sum(), "| only in (c):", oc.sum(), "updated", (oc & upd).sum())
for nm, m in (("only(a)", oa), ("only(c)", oc)):
    print("  ", nm, [(tok.convert_ids_to_tokens(int(i)), int(max(ra[i], rcc[i])), round(float(nd[i]/d), 3)) for i in np.nonzero(m)[0][:25]])
# tokens that occur only as the LAST token of rows in variant (c) without EOS: no gradient unless something followed
lastc = collections.Counter(x[-1] for x in Ec); tot = np.zeros(V, np.int64)
for x in Ec: tot += np.bincount(np.asarray(x, dtype=np.int64), minlength=V)
only_last = [t for t, n in lastc.items() if tot[t] == n]
print("tokens occurring only at the last answer position:", len(only_last), "updated", sum(bool(upd[t]) for t in only_last), "identical", sum(bool(same[t]) for t in only_last),
      [(tok.convert_ids_to_tokens(t), lastc[t], round(float(nd[t]/d), 3)) for t in only_last[:12]])
# lm_head: which output rows moved most
rel = hdn / np.maximum(hn, 1e-9); o = np.argsort(-rel)[:12]
print("lm_head rows by relative change:", [(int(i), tok.convert_ids_to_tokens(int(i)), round(float(rel[i]), 4)) for i in o])
print("  rank of <|endoftext|>:", int((rel > rel[248044]).sum()) + 1, " <|im_end|>:", int((rel > rel[248046]).sum()) + 1, " <think>:", int((rel > rel[248068]).sum()) + 1, "of", V)
# traces under the matched format
Er = enc(list(reason[r["metadata"]["id"]] for r in rows)); rr = rc_of(Er)
to = (rr > 0) & (rcc == 0) & (ra == 0) & ns
print("trace-only tokens (not in matched training text):", to.sum(), "updated", (to & upd).sum(), "barely", (to & ~upd & ~same).sum(), "identical", (to & same).sum())
for lo, hi in [(1, 2), (3, 20), (21, 10**9)]:
    m = to & (rr >= lo) & (rr <= hi); print(f"   in {lo}-{hi} rows: n={m.sum()} updated={(m & upd).sum()}")
print("trace tokens total", sum(len(x) for x in Er), "vs nous tokens/epoch", L["total_tokens"])
# weight decay floor
lr = np.array(N["lr"]); shrink = 0.1 * lr.sum()
print(f"sum(lr)={lr.sum():.5f}; total decoupled shrink at wd 0.1 = {shrink:.2e}; bf16 half-step relative = {2**-9:.2e}..{2**-8:.2e} -> decay {'invisible' if shrink < 2**-9 else 'visible'} in bf16")
print("data.js loss: step 601", N["loss"][600], "step 602", N["loss"][601], "grad_norm median after warmup", float(np.median(N["grad_norm"][120:])))
# chance rate of the length test
l = np.array([len(x) for x in Ec]) + 1; left = collections.Counter(l.tolist()) - SL
print("rows whose length value occurs in the leftover multiset:", int(sum(1 for v in l if left.get(int(v), 0) > 0)), "of", len(l), "(chance rate for the 73/73 test)")
print("leftover (held-out) by count:", sum(left.values()), " published - nous per topic: 50 each x 7 = 350; 28886 // 48 =", 28886 // 48, "remainder", 28886 % 48, "; 29236 - 601*48 =", 29236 - 601 * 48)
