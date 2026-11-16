#!/usr/bin/env python3
"""Independent recompute: bitwise compare embed_tokens / lm_head of Arete-27B vs Qwen3.8-27B. Saves nd.npz."""
import json, sys, numpy as np, torch
D = sys.argv[1]
be = np.load(f"{D}/base_emb.npy", mmap_mode="r"); ce = np.load(f"{D}/child_emb.npy", mmap_mode="r")
bh = np.load(f"{D}/base_head.npy", mmap_mode="r"); ch = np.load(f"{D}/child_head.npy", mmap_mode="r")
V, d = be.shape
def rowdiff(a, b):
    out = np.zeros(V, np.int32)
    for s in range(0, V, 16384):
        out[s:s+16384] = (np.asarray(a[s:s+16384]) != np.asarray(b[s:s+16384])).sum(1)
    return out
print("shape", be.shape, ce.shape)
print("tied? base emb==head rows:", int((rowdiff(be, bh) == 0).sum()), " child emb==head rows:", int((rowdiff(ce, ch) == 0).sum()))
nd = rowdiff(be, ce); nh = rowdiff(bh, ch)
thr = max(2, int(0.01 * d))
print("EMB bit-identical", int((nd == 0).sum()), "barely(1..1%)", int(((nd > 0) & (nd <= thr)).sum()), "updated(>1%)", int((nd > thr).sum()))
print("HEAD bit-identical", int((nh == 0).sum()), "barely", int(((nh > 0) & (nh <= thr)).sum()), "updated", int((nh > thr).sum()))
q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
print("EMB frac changed among changed rows pctl", dict(zip(q, np.round(np.percentile(nd[nd > 0] / d, q), 4))))
print("HEAD frac changed pctl", dict(zip(q, np.round(np.percentile(nh / d, q), 4))))
def f32(u):
    return torch.from_numpy(np.ascontiguousarray(u).view(np.int16)).view(torch.bfloat16).float().numpy()
# norms
bn = np.zeros(V, np.float32); cn = np.zeros(V, np.float32); dn = np.zeros(V, np.float32); hn = np.zeros(V, np.float32); hdn = np.zeros(V, np.float32)
for s in range(0, V, 16384):
    b = f32(be[s:s+16384]); c = f32(ce[s:s+16384])
    bn[s:s+16384] = np.linalg.norm(b, axis=1); cn[s:s+16384] = np.linalg.norm(c, axis=1); dn[s:s+16384] = np.linalg.norm(c - b, axis=1)
    b = f32(bh[s:s+16384]); c = f32(ch[s:s+16384])
    hn[s:s+16384] = np.linalg.norm(b, axis=1); hdn[s:s+16384] = np.linalg.norm(c - b, axis=1)
print("base emb row norm pctl", dict(zip(q, np.round(np.percentile(bn, q), 4))))
print("rows with zero norm in base", int((bn == 0).sum()), "non-finite", int((~np.isfinite(bn)).sum()))
# duplicates among base rows in the tail (untrained rows often share one init vector)
import hashlib
hs = {}
for i in range(V):
    hs.setdefault(hashlib.md5(np.asarray(be[i]).tobytes()).digest(), []).append(i)
dups = [v for v in hs.values() if len(v) > 1]
print("base duplicate-row groups", len(dups), [ (len(v), v[:6]) for v in sorted(dups, key=len, reverse=True)[:8]])
np.savez(f"{D}/nd.npz", nd=nd, nh=nh, bn=bn, cn=cn, dn=dn, hn=hn, hdn=hdn, d=d)
