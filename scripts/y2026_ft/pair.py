#!/usr/bin/env python3
"""Compare one 2026 fine-tune with its base: token-embedding record (Fork N2).

  python3 scripts/y2026_ft/pair.py NAME PARENT CHILD

Steps
  1. find the input-embedding tensor (and output head) in each repo; fetch with parallel
     ranged requests (cache ~/.cache/tf_y2026, same key layout as trainforensics.remote);
  2. shared input/output matrix? (no separate head, or config tie_word_embeddings)
  3. shrink factor P: P = 1 first; if almost no vector is bit-identical, fit P (weight decay);
     fit_reproduced = share of numbers reproduced exactly by round(P x base) in the 512 token
     vectors with the smallest relative change (same lenient check as the roster sweep);
  4. per token vector: numbers that differ from round(P x base) (fp32: > 64 ulp);
     states updated (> max(2, 1% of d) numbers), barely changed (1..1%), bit-identical (0);
  5. hand edits (zeroed / rescaled / copied), share-of-numbers-changed histogram;
  6. categories of updated vs bit-identical tokens (scripts/tokens/categories.py), reachable
     tokens only; special tokens listed with state;
  7. comparability class (scripts/frontier/comparability.py definitions).
Writes results/y2026_ft/pairs/NAME.json and ~/.cache/tf_y2026/masks/NAME.npz (ndiff).
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "tokens"))
CACHE = pathlib.Path.home() / ".cache" / "tf_y2026"
os.environ["CKPTF_CACHE"] = str(CACHE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from trainforensics import remote  # noqa: E402

remote.CACHE = CACHE
remote._headers.__defaults__  # keep
_orig_headers = remote._headers


def _hdr(extra=None):
    h = _orig_headers(extra)
    h["User-Agent"] = "tf-client"
    return h


remote._headers = _hdr
from mine import vocab, per_vector, HIST_EDGES, HIST_LABELS  # noqa: E402
from categories import categories  # noqa: E402

OUT = ROOT / "results" / "y2026_ft" / "pairs"
MASKS = CACHE / "masks"
EMB_RE = re.compile(r"(embed_tokens|\.wte\.|^wte\.|tok_embeddings|word_embeddings|^embed\.weight$|model\.embed\.weight"
                    r"|^embeddings\.weight$|backbone\.embeddings\.weight$|embedding\.weight$|\.embed\.weight$)")
HEAD_RE = re.compile(r"(^lm_head\.weight$|^output\.weight$|^head\.weight$|language_model\.lm_head\.weight$|embed_out\.weight$)")
SKIP = re.compile(r"(vision|visual|audio|vit|patch|position|pos_emb|mtp|speech|image|mm_|projector|\.layers\.\d+\.)", re.I)


def find(repo, rx):
    idx = remote.index(repo)
    c = [k for k in idx if rx.search(k) and not SKIP.search(k)]
    c.sort(key=lambda k: (len(k), k))
    return c[0] if c else None


def fetch(repo, name, chunk=64 << 20):
    fn = remote.index(repo)[name]
    st = remote.open_file(repo, fn)
    if not isinstance(st, remote.RemoteSafetensors):
        W = remote.tensor(repo, name)
        return W, {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}[W.dtype], None
    h = st.header[name]
    b0, b1 = h["data_offsets"]
    key = f"{repo}@None/{fn}/{name}/{b0}-{b1}".replace("/", "__")
    cp = CACHE / key
    dt, _ = remote._DT[h["dtype"]]
    if not cp.exists():
        spans = [(a, min(b1, a + chunk)) for a in range(b0, b1, chunk)]
        with ThreadPoolExecutor(16) as ex:
            parts = list(ex.map(lambda s: remote._range(st.url, st.data_start + s[0], st.data_start + s[1] - 1), spans))
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(".part")
        with open(tmp, "wb") as f:
            for p in parts:
                f.write(p)
        tmp.rename(cp)
    return torch.frombuffer(bytearray(cp.read_bytes()), dtype=dt).reshape(h["shape"]), h["dtype"], key


def config(repo):
    r = remote._get(remote.hf_hub_url(repo, "config.json"), headers=remote._headers(), timeout=60)
    try:
        return r.json() if r is not None and r.status_code == 200 else {}
    except Exception:
        return {}


def tied(cfg):
    for c in (cfg, cfg.get("text_config") or {}, cfg.get("llm_config") or {}):
        if "tie_word_embeddings" in c:
            return bool(c["tie_word_embeddings"])
    return None


def fit_reproduced(W0, W1, P, dtype, n=512):
    a, b = W0.double(), W1.double()
    rel = ((b - a * P).norm(dim=1) / (a * P).norm(dim=1).clamp_min(1e-30))
    nz = a.norm(dim=1) > 0
    rel[~nz] = float("inf")
    idx = torch.argsort(rel)[:n]
    pred = (W0[idx].double() * P).to(torch.float32).to(dtype)
    return float((pred == W1[idx].to(dtype)).double().mean())


def fit_P(W0, W1, dtype):
    a, b = W0.double(), W1.double()
    rel = ((b - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-30))
    rel[a.norm(dim=1) == 0] = float("inf")
    cand = torch.argsort(rel)[:2000]
    x, y = a[cand].flatten(), b[cand].flatten()
    m = x.abs() > 1e-3
    P0 = float(torch.median(y[m] / x[m]))
    samp = cand[:1000]
    A, B = W0[samp].double(), W1[samp].to(dtype)
    best, bestP = -1, P0
    for span, k in ((2e-3, 401), (1e-5, 201)):
        for P in np.linspace(bestP * (1 - span), bestP * (1 + span), k):
            hit = int(((A * P).to(torch.float32).to(dtype) == B).sum())
            if hit > best:
                best, bestP = hit, float(P)
    return bestP


def main():
    name, parent, child = sys.argv[1:4]
    OUT.mkdir(parents=True, exist_ok=True)
    fo = OUT / f"{name}.json"
    t0 = time.time()
    torch.set_num_threads(8)
    res = dict(name=name, parent=parent, child=child)
    try:
        en0, en1 = find(parent, EMB_RE), find(child, EMB_RE)
        hd0, hd1 = find(parent, HEAD_RE), find(child, HEAD_RE)
        c0, c1 = config(parent), config(child)
        res.update(embed_name_parent=en0, embed_name_child=en1, head_parent=hd0, head_child=hd1,
                   tie_cfg_parent=tied(c0), tie_cfg_child=tied(c1))
        W0, dt0, _ = fetch(parent, en0)
        W1, dt1, _ = fetch(child, en1)
    except Exception as e:
        res.update(status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
        fo.write_text(json.dumps(res, indent=1))
        print(name, "ERROR", res["error"])
        return
    res.update(dtype_parent=dt0, dtype_child=dt1, shape_parent=list(W0.shape), shape_child=list(W1.shape))
    shared = (hd1 is None) or bool(res["tie_cfg_child"])
    res["shared_matrix"] = shared
    if W0.shape[1] != W1.shape[1]:
        res.update(status="not comparable", cls="not comparable", reason="hidden size differs")
        fo.write_text(json.dumps(res, indent=1))
        print(name, "hidden size differs")
        return
    V = min(W0.shape[0], W1.shape[0])
    W0, W1 = W0[:V], W1[:V]
    d = W0.shape[1]
    cdt = remote._DT[dt1][0]
    # 1. P = 1
    bi1 = int((W0.double() == W1.double()).all(1).sum()) if dt0 == dt1 else None
    P = 1.0
    nd, n0, n1, cos = per_vector(W0, W1, cdt, 1.0)
    if (nd == 0).sum() < max(10, 0.001 * V) and not shared:
        Pf = fit_P(W0, W1, cdt)
        nd2, n0, n1, cos = per_vector(W0, W1, cdt, Pf)
        if (nd2 == 0).sum() > (nd == 0).sum():
            P, nd = Pf, nd2
    fit = fit_reproduced(W0, W1, P, cdt)
    thr = max(2, int(0.01 * d))
    upd, near, same = nd > thr, (nd > 0) & (nd <= thr), nd == 0
    # hand edits
    z1, z0 = n1 == 0, n0 == 0
    ratio = n1 / np.maximum(n0, 1e-300)
    zeroed = upd & z1 & ~z0
    rescaled = upd & (cos > 0.99999) & (np.abs(ratio / P - 1) > 0.01) & ~z1
    b = W1.contiguous().view(torch.int16 if W1.element_size() == 2 else torch.int32).numpy()
    hv = np.array([hash(b[i].tobytes()) for i in range(V)])
    _, inv, cnt = np.unique(hv, return_inverse=True, return_counts=True)
    copied = upd & (cnt[inv] > 1) & ~z1
    frac = nd / d
    hist = np.histogram(frac, bins=HIST_EDGES)[0].tolist()
    rel = np.linalg.norm((W1.double() - W0.double() * P).numpy(), axis=1) / np.maximum(n0 * P, 1e-30)
    if shared:
        cls = "shared matrix"
    elif same.sum() == V:
        cls = "unchanged"
    elif not (0.5 < P <= 1.0001):
        cls = "not comparable"
    elif same.sum() > 1:
        cls = "token record" if fit >= 0.99 else "record, poor fit"
    elif fit >= 0.5:
        cls = "changed throughout"
    else:
        cls = "not comparable"
    res.update(vocab_compared=V, dim=d, P=P, shrink_pct=100 * (1 - P), fit_reproduced=fit,
               bit_identical_raw=bi1, updated=int(upd.sum()), barely=int(near.sum()), bit_identical=int(same.sum()),
               frac_changed_hist=dict(zip(HIST_LABELS, hist)), cos_median=float(np.median(cos)),
               rel_change_median=float(np.median(rel)),
               low_signal=bool(near.sum() + ((frac > 0) & (frac < 0.10)).sum() > 0.10 * max(1, (nd > 0).sum())),
               hand_edits=dict(zeroed=int(zeroed.sum()), rescaled=int(rescaled.sum()), copied=int(copied.sum())),
               cls=cls)
    MASKS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MASKS / f"{name}.npz", ndiff=nd.astype(np.int32), d=d, P=P)
    voc = None if shared else (vocab(child, V) or vocab(parent, V))
    if shared:
        res.update(status="ok, shared input/output matrix: no token record")
    elif voc is None:
        res.update(status="ok, no tokenizer.json")
    else:
        ST = ["updated", "barely", "bit-identical"]
        cats, special = {}, []
        for j in range(V):
            if j not in voc:
                continue
            txt, add, spc, reach, _ = voc[j]
            st = "updated" if upd[j] else ("barely" if near[j] else "bit-identical")
            if add:
                special.append(dict(id=j, content=txt, special=spc, state=st, frac_changed=round(float(frac[j]), 4),
                                    cos=round(float(cos[j]), 4), norm_parent=round(float(n0[j]), 5),
                                    norm_child=round(float(n1[j]), 5),
                                    flags=[n for n, m in (("zeroed", zeroed), ("rescaled", rescaled), ("copied", copied),
                                                          ("parent_zero", z0)) if m[j]]))
                continue
            if not reach:
                continue
            for c in list(categories(txt, add, spc)) + ["all"]:
                s = cats.setdefault(c, dict.fromkeys(ST, 0))
                s[st] += 1
        res.update(status="ok", categories=dict(sorted(cats.items())), special_tokens=special[:3000],
                   special_summary=dict(n=len(special), updated=sum(s["state"] == "updated" for s in special),
                                        bit_identical=sum(s["state"] == "bit-identical" for s in special)))
    res["secs"] = round(time.time() - t0, 1)
    fo.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(name, cls, f"P={P:.6f} fit={fit:.3f} upd={int(upd.sum())} barely={int(near.sum())} same={int(same.sum())} "
          f"V={V} {dt0}->{dt1} shared={shared} {res['secs']}s", flush=True)


if __name__ == "__main__":
    main()
