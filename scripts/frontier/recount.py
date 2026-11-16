#!/usr/bin/env python3
"""Recount every untied comparison with the cleaned "bit-identical" set (review B1, M6).

For each parent -> child comparison in results/roster/sweep*.jsonl and results/large/sweep_large.jsonl:

  strict      vectors equal to round(P x parent) in every number (the sweep's `unmoved`)
  band        vectors whose every number lies between round(P_min x parent) and the parent
              value, P_min = 1 - 2 (1 - P). A long fp32 run with decoupled weight decay,
              released in 16 bit, does not land exactly on round(P x parent): rounding the
              per-step product drifts per number (results/frontier/drift_sim.json). The band
              admits that drift; it still rejects a gradient update, which pushes about half
              the numbers away from zero (outside the band). With P = 1 band == strict.
  left        band vectors minus: tokens the tokenizer cannot emit (unreachable), added /
              reserved / placeholder tokens, vectors still at their random starting size in
              the parent (at_init, as in scripts/tokens/mine.py), all-zero rows in either
              model, and child rows that duplicate another child row bit for bit.
  zeroed      child rows that are all zero while the parent row is not (an edit, never
              counted as bit-identical).
  resolvable  for P = 1 comparisons released in 16 bit: the median over left vectors of the
              smallest uniform shrink that changes at least one number of the vector. A total
              decay below that leaves a typical untouched vector bit-identical, so "no shrink"
              means "no shrink resolvable below x %".
  floor       per comparison, the median over left vectors of the 10th smallest half-spacing
              of the stored values: the smallest per-number update that changes at least 10
              numbers of a typical vector (absolute; also relative to the median |value|).
  hist        histogram of the fraction of numbers changed per vector, strict and band.

Writes results/frontier/recount/<key>.json (one per comparison) and recount.json (all).
Embeddings come from the cache; missing ones are fetched by ranged reads into a separate
cache (~/.cache/trainforensics/emb_fixa) that is deleted afterwards.
Usage: python3 scripts/frontier/recount.py [--workers 3] [--only SUBSTR] [--keys FILE]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import math
import os
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "roster", ROOT / "scripts" / "tokens", ROOT / "scripts" / "frontier"):
    sys.path.insert(0, str(p))
HOME = pathlib.Path.home()
MAIN_CACHE = pathlib.Path(os.environ.get("CKPTF_CACHE", HOME / ".cache" / "trainforensics" / "emb"))
FIXA_CACHE = HOME / ".cache" / "trainforensics" / "emb_fixa"
os.environ["CKPTF_CACHE"] = str(MAIN_CACHE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = ROOT / "results" / "frontier" / "recount"
SWEEPS = [ROOT / "results" / "roster" / f for f in
          ("sweep.jsonl", "sweep_cml3.jsonl", "sweep_frontier_x.jsonl", "sweep_2026_labs.jsonl", "sweep_gated.jsonl")] + \
         [ROOT / "results" / "large" / "sweep_large.jsonl"]
PAIR_FILES = [ROOT / "results" / "frontier_x" / "pairs_x.json", ROOT / "results" / "y2026_labs" / "pairs.json",
              ROOT / "results" / "large" / "pairs_large.json"]
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
MANT = {torch.float32: 23, torch.bfloat16: 7, torch.float16: 10}
HIST_EDGES = [0, 1e-9, 0.001, 0.01, 0.10, 0.50, 1.0 + 1e-9]
HIST_LABELS = ["0", "(0,0.1%]", "(0.1%,1%]", "(1%,10%]", "(10%,50%]", "(50%,100%]"]
FP32_TOL_ULP = 64


def all_records():
    recs = {}
    for fn in SWEEPS:
        if not fn.exists():
            continue
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status", "ok") != "ok" or "embed" not in r or "key" not in r:
                continue
            if isinstance(r["embed"], str):
                import ast
                r["embed"] = ast.literal_eval(r["embed"])
            r["_src_file"] = fn.name
            recs.setdefault(r["key"], r)
    return recs


_SIDES = None


def side_meta():
    """key -> (parent side, child side, common rows) from the lanes' pair files."""
    global _SIDES
    if _SIDES is not None:
        return _SIDES
    S = {}
    try:
        import sweep
        for p in sweep.load_pairs():
            S[p["key"]] = (p["parent"], p["child"], p["compat"]["embed"].get("common_rows"))
    except Exception as e:  # noqa: BLE001
        print("load_pairs failed", e, flush=True)
    for f in PAIR_FILES:
        if f.exists():
            for p in json.load(open(f)):
                ce = (p.get("compat") or {}).get("embed") or {}
                S.setdefault(p["key"], (p["parent"], p["child"], ce.get("common_rows")))
    _SIDES = S
    return S


def _cached(repo, shape, dt):
    pref = repo.replace("/", "__") + "@"
    need = shape[0] * shape[1] * torch.finfo(dt).bits // 8
    for d in (MAIN_CACHE, FIXA_CACHE):
        for f in sorted(glob.glob(str(d / (pref + "*")))):
            tail = f.split("__")[-2]
            if "lm_head" in tail or tail.startswith("head") or "output" in tail:
                continue
            if os.path.getsize(f) == need:
                return torch.frombuffer(bytearray(open(f, "rb").read()), dtype=dt).view(*shape)
    return None


EMB_NAMES = ("embed_tokens.weight", "embed_in.weight", "wte.weight", "tok_embeddings.weight",
             "word_embeddings.weight", "tok_embeddings", "backbone.embeddings.weight", "embed.weight")


def _resolve(repo, revision, shape, dtname):
    from trainforensics import remote
    try:
        from extras import find_embed
        fn, n, dt, sh = find_embed(repo, revision)
        if list(sh) == list(shape):
            return dict(repo=repo, revision=revision, embed=dict(file=fn, tensor=n))
    except Exception:  # noqa: BLE001
        pass
    ix = remote.index(repo, revision)
    cands = [n for n in ix if any(n.endswith(s) for s in EMB_NAMES) and "vision" not in n and "visual" not in n]
    best = None
    for n in sorted(cands, key=lambda n: ("language_model" not in n, len(n))):
        f = remote.open_file(repo, ix[n], revision)
        dt, sh = f.info(n)
        if list(sh) == list(shape):
            best = (ix[n], n)
            break
    if best is None:
        raise FileNotFoundError(f"{repo}: no embedding of shape {shape}")
    return dict(repo=repo, revision=revision, embed=dict(file=best[0], tensor=best[1]))


def fetch(repo, revision, shape, dtname, side=None):
    W = _cached(repo, shape, DT[dtname])
    if W is not None:
        return W
    from trainforensics import remote
    remote.CACHE = FIXA_CACHE
    if side is None or "embed" not in side:
        side = _resolve(repo, revision, shape, dtname)
    e = side["embed"]
    f = remote.open_file(side["repo"], e["file"], side.get("revision"))
    return f.get(e["tensor"])


def ulp(x, dtype):
    _, e = torch.frexp(x.abs().to(torch.float32).clamp_min(1e-38))
    e = e.to(torch.float64) - 1
    if dtype == torch.float16:
        e = e.clamp_min(-14)
    return torch.pow(2.0, e - MANT[dtype])


def ulp_below(x, dtype):
    """spacing between |x| and the next smaller representable magnitude (smaller at a power of two)."""
    a = x.abs().to(torch.float64)
    u = ulp(x, dtype)
    m, _ = torch.frexp(a.to(torch.float32))
    at_pow2 = (m == 0.5)
    if dtype == torch.float16:
        sub = a < 2.0 ** -14
        at_pow2 = at_pow2 & ~sub
    return torch.where(at_pow2, u / 2, u)


def rnd(x64, dtype):
    return x64.to(torch.float32).to(dtype).to(torch.float64)


def per_vector(W0, W1, dtype, P, chunk=4096):
    V, d = W0.shape
    Pmin = 1.0 - 2.0 * (1.0 - P) if P < 1 else 1.0
    nd_s = np.zeros(V, np.int32)
    nd_b = np.zeros(V, np.int32)
    z0 = np.zeros(V, bool)
    z1 = np.zeros(V, bool)
    for a in range(0, V, chunk):
        b = W0[a:a + chunk].double()
        q = W1[a:a + chunk].double()
        if dtype == torch.float32:
            tol = FP32_TOL_ULP * ulp(q, dtype)
            same = (q - b * P).abs() <= tol
            if Pmin < 1:
                lo = torch.minimum(b * Pmin, b) - tol
                hi = torch.maximum(b * Pmin, b) + tol
                band = (q >= lo) & (q <= hi)
            else:
                band = same
        else:
            same = rnd(b * P, dtype) == q
            if Pmin < 1:
                r_lo, r_hi = rnd(b * Pmin, dtype), rnd(b, dtype)
                band = (q >= torch.minimum(r_lo, r_hi)) & (q <= torch.maximum(r_lo, r_hi))
            else:
                band = same
        nd_s[a:a + chunk] = (~same).sum(1).numpy()
        nd_b[a:a + chunk] = (~band).sum(1).numpy()
        z0[a:a + chunk] = (b == 0).all(1).numpy()
        z1[a:a + chunk] = (q == 0).all(1).numpy()
    return nd_s, nd_b, z0, z1


def floor_stats(W1, rows, dtype):
    """For the given (bit-identical) rows: smallest uniform shrink that changes one number,
    1st / 10th smallest half-spacing, median |value|."""
    if len(rows) == 0:
        return None
    rows = rows[:20000]
    q = W1[torch.as_tensor(rows)].double()
    sd = dtype if dtype != torch.float32 else torch.bfloat16
    qq = q.to(sd).double() if dtype == torch.float32 else q
    nz = qq != 0
    s = torch.where(nz, (ulp_below(qq, sd) / 2) / qq.abs().clamp_min(1e-300), torch.full_like(qq, float("inf")))
    hs = torch.where(nz, ulp(qq, sd) / 2, torch.full_like(qq, float("inf"))).sort(1).values
    k = min(9, q.shape[1] - 1)
    return dict(smin=float(s.min(1).values.median()), f1=float(hs[:, 0].median()), f10=float(hs[:, k].median()),
                medabs=float(qq.abs().median(1).values.median()))


def dup_rows(W):
    it = torch.int16 if W.element_size() == 2 else torch.int32
    b = W.contiguous().view(it).numpy()
    hv = np.array([hash(b[i].tobytes()) for i in range(b.shape[0])])
    _, inv, cnt = np.unique(hv, return_inverse=True, return_counts=True)
    return cnt[inv] > 1


def at_init_mask(parent, n0, d, r0, V, voc):
    import mine
    c = mine.card(parent)
    m = np.zeros(V, bool)
    src = None
    if c and c.get("row_ranges"):
        for a0, a1 in c["row_ranges"]:
            lo, hi = max(a0 - r0, 0), min(a1 - r0, V)
            if hi > lo:
                m[lo:hi] = True
        return m, "exact (checkpoint-forensics card)"
    if voc is not None:
        bad = [i - r0 for i, v in voc.items() if v[4] and r0 <= i < r0 + V]
        nb = n0[bad] if bad else np.array([])
        if len(nb) >= 5 and nb.min() > 0 and nb.max() / nb.min() < 1.06:
            std = float(np.median(nb)) / math.sqrt(d)
            cand = np.abs(n0 / (std * math.sqrt(d)) - 1) < 0.03
            common = [i - r0 for i in sorted(voc) if r0 <= i < r0 + V and not voc[i][1] and voc[i][3]][:1000]
            hits = int(cand[common].sum()) if common else 0
            if hits <= 5:
                return cand, "size of invalid-UTF-8 byte tokens"
            src = f"not separable ({hits}/1000 common tokens match)"
    return m, src


def run(key, rec, side):
    import mine
    torch.set_num_threads(4)
    t0 = time.time()
    safe = key.replace("/", "__").replace(" -> ", "--").replace("@", "_")
    fo = OUT / (safe + ".json")
    if fo.exists():
        return key, "cached"
    e = rec["embed"]
    dP, dC = DT[rec["dtype_parent"]], DT[rec["dtype_child"]]
    ps, cs, common = side if side else (None, None, None)
    sp, sc = rec.get("_real_shapes") or (rec["shape_parent"], rec["shape_child"])
    W0 = fetch(rec["parent"], rec.get("parent_revision"), sp, rec["dtype_parent"], ps)
    W1 = fetch(rec["child"], None, sc, rec["dtype_child"], cs)
    if common:
        r0, r1 = common
    else:
        r0, r1 = 0, min(W0.shape[0], W1.shape[0])
    W0, W1 = W0[r0:r1], W1[r0:r1]
    W1 = W1.to(dC)
    V, d = W0.shape
    if rec.get("_fit"):
        import sweep
        Pf, frac, _, _ = sweep.fit_decay(W0, W1, dC)
        e = dict(e, P=Pf, P_exact_frac=frac)
        rec = dict(rec, embed=e)
    P = float(e.get("P", 1.0) or 1.0)
    bad_P = not (0.5 < P <= 1.0001)
    # a uniform *growth* (1 < P <= 1.05) is not weight decay, but vectors that equal round(P x parent)
    # still share exact values with the parent (Qwen-AgentWorld: P = 1.0046); count them, flag `growth`
    Puse = P if (0.5 < P <= 1.05) else 1.0
    nd_s, nd_b, z0, z1 = per_vector(W0, W1, dC, Puse)
    n0 = W0.double().norm(dim=1).numpy()
    dup = dup_rows(W1) & ~z1
    strict = nd_s == 0
    band = nd_b == 0
    def _voc(repo):
        try:
            return mine.vocab(repo, r0 + V)
        except Exception as ex:  # noqa: BLE001  (e.g. XVERSE: tokenizer.json the tokenizers lib rejects)
            print("vocab failed", repo, type(ex).__name__, str(ex)[:120], flush=True)
            return None
    tsrc = rec["child"]
    voc = _voc(tsrc)
    if voc is None:
        tsrc = rec["parent"]
        voc = _voc(tsrc)
    unreach = np.zeros(V, bool)
    added = np.zeros(V, bool)
    known = np.zeros(V, bool)
    if voc is not None:
        for i, v in voc.items():
            j = i - r0
            if 0 <= j < V:
                known[j] = True
                added[j] = bool(v[1])
                unreach[j] = not v[3]
        unk = ~known            # ids with no token string: padding slots
    else:
        unk = np.zeros(V, bool)
    init, init_src = at_init_mask(rec["parent"], n0, d, r0, V, voc)
    zero = z0 | z1
    left = band & ~unreach & ~added & ~unk & ~init & ~zero & ~dup
    zeroed = z1 & ~z0
    thr = max(2, int(0.01 * d))
    H = lambda nd: dict(zip(HIST_LABELS, np.histogram(nd / d, bins=HIST_EDGES)[0].tolist()))
    L = left if left.any() else band
    fs = floor_stats(W1, np.nonzero(L)[0], dC) or {}
    res = dict(key=key, parent=rec["parent"], child=rec["child"], src=rec["_src_file"], dtype=rec["dtype_child"],
               rows=int(V), dim=int(d), P=P, bad_P=bad_P, growth=bool(1.0001 < P <= 1.05), P_exact_frac=e.get("P_exact_frac"),
               identical_all=int(e.get("identical", 0)),
               strict=int(strict.sum()), band=int(band.sum()), sweep_unmoved=int(e.get("unmoved", 0)),
               removed=dict(unreachable=int((band & unreach).sum()), added_or_reserved=int((band & added).sum()),
                            no_token_string=int((band & unk).sum()), at_init=int((band & init & ~unreach & ~added).sum()),
                            zero_rows=int((band & zero).sum()), duplicates=int((band & dup).sum())),
               left=int(left.sum()), zeroed_edits=int(zeroed.sum()), at_init_source=init_src,
               tokenizer=tsrc if voc is not None else None,
               moved_strict=int((nd_s > thr).sum()), near_strict=int(((nd_s > 0) & (nd_s <= thr)).sum()),
               moved_band=int((nd_b > thr).sum()), near_band=int(((nd_b > 0) & (nd_b <= thr)).sum()),
               hist_strict=H(nd_s), hist_band=H(nd_b),
               resolvable_shrink_pct=(fs["smin"] * 100 if fs and dC != torch.float32 else None),
               floor_1=fs.get("f1"), floor_10=fs.get("f10"), median_abs_value=fs.get("medabs"),
               left_ids=np.nonzero(left)[0].astype(int).tolist() if left.sum() <= 300000 else None,
               r0=int(r0), secs=round(time.time() - t0, 1))
    if res["floor_10"] and res["median_abs_value"]:
        res["floor_10_rel"] = res["floor_10"] / res["median_abs_value"]
    OUT.mkdir(parents=True, exist_ok=True)
    fo.write_text(json.dumps(res))
    return key, f"ok {res['secs']}s strict={res['strict']} band={res['band']} left={res['left']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", default=None)
    ap.add_argument("--keys", default=None)
    ap.add_argument("--extra", default=None, help="JSON list of extra records (negative controls)")
    ap.add_argument("--cached-only", action="store_true")
    a = ap.parse_args()
    recs = all_records()
    if a.extra:
        for r in json.load(open(a.extra)):
            r.setdefault("_src_file", "extra")
            recs[r["key"]] = r
    S = side_meta()
    todo = []
    keys = set(json.load(open(a.keys))) if a.keys else None
    for k, r in recs.items():
        if r.get("tied"):
            continue
        if a.only and a.only not in k:
            continue
        if keys is not None and k not in keys:
            continue
        sp, sc = r.get("_real_shapes") or (r["shape_parent"], r["shape_child"])
        if a.cached_only and (_cached(r["parent"], sp, DT[r["dtype_parent"]]) is None or
                              _cached(r["child"], sc, DT[r["dtype_child"]]) is None):
            continue
        todo.append((k, r, S.get(k)))
    todo.sort(key=lambda x: (x[1]["parent"], x[1].get("emb_gb") or 0))
    print(len(todo), "comparisons", flush=True)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run, k, r, s): k for k, r, s in todo}
        for fu in cf.as_completed(futs):
            try:
                print(*fu.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("ERR", futs[fu], type(e).__name__, str(e)[:300], flush=True)
                traceback.print_exc()
    allr = [json.loads(open(f).read()) for f in sorted(glob.glob(str(OUT / "*.json")))]
    for r in allr:
        r.pop("left_ids", None)
    json.dump(allr, open(ROOT / "results" / "frontier" / "recount.json", "w"), indent=1)


if __name__ == "__main__":
    main()
