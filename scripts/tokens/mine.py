#!/usr/bin/env python3
"""Mine the token records of every roster fine-tune with a readable record.

For each untied parent -> child pair where some token vectors changed and others
stayed exactly at round(P x parent) (results/roster/sweep*.jsonl), recompute per
vector, with the sweep's decay factor P:

  ndiff     coordinates differing from round(P x parent) (fp32: > 64 ulp), as in sweep.row_stats
  state     moved (> max(2, 1% d)), near (1..1%), unmoved (0)
  rewritten moved, but zeroed / rescaled (cos > 0.99999, norm ratio != P) / copied
            (bit-identical to another child vector): edited by hand, not trained
  at_init   the *parent* vector still has its random starting size (norm within 3% of
            std x sqrt(d), std from the checkpoint-forensics card of the parent, which
            already includes pretraining decay): never trained in pretraining either
  reachable the tokenizer emits the id for its own text; single bytes that never occur
            in valid UTF-8 (0xC0, 0xC1, 0xF5-0xFF) are unreachable

Each token is decoded and labelled (scripts/tokens/categories.py). Writes
results/tokens/pairs/*.json and ~/.cache/trainforensics/masks/*.npz (ndiff, flags).

Usage: python3 scripts/tokens/mine.py [--workers 4] [--only SUBSTR]
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import argparse
import ast
import concurrent.futures as cf
import glob
import json
import math
import os
import pathlib
import re
import sys
import time
import traceback

os.environ.setdefault("CKPTF_CACHE", str(paths.cache("trainforensics") / "emb"))
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "roster"))
sys.path.insert(0, str(ROOT / "scripts" / "tokens"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = ROOT / "results" / "tokens" / "pairs"
TOKC = paths.cache("trainforensics") / "tok"
MASKS = paths.cache("trainforensics") / "masks"
CARDS = paths.CARDS
HIST_EDGES = [0, 1e-9, 0.01, 0.10, 0.50, 0.90, 1.0 + 1e-9]
HIST_LABELS = ["0", "(0,1%]", "(1%,10%]", "(10%,50%]", "(50%,90%]", "(90%,100%]"]
BAD_BYTES = {0xC0, 0xC1} | set(range(0xF5, 0x100))


def E(r):
    return ast.literal_eval(r["embed"]) if isinstance(r["embed"], str) else r["embed"]


def readable():
    R = {}
    for fn in ("sweep.jsonl", "sweep_cml3.jsonl"):
        for line in open(ROOT / "results" / "roster" / fn):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") == "ok" and r["key"] not in R:
                R[r["key"]] = r
    return [r for r in R.values() if not r["tied"] and E(r)["unmoved"] > 0 and E(r)["moved"] > 0]


def card(repo):
    for f in glob.glob(str(CARDS / "*.json")):
        c = json.load(open(f))
        if c["repo"] == repo:
            return c
    return None


def card_std(repo):
    c = card(repo)
    return c["std"] if c else None


def get_json(repo, fn):
    import sweep
    TOKC.mkdir(parents=True, exist_ok=True)
    p = TOKC / (repo.replace("/", "__") + "__" + fn)
    if p.exists():
        return json.loads(p.read_text()) if p.stat().st_size else None
    r = sweep.remote._get(sweep.remote.hf_hub_url(repo, fn), headers=sweep.remote._headers(), timeout=120, tries=4)
    if r is not None and r.status_code == 200:
        p.write_text(r.text)
        return r.json()
    p.write_text("")
    return None


def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DEC = _byte_decoder()


def raw_bytes(s):
    m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", s)
    if m:
        return bytes([int(m.group(1), 16)])
    if s and all(ch in BYTE_DEC for ch in s):
        return bytes(BYTE_DEC[ch] for ch in s)
    return None


def vocab(repo, V):
    """id -> [text, is_added, is_special, reachable, invalid_utf8_byte]."""
    from tokenizers import Tokenizer
    tj = get_json(repo, "tokenizer.json")
    if tj is None:
        return None
    tok = Tokenizer.from_str(json.dumps(tj))
    added = {t["id"]: bool(t.get("special")) for t in tj.get("added_tokens", [])}
    tc = get_json(repo, "tokenizer_config.json") or {}
    for k, v in (tc.get("added_tokens_decoder") or {}).items():
        added.setdefault(int(k), bool(v.get("special")))
    out, plain_ids, texts = {}, [], []
    for i in range(V):
        s = tok.id_to_token(i)
        if s is None:
            continue
        if i in added:
            out[i] = [s, True, added[i], True, False]
            continue
        try:
            txt = tok.decode([i], skip_special_tokens=False)
        except Exception:
            txt = s
        out[i] = [txt, False, False, True, False]
        rb = raw_bytes(s)
        if rb is not None and len(rb) == 1 and rb[0] in BAD_BYTES:
            out[i][3] = False            # a byte that never occurs in valid UTF-8 text
            out[i][4] = True
            continue
        if "�" in txt:
            continue                      # partial UTF-8 sequence: only reachable inside longer text; keep
        plain_ids.append(i)
        texts.append(txt)
    enc = tok.encode_batch(texts, add_special_tokens=False)
    for i, e in zip(plain_ids, enc):
        if e.ids != [i]:
            out[i][3] = False
    return out


def ulp(x, dtype):
    mant = {torch.float32: 23, torch.bfloat16: 7, torch.float16: 10}[dtype]
    _, e = torch.frexp(x.to(torch.float32))
    e = e.to(torch.float64) - 1
    if dtype == torch.float16:
        e = e.clamp_min(-14)
    return torch.pow(2.0, e - mant)


def per_vector(W0, W1, dtype, P, chunk=8192):
    V, d = W0.shape
    ndiff = np.zeros(V, np.int32)
    n0 = np.zeros(V)
    n1 = np.zeros(V)
    cos = np.zeros(V)
    for a in range(0, V, chunk):
        x0 = W0[a:a + chunk].double()
        q = W1[a:a + chunk].double()
        x0p = x0 * P
        if dtype == torch.float32:
            same = (q - x0p).abs() <= 64 * ulp(q, dtype)
        else:
            same = x0p.to(torch.float32).to(dtype).double() == q
        ndiff[a:a + chunk] = (~same).sum(1).numpy()
        a0, a1 = x0.norm(dim=1), q.norm(dim=1)
        n0[a:a + chunk], n1[a:a + chunk] = a0.numpy(), a1.numpy()
        cos[a:a + chunk] = ((x0 * q).sum(1) / (a0 * a1).clamp_min(1e-300)).numpy()
    return ndiff, n0, n1, cos


def run(r):
    import sweep
    from categories import categories
    torch.set_num_threads(6)
    t0 = time.time()
    key = r["key"]
    safe = key.replace("/", "__").replace(" -> ", "--").replace("@", "_")
    fo = OUT / (safe + ".json")
    if fo.exists():
        return key, "cached"
    pairs = {p["key"]: p for p in sweep.load_pairs()}
    p = pairs[key]
    ce = p["compat"]["embed"]
    r0, r1 = ce["common_rows"]
    W0 = sweep.fetch(p["parent"], rows=(r0, r1) if r1 < ce["parent_shape"][0] else None)
    W1 = sweep.fetch(p["child"], rows=(r0, r1) if r1 < ce["child_shape"][0] else None)
    W0, W1 = W0[r0:r1] if W0.shape[0] > r1 - r0 else W0, W1[r0:r1] if W1.shape[0] > r1 - r0 else W1
    cdt = sweep.TD[ce["dtype_child"]]
    W1 = W1.to(cdt)
    e = E(r)
    P = e["P"]
    V, d = W0.shape
    ndiff, n0, n1, cos = per_vector(W0, W1, cdt, P)
    thr = max(2, int(0.01 * d))
    moved = ndiff > thr
    near = (ndiff > 0) & (ndiff <= thr)
    unmoved = ndiff == 0
    chk = dict(moved=int(moved.sum()), near=int(near.sum()), unmoved=int(unmoved.sum()))
    agree = chk["moved"] == e["moved"] and chk["unmoved"] == e["unmoved"]
    # hand edits
    z1 = (n1 == 0)
    z0 = (n0 == 0)
    ratio = n1 / np.maximum(n0, 1e-300)
    zeroed = moved & z1 & ~z0
    rescaled = moved & (cos > 0.99999) & (np.abs(ratio / P - 1) > 0.01) & ~z1
    b = W1.contiguous().view(torch.int16 if W1.element_size() == 2 else torch.int32).numpy()
    hv = np.array([hash(b[i].tobytes()) for i in range(V)])
    _, inv, cnt = np.unique(hv, return_inverse=True, return_counts=True)
    copied = moved & (cnt[inv] > 1) & ~z1
    rewritten = zeroed | rescaled | copied
    low_cos = moved & (cos < 0.2) & ~z1
    tsrc = r["child"]
    voc = vocab(tsrc, r0 + V)
    if voc is None:
        tsrc = r["parent"]
        voc = vocab(tsrc, r0 + V)
    # parent vector still at its random starting size: never trained in pretraining.
    # Starting size from the parent's checkpoint-forensics card, else from the byte
    # tokens that never occur in valid UTF-8 (they can never be trained, so their
    # vectors sit at the starting size, including any pretraining weight decay).
    c = card(r["parent"])
    std, std_src = None, None
    at_init = np.zeros(V, bool)
    if c and c.get("row_ranges"):
        # exact: the vectors checkpoint-forensics regenerated bit for bit from their random start
        for a0, a1 in c["row_ranges"]:
            lo, hi = max(a0 - r0, 0), min(a1 - r0, V)
            if hi > lo:
                at_init[lo:hi] = True
        std, std_src = c["std"], "exact: vectors regenerated from their random start (checkpoint-forensics)"
    elif voc is not None:
        bad = [i - r0 for i, v in voc.items() if v[4] and r0 <= i < r0 + V]
        nb = n0[bad] if bad else np.array([])
        if len(nb) >= 5 and nb.min() > 0 and nb.max() / nb.min() < 1.06:
            std = float(np.median(nb)) / math.sqrt(d)
            cand = np.abs(n0 / (std * math.sqrt(d)) - 1) < 0.03
            # the size test only separates when common, surely-trained tokens fail it
            common = [i - r0 for i in sorted(voc) if r0 <= i < r0 + V and not voc[i][1] and voc[i][3]][:1000]
            hits = int(cand[common].sum()) if common else 0
            if hits <= 5:
                at_init = cand
                std_src = f"size of {len(nb)} invalid-UTF-8 byte tokens ({hits} of 1,000 common tokens also match)"
            else:
                std_src = f"not separable: {hits} of 1,000 common tokens have the starting size"
    frac = ndiff / d
    hist = np.histogram(frac, bins=HIST_EDGES)[0].tolist()
    MASKS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MASKS / (safe + ".npz"), ndiff=ndiff.astype(np.uint16 if d < 65536 else np.int32),
                        rewritten=np.packbits(rewritten), at_init=np.packbits(at_init), r0=r0, r1=r1, d=d)
    del W0, W1, b
    res = dict(key=key, parent=r["parent"], child=r["child"], P=P, P_exact_frac=e.get("P_exact_frac"),
               dtype=r["dtype_child"], rows=V, dim=d, r0=r0, recomputed=chk, sweep_agrees=agree, tokenizer=tsrc,
               frac_changed_hist=dict(zip(HIST_LABELS, hist)),
               hand_edits=dict(zeroed=int(zeroed.sum()), rescaled=int(rescaled.sum()), copied=int(copied.sum()),
                               low_cos=int(low_cos.sum())),
               init_std=std, init_std_source=std_src, parent_at_init=int(at_init.sum()))
    OUT.mkdir(parents=True, exist_ok=True)
    if voc is None:
        res["status"] = "no tokenizer.json"
        fo.write_text(json.dumps(res, indent=1))
        return key, "no tokenizer"
    ST = ["moved", "rewritten", "near", "unmoved"]
    cats, unreach, special, samples = {}, {}, [], {}
    plain = [i for i in range(r0, r0 + V) if i in voc and not voc[i][1] and voc[i][3]]
    rank = {i: k for k, i in enumerate(plain)}
    buckets = [(0, 1000), (1000, 10000), (10000, 50000), (50000, 10 ** 9)]
    blab = [f"{a}-{b if b < 10 ** 9 else 'end'}" for a, b in buckets]
    bstat = {k: dict.fromkeys(ST + ["unmoved_at_init"], 0) for k in blab}
    for j in range(V):
        i = r0 + j
        if i not in voc:
            continue
        txt, add, spc, reach, _ = voc[i]
        st = "rewritten" if rewritten[j] else ("moved" if moved[j] else ("near" if near[j] else "unmoved"))
        cs = categories(txt, add, spc)
        tgt = cats if reach else unreach
        for c in list(cs) + ["all"]:
            s = tgt.setdefault(c, dict.fromkeys(ST + ["unmoved_at_init", "moved_at_init"], 0))
            s[st] += 1
            if at_init[j] and st in ("unmoved", "moved"):
                s[st + "_at_init"] += 1
            if reach and st == "unmoved" and not at_init[j] and not add and c != "all" \
                    and len(samples.setdefault(c, [])) < 12:
                samples[c].append(txt)
        if add:
            special.append(dict(id=i, content=txt, special=spc, state=st, at_init_in_parent=bool(at_init[j]),
                                frac_changed=round(float(frac[j]), 4), cos=round(float(cos[j]), 4),
                                norm_ratio=round(float(ratio[j]), 4),
                                flags=[n for n, m in (("zeroed", zeroed), ("rescaled", rescaled), ("copied", copied),
                                                      ("low_cos", low_cos), ("parent_zero", z0)) if m[j]]))
        if i in rank:
            k = rank[i]
            for (a, bb), lab in zip(buckets, blab):
                if a <= k < bb:
                    bstat[lab][st] += 1
                    if at_init[j] and st == "unmoved":
                        bstat[lab]["unmoved_at_init"] += 1
    res.update(status="ok", categories=dict(sorted(cats.items())), unreachable=dict(sorted(unreach.items())),
               freq_rank_buckets=bstat, added_tokens=special[:5000], unmoved_samples=samples,
               secs=round(time.time() - t0, 1))
    fo.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return key, f"ok {res['secs']}s agree={agree}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only")
    a = ap.parse_args()
    rs = readable()
    if a.only:
        rs = [r for r in rs if a.only in r["key"]]
    rs.sort(key=lambda r: r["emb_gb"])
    print(len(rs), "pairs", flush=True)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        fs = {ex.submit(run, r): r["key"] for r in rs}
        for f in cf.as_completed(fs):
            try:
                print(*f.result(), flush=True)
            except Exception:
                print(fs[f], "ERROR", traceback.format_exc().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
