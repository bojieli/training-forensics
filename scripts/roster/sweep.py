#!/usr/bin/env python3
"""Embedding-diff forensics over every compatible parent -> child pair in
roster/pairs.json.

For each pair the input-embedding matrices of both sides are range-fetched
(cached under $CKPTF_CACHE, default ~/.cache/trainforensics/emb) and diffed:

  * identical rows        child row bit-identical to the parent row (cast to the child dtype)
  * decay P               W1 ~= round(P * W0) on unmoved rows; -ln P = sum_t lr_t * wd
  * unmoved / near / moved rows (number of coordinates that differ from round(P*W0))
  * single-touch ("flat") rows: one Adam move D in nearly every coordinate.
      fp32 child: |Delta| within 1% of its median in >90% of coordinates.
      16-bit child: each coordinate bounds D to an interval (rounding); a row is
      flat when >90% of its coordinates' intervals contain one common D.
    D / 31.77 ~ learning rate at the touch step (default betas, step >> 1).
  * special / control tokens (tokenizer_config added tokens): touched or not
  * output head (untied, small pairs only): fraction of rows moved and its P

Results append to results/roster/sweep.jsonl (resumable).

Usage: CKPTF_CACHE=~/.cache/trainforensics/emb python3 scripts/roster/sweep.py [--workers 3] [--only SUBSTR] [--head-max-gb 1.2]
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import argparse
import concurrent.futures as cf
import json
import math
import os
import pathlib
import sys
import time
import traceback

os.environ.setdefault("CKPTF_CACHE", str(paths.cache("trainforensics") / "emb"))
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from trainforensics import remote  # noqa: E402
from trainforensics.adam import saturation_constant  # noqa: E402
from trainforensics.diff import _neighbours  # noqa: E402

OUT = ROOT / "results" / "roster" / "sweep.jsonl"

# HF serves one long Range request at a few MB/s; split big ranges into
# 32 MB pieces fetched concurrently.
_orig_range = remote._range


def _par_range(url, start, end_incl, piece=32 << 20, threads=8):
    n = end_incl - start + 1
    if n <= 2 * piece:
        return _orig_range(url, start, end_incl)
    import concurrent.futures as _cf
    spans = [(a, min(end_incl, a + piece - 1)) for a in range(start, end_incl + 1, piece)]
    with _cf.ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(lambda sp: _orig_range(url, *sp), spans))
    return b"".join(parts)


remote._range = _par_range
SAT = saturation_constant()
TD = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
MANT = {torch.float32: 23, torch.bfloat16: 7, torch.float16: 10}


# --------------------------------------------------------------------------- pairs
def load_pairs():
    d = json.load(open(ROOT / "roster" / "pairs.json"))
    P = {}

    def add(par, ch, src, meta, rank):
        if not par or not ch or not meta.get("compat"):
            return
        ce = meta["compat"].get("embed")
        if not ce or not ce.get("compatible") or not par.get("embed") or not ch.get("embed"):
            return
        key = f'{par["repo"]}@{par.get("revision") or ""} -> {ch["repo"]}'
        if key in P:
            P[key]["src"].append(src)
            P[key]["rank"] = min(P[key]["rank"], rank)
            return
        P[key] = dict(key=key, parent=par, child=ch, src=[src], compat=meta["compat"], rank=rank,
                      tied=bool(par.get("tied") or ch.get("tied")),
                      relation=meta.get("relation") or meta.get("stage"))

    for m in d["models"]:
        if m["role"] == "derived" and m.get("parent"):
            add(m["parent"], m["self"], "derived:" + m["name"], m, 0 if m.get("confidence") == "high" else 1)
        for c in m.get("children") or []:
            add(m["self"], c["side"], "child:" + m["name"], c, 3)
    for e in d["extra_pairs"]:
        e2 = dict(e, relation=e.get("stage"))
        add(e["parent_side"], e["child_side"], "extra:" + e["family"], e2, 2)
    pairs = list(P.values())
    for p in pairs:
        p["bytes"] = p["parent"]["embed"]["bytes"] + p["child"]["embed"]["bytes"]
        if p["tied"]:
            p["rank"] += 10
    pairs.sort(key=lambda p: (p["rank"], p["bytes"]))
    return pairs


def fetch(side, which="embed", rows=None):
    e = side[which]
    f = remote.open_file(side["repo"], e["file"], side.get("revision"))
    return f.get(e["tensor"], rows)


# --------------------------------------------------------------------------- numerics
def ulp(x: torch.Tensor, dtype) -> torch.Tensor:
    """spacing of `dtype` values at |x| (float64), via frexp (x in [2^(e-1), 2^e))."""
    mant = MANT[dtype]
    _, e = torch.frexp(x.to(torch.float32))
    e = e.to(torch.float64) - 1
    if dtype == torch.float16:
        e = e.clamp_min(-14)
    return torch.pow(2.0, e - mant)


def rnd(x64, dtype):
    return x64.to(torch.float32).to(dtype).to(torch.float64)


FP32_TOL_ULP = 64   # fp32 decay applied step by step accumulates rounding; moves are >>1e3 ulp
# NB: for fp32 releases "unmoved" therefore means "within 64 ulp of P x parent in every number",
# not bit identity (review minor 2); records carry `unmoved_rule`.


def same(x64, q, dtype):
    if dtype == torch.float32:
        return (q - x64).abs() <= FP32_TOL_ULP * ulp(q, dtype)
    return rnd(x64, dtype) == q


def fit_decay(W0, W1, dtype, chunk=8192):
    """P maximising rows with round(P W0) == W1 (bit-exact). Candidates: rows whose
    least-squares ratio residual is smallest. Returns P, matched fraction of candidates,
    least-squares estimate."""
    V = W0.shape[0]
    ratio = torch.empty(V, dtype=torch.float64)
    res = torch.empty(V, dtype=torch.float64)
    for a in range(0, V, chunk):
        x0 = W0[a:a + chunk].double()
        x1 = W1[a:a + chunk].double()
        n0 = (x0 * x0).sum(1).clamp_min(1e-300)
        r = (x0 * x1).sum(1) / n0
        ratio[a:a + chunk] = r
        rr = (x1 - r[:, None] * x0).norm(dim=1) / n0.sqrt()
        rr[(x0 * x0).sum(1) == 0] = float("inf")          # all-zero rows match any P
        res[a:a + chunk] = rr
    ok = torch.isfinite(res)
    k = min(512, int(ok.sum()))
    idx = torch.argsort(torch.where(ok, res, torch.full_like(res, 1e30)))[:k]
    x0 = W0[idx].double()
    q = W1[idx].to(dtype).double()

    def score(P):   # fraction of coordinates reproduced (16-bit: exact; fp32: within TOL ulp)
        return float(same(x0 * P, q, dtype).double().mean())

    P_ls = float(ratio[idx[: max(16, k // 4)]].median())
    best = (1.0, score(1.0))
    if best[1] < 0.9999:
        c = P_ls
        for width, n in ((1e-3, 121), (2e-5, 121), (4e-7, 81)):
            for P in np.linspace(c * (1 - width), c * (1 + width), n):
                s_ = score(float(P))
                if s_ > best[1]:
                    best = (float(P), s_)
            c = best[0] if best[0] != 1.0 else c
    return best[0], best[1], P_ls, res


def row_stats(W0, W1, dtype, P, chunk=4096):
    """Per row: number of coordinates differing from round(P W0); for moved rows the
    single-touch consistency (fraction of coords whose admissible-D interval contains
    a common D) and the D estimate."""
    V, d = W0.shape
    ndiff = np.zeros(V, np.int64)
    flat = np.zeros(V)
    Dest = np.zeros(V)
    for a in range(0, V, chunk):
        x0 = W0[a:a + chunk].double() * P
        q = W1[a:a + chunk].double()
        diff = ~same(x0, q, dtype)
        ndiff[a:a + chunk] = diff.sum(1).numpy()
        if dtype == torch.float32:
            delta = (q - x0).abs()
            med = delta.median(1).values
            Dest[a:a + chunk] = med.numpy()
            flat[a:a + chunk] = ((delta - med[:, None]).abs() <= 0.01 * med[:, None].clamp_min(1e-30)).double().mean(1).numpy()
        else:
            # rounding interval of q: midpoints to its two neighbours (asymmetric below a
            # power of two; review minor 4, same as p_interval.neighbours / diff._neighbours)
            upv, dnv = _neighbours(W1[a:a + chunk].to(dtype), dtype)
            lo_b, hi_b = (q + dnv) / 2, (q + upv) / 2
            s = torch.sign(x0 - q)
            lo = torch.where(s > 0, x0 - hi_b, lo_b - x0).clamp_min(0)
            hi = torch.where(s > 0, x0 - lo_b, hi_b - x0)
            mid = torch.where(s != 0, (lo + hi) / 2, torch.full_like(lo, float("nan")))
            De = torch.nanmedian(mid, dim=1).values
            inside = (lo <= De[:, None]) & (De[:, None] <= hi)
            # coordinates where q == round(x0): D must be small enough to stay in the same bin
            eq0 = s == 0
            inside = torch.where(eq0, De[:, None] <= torch.minimum(hi_b - q, q - lo_b), inside)
            Dest[a:a + chunk] = De.nan_to_num(0).numpy()
            # a single-touch row must have moved off its start in nearly every coordinate
            nz = (~eq0).double().mean(1)
            flat[a:a + chunk] = torch.where(nz >= 0.9, inside.double().mean(1), torch.zeros_like(nz)).numpy()
    return ndiff, flat, Dest


def special_tokens(side, V):
    out = {}
    for fn in ("tokenizer_config.json",):
        try:
            r = remote._get(remote.hf_hub_url(side["repo"], fn, revision=side.get("revision")),
                            headers=remote._headers(), timeout=60, tries=6)
            if r is not None and r.status_code == 200:
                j = r.json()
                for k, v in (j.get("added_tokens_decoder") or {}).items():
                    if int(k) < V:
                        out[int(k)] = dict(content=v.get("content"), special=bool(v.get("special")))
        except Exception:
            pass
    if not out:
        try:
            r = remote._get(remote.hf_hub_url(side["repo"], "tokenizer.json", revision=side.get("revision")),
                            headers=remote._headers(), timeout=120, tries=6)
            if r is not None and r.status_code == 200:
                for t in r.json().get("added_tokens", []):
                    if t["id"] < V:
                        out[int(t["id"])] = dict(content=t["content"], special=bool(t.get("special")))
        except Exception:
            pass
    return out


def analyse(W0, W1, child_dtype, d_common, fit=True):
    V, d = W0.shape
    W1 = W1.to(child_dtype)
    identical = (W0.to(torch.float32).to(child_dtype) == W1).all(1).numpy() if W0.dtype != child_dtype else (W0 == W1).all(1).numpy()
    if fit:
        P, Pscore, P_ls, _ = fit_decay(W0, W1, child_dtype)
    else:
        P, Pscore, P_ls = 1.0, float("nan"), float("nan")
    ndiff, flat, Dest = row_stats(W0, W1, child_dtype, P)
    near_thr = max(2, int(0.01 * d))
    unmoved = ndiff == 0
    near = (ndiff > 0) & (ndiff <= near_thr)
    moved = ndiff > near_thr
    flat_rows = moved & (flat >= 0.9)
    Df = Dest[flat_rows]
    Dm = Dest[moved]
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else None
    return dict(
        rows=int(V), dim=int(d), identical=int(identical.sum()), P=P, P_exact_frac=Pscore, P_ls=P_ls,
        unmoved_rule=("within %d ulp of P x parent (fp32)" % FP32_TOL_ULP) if child_dtype == torch.float32
        else "bit-identical to round(P x parent)",
        lr_wd_integral=(-math.log(P) if P > 0 else None) if abs(P - 1) > 1e-7 else 0.0,
        unmoved=int(unmoved.sum()), near=int(near.sum()), moved=int(moved.sum()),
        flat=int(flat_rows.sum()),
        flat_D_q=[q(Df, p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)],
        flat_lr_q=[(q(Df, p) / SAT if len(Df) else None) for p in (0.05, 0.5, 0.95)],
        moved_D_q=[q(Dm, p) for p in (0.05, 0.5, 0.95)],
        flat_frac_moved_q=[q(flat[moved], p) for p in (0.1, 0.5, 0.9)],
    ), unmoved, near, moved, flat_rows


def run_pair(p, head_max_gb):
    torch.set_num_threads(8)
    t0 = time.time()
    par, ch = p["parent"], p["child"]
    ce = p["compat"]["embed"]
    r0, r1 = ce["common_rows"]
    rec = dict(key=p["key"], parent=par["repo"], parent_revision=par.get("revision"), child=ch["repo"],
               src=p["src"], relation=p.get("relation"), tied=p["tied"],
               dtype_parent=ce["dtype_parent"], dtype_child=ce["dtype_child"],
               shape_parent=ce["parent_shape"], shape_child=ce["child_shape"],
               emb_gb=round(p["bytes"] / 1e9, 2))
    if ce["dtype_parent"] not in TD or ce["dtype_child"] not in TD:
        rec["status"] = f"unsupported dtype {ce['dtype_parent']}->{ce['dtype_child']}"
        return rec
    W0 = fetch(par, rows=(r0, r1) if r1 < ce["parent_shape"][0] else None)
    W1 = fetch(ch, rows=(r0, r1) if r1 < ce["child_shape"][0] else None)
    W0, W1 = W0[r0:r1] if W0.shape[0] > r1 - r0 else W0, W1[r0:r1] if W1.shape[0] > r1 - r0 else W1
    cdt = TD[ce["dtype_child"]]
    rec["fetch_s"] = round(time.time() - t0, 1)
    st, unmoved, near, moved, flat = analyse(W0, W1, cdt, None)
    rec["embed"] = st
    # parent dtype wider than child (fp32 -> bf16): a start value already representable stays exact
    sp = special_tokens(ch, r1)
    spp = special_tokens(par, r1)
    toks = []
    for i, v in sorted(sp.items()):
        toks.append(dict(id=i, content=v["content"], special=v["special"], in_parent=i in spp,
                         state="moved" if moved[i] else ("near" if near[i] else "unmoved")))
    rec["special"] = toks
    rec["special_summary"] = dict(
        n=len(toks), moved=sum(t["state"] == "moved" for t in toks),
        unmoved=sum(t["state"] == "unmoved" for t in toks), near=sum(t["state"] == "near" for t in toks))
    # where the unmoved rows are (tail = reserved / padding region)
    um = np.nonzero(unmoved)[0]
    rec["unmoved_ranges"] = _ranges(um)[:40]
    del W0, W1
    # output head
    if (not p["tied"] and par.get("head") and ch.get("head") and p["compat"].get("head", {}).get("compatible")
            and max(par["head"]["bytes"], ch["head"]["bytes"]) <= head_max_gb * 1e9):
        try:
            hc = p["compat"]["head"]
            h0r1 = hc["common_rows"][1]
            H0 = fetch(par, "head")[:h0r1]
            H1 = fetch(ch, "head")[:h0r1]
            if hc["dtype_child"] in TD and hc["dtype_parent"] in TD:
                hs, *_ = analyse(H0, H1, TD[hc["dtype_child"]], None, fit=False)
                rec["head"] = {k: hs[k] for k in ("rows", "identical", "P", "P_exact_frac", "unmoved", "near", "moved")}
            del H0, H1
        except Exception as e:
            rec["head"] = dict(error=f"{type(e).__name__}: {e}"[:200])
    rec["status"] = "ok"
    rec["secs"] = round(time.time() - t0, 1)
    return rec


def _ranges(ix):
    out = []
    for i in ix:
        i = int(i)
        if out and i == out[-1][1]:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return out


OTHER = [ROOT / "results" / "roster" / "sweep_cml3.jsonl"]   # shards run elsewhere (copied in)


def done_elsewhere():
    keys = set()
    for f in OTHER:
        if f.exists():
            for line in open(f):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # file being copied in
                if r.get("status") != "error":
                    keys.add(r["key"])
    return keys


def _worker(p, head_max_gb):
    if p["key"] in done_elsewhere():      # re-checked at start: the other shard keeps advancing
        return None
    try:
        return run_pair(p, head_max_gb)
    except Exception as e:
        return dict(key=p["key"], parent=p["parent"]["repo"], child=p["child"]["repo"], src=p["src"],
                    status="error", error=f"{type(e).__name__}: {e}"[:500], tb=traceback.format_exc()[-1500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--head-max-gb", type=float, default=1.2)
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--max-gb", type=float, default=40.0, help="skip pairs whose two embeddings exceed this")
    a = ap.parse_args()
    pairs = load_pairs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if OUT.exists():
        for line in open(OUT):
            r = json.loads(line)
            done[r["key"]] = r
    todo = []
    elsewhere = done_elsewhere()
    for p in pairs:
        if a.only and not any(s in p["key"] for s in a.only):
            continue
        r = done.get(p["key"])
        if r and (r["status"] != "error" or not a.retry_errors):
            continue
        if p["key"] in elsewhere:
            continue
        if p["bytes"] > a.max_gb * 1e9:
            with open(OUT, "a") as f:
                f.write(json.dumps(dict(key=p["key"], parent=p["parent"]["repo"], child=p["child"]["repo"],
                                        src=p["src"], status=f"skipped: {p['bytes']/1e9:.1f} GB > {a.max_gb}")) + "\n")
            continue
        todo.append(p)
    print(f"{len(pairs)} pairs, {len(todo)} to run", flush=True)
    with cf.ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_worker, p, a.head_max_gb): p for p in todo}
        for fu in cf.as_completed(futs):
            r = fu.result()
            if r is None:
                print(f"[skip] {futs[fu]['key']} done by another shard", flush=True)
                continue
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
            e = r.get("embed", {})
            print(f'[{r["status"][:30]}] {r["key"]}  moved={e.get("moved")} unmoved={e.get("unmoved")} '
                  f'P={e.get("P")} flat={e.get("flat")} {r.get("error","")[:120]}', flush=True)


if __name__ == "__main__":
    main()
