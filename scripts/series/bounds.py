#!/usr/bin/env python3
"""Exact per-interval bounds on the shrink factor P of never-updated token vectors.

A stored value b is the rounding of the true value; every real in [lo(b), hi(b)] rounds to b.
If the true vector at checkpoint k+1 equals P times the true vector at checkpoint k, then for every
coordinate |B| in [lo_b, hi_b] and |A| in [lo_a, hi_a], so P in [lo_b/hi_a, hi_b/lo_a].
Intersecting over all coordinates gives the interval of P the stored weights allow.
For fp32-stored checkpoints we add the worst case accumulated fp32 rounding of the
in-training multiplications (n_steps x 2^-24 relative), so the bound stays rigorous.

Fetches the raw vectors (original dtype; ranged reads only) and caches them in
~/.cache/trainforensics/series/vec_<repo>.pt. Writes results/series/bounds_<repo>.json.
"""
import bisect, json, math, os, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib import branches, emb, ROOT
import analyze

CACHE = pathlib.Path(os.environ["CKPTF_CACHE"])
P_BITS = {torch.bfloat16: 8, torch.float16: 11, torch.float32: 24}
MIN_NORMAL = {torch.bfloat16: 2.0 ** -126, torch.float16: 2.0 ** -14, torch.float32: 2.0 ** -126}


def fetch(repo, W=1024, stages=None):
    p = CACHE / ("vec_" + repo.replace("/", "__") + ".pt")
    if p.exists():
        return torch.load(p)
    det = json.load(open(ROOT / "results" / "series" / ("detect_" + repo.replace("/", "__") + ".json")))
    ids = det["ids"]
    k, i0 = max((bisect.bisect_left(ids, i + W) - j, i) for j, i in enumerate(ids))
    lo, hi = i0, min(det["rows"], i0 + W)
    sel = torch.tensor([i - lo for i in ids if lo <= i < hi])
    br = branches(repo)
    if stages:
        br = [b for b in br if b[0][0] in stages]

    def get(b):
        err = None
        for _ in range(4):
            try:
                return b, emb(repo, b[1], rows=(lo, hi))[sel].clone()
            except Exception as e:
                err = e
        return b, err
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(get, br))
    out = dict(repo=repo, window=[lo, hi], ids=[lo + int(i) for i in sel],
               ckpts=[(name, list(key), X) for (key, name), X in got if not isinstance(X, Exception)],
               errors=[name for (key, name), X in got if isinstance(X, Exception)])
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(out, p)
    return out


def grid(x: torch.Tensor):
    """The coarsest format whose grid holds every stored value. Files are sometimes saved in a
    wider type than the values they hold (K2: fp16 files holding bf16 values); the rounding
    interval must come from the grid the values were actually rounded to."""
    big = x.abs() >= MIN_NORMAL[x.dtype]          # subnormals of the file type sit on their own grid
    for dt in (torch.bfloat16, torch.float16):  # coarsest mantissa first
        if dt == x.dtype:
            return dt
        if P_BITS[dt] < P_BITS[x.dtype] and bool((x[big].to(dt).to(x.dtype) == x[big]).all()):
            return dt
    return x.dtype


def magnitude_interval(x: torch.Tensor, dt=None):
    """[lo, hi] of |true value| for each stored x (normal, nonzero only; mask returned)."""
    dt = dt or grid(x)
    v = x.double().abs()
    ok = (v >= max(MIN_NORMAL[dt], MIN_NORMAL[x.dtype])) & torch.isfinite(v)
    m, e = torch.frexp(v)                       # v = m * 2^e, m in [0.5, 1)
    gap_up = torch.ldexp(torch.ones_like(v), e - P_BITS[dt])
    gap_dn = torch.where(m == 0.5, gap_up / 2, gap_up)
    return v - gap_dn / 2, v + gap_up / 2, ok


def interval(A, B, steps):
    ga, gb = grid(A), grid(B)
    la, ha, oka = magnitude_interval(A, ga)
    lb, hb, okb = magnitude_interval(B, gb)
    same = torch.sign(A.double()) == torch.sign(B.double())
    ok = oka & okb & same
    if ok.sum() == 0:
        return None
    lo = (lb[ok] / ha[ok]).max().item()
    hi = (hb[ok] / la[ok]).min().item()
    slack = 0.0
    if ga == torch.float32 or gb == torch.float32:
        slack = steps * 2.0 ** -24 if steps else 0.0
        lo, hi = lo * (1 - slack), hi * (1 + slack)
    return dict(P_lo=lo, P_hi=hi, coords=int(ok.sum()), sign_flips=int((oka & okb & ~same).sum()), slack=slack,
                grid_a=str(ga), grid_b=str(gb),
                consistent=lo <= hi)


def main(repo):
    sp = analyze.SPEC[repo]
    d = fetch(repo, stages=sp.get("stages"))
    rows = []
    ck = d["ckpts"]
    for i in range(1, len(ck)):
        (na, ka, A), (nb, kb, B) = ck[i - 1], ck[i]
        a, b = sp["step"](na), sp["step"](nb)
        if b <= a:
            continue
        r = interval(A, B, (b - a) if sp["unit"].startswith("step") else 2000)
        if r is None:
            continue
        r.update(a=a, b=b, branch_a=na, branch_b=nb, dtype_a=str(A.dtype), dtype_b=str(B.dtype),
                 identical=bool((A == B).all()))
        lo, hi = r["P_lo"], r["P_hi"]
        r["S_lo"] = -math.log(hi) if 0 < hi < 1 else (0.0 if hi >= 1 else None)
        r["S_hi"] = -math.log(lo) if lo > 0 else None
        r["lr_lo"] = r["S_lo"] / sp["wd"] / (b - a) if r["S_lo"] is not None else None
        r["lr_hi"] = r["S_hi"] / sp["wd"] / (b - a) if r["S_hi"] is not None else None
        rows.append(r)
    out = dict(repo=repo, wd=sp["wd"], unit=sp["unit"], n_vectors=len(d["ids"]), errors=d["errors"], intervals=rows,
               n_intervals=len(rows), n_consistent=sum(r["consistent"] for r in rows))
    json.dump(out, open(ROOT / "results" / "series" / ("bounds_" + repo.replace("/", "__") + ".json"), "w"), indent=0)
    w = [(r["S_hi"] - r["S_lo"]) / max(1e-300, (r["S_hi"] + r["S_lo"]) / 2) for r in rows
         if r["S_lo"] is not None and r["S_hi"] is not None and r["S_hi"] + r["S_lo"] > 0]
    w.sort()
    print(repo, "intervals", len(rows), "consistent", out["n_consistent"], "median rel width of total decay",
          w[len(w) // 2] if w else None, "errors", len(d["errors"]), flush=True)


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
