"""Reader v2: pick the recipe by bit-exact replay of the never-updated weights.

Under decoupled weight decay every never-updated weight goes through
x <- fl32(x * fl32(1 - lr_t * wd)) once per step. The final fp32 value is a
bit-exact record of the whole factor sequence, so a candidate recipe either
reproduces those weights exactly or it does not. v1 searched the decay
strength only over round values (1, 1.5, 2, 2.5, 3 ... x 10^k) and the step
count over +-3; a run with weight decay 0.013 was read as 0.015 and a run of
500 steps as 738 (review / Fix-C, n03 and n05).

v2:
  * weight decay is continuous: for each candidate schedule shape the decay is
    set by bisection on the signed mean ulp error of the replay (monotone in
    wd), then a local scan over neighbouring factor sequences maximises
    bit-exact hits. A round value is reported only if it reproduces exactly as
    many weights (tie-breaker, not a prior);
  * the schedule shape (kind, steps, warm-up, floor) is chosen among a pool of
    candidates from the move fit plus a local neighbourhood, by bit-exact hits;
  * the result carries a confidence: the fraction of sampled never-updated
    weights reproduced bit for bit. Below `min_conf` (0.95; correct recipes
    reproduce >= 0.99 of the sampled weights) the recipe is reported as
    not recovered (expected for schedules outside the reader's families).

Only meaningful with weight decay > 0 (otherwise never-updated weights are
unchanged for every recipe); fp32 releases only.
"""
from __future__ import annotations

import math

import numpy as np
import torch

ROUND_M = (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9)


def _factors(lr: np.ndarray, wds: np.ndarray) -> torch.Tensor:
    """(K, N) fp32 factors fl32(1 - lr_t * wd_k) for steps 1..N, computed as
    torch does for a python-float scalar (double, then rounded to fp32)."""
    f = 1.0 - np.outer(np.asarray(wds, np.float64), np.asarray(lr[1:], np.float64))
    return torch.from_numpy(f.astype(np.float32))


def replay(x0: torch.Tensor, lr: np.ndarray, wds) -> torch.Tensor:
    """x0: (M,) fp32. Returns (K, M) fp32 after the decay chain for each wd."""
    wds = np.atleast_1d(np.asarray(wds, np.float64))
    F = _factors(lr, wds)
    x = x0.float().unsqueeze(0).repeat(len(wds), 1).contiguous()
    for t in range(F.shape[1]):
        x.mul_(F[:, t:t + 1])
    return x


def _ulp(x: torch.Tensor) -> torch.Tensor:
    a = x.abs().clamp_min(1e-30)
    return torch.nextafter(a, torch.full_like(a, float("inf"))) - a


def signed_err(x0, x1, lr, wd) -> float:
    """Median over weights of (|pred| - |obs|) in units of the observed ulp.
    Decreases monotonically with wd (more decay, smaller |pred|)."""
    p = replay(x0, lr, [wd])[0]
    # median: robust to sampled weights that did receive a late, tiny update
    return float(((p.abs() - x1.abs()) / _ulp(x1)).median())


def hits(x0, x1, lr, wds) -> np.ndarray:
    P = replay(x0, lr, wds)
    return (P == x1.unsqueeze(0)).sum(1).numpy()


def _edge(x0, x1, lr, w, h, sign, max_rel=0.5, iters=40):
    """Boundary of the plateau of wd values that reproduce h weights, on one side."""
    d = 1e-7
    while d < max_rel and hits(x0, x1, lr, [w * (1 + sign * d)])[0] >= h:
        d *= 2
    if d >= max_rel:
        return w * (1 + sign * max_rel)
    lo, hi = d / 2 if d > 1e-7 else 0.0, d
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if hits(x0, x1, lr, [w * (1 + sign * m)])[0] >= h:
            lo = m
        else:
            hi = m
    return w * (1 + sign * lo)


def bisect_wd(x0, x1, lr, wd_guess: float, iters: int = 40) -> float:
    """wd at which the replay's mean signed ulp error crosses zero (monotone in wd)."""
    lo, hi = wd_guess * 0.3, wd_guess * 3.0
    for _ in range(8):
        if signed_err(x0, x1, lr, lo) > 0:
            break
        lo *= 0.3
    for _ in range(8):
        if signed_err(x0, x1, lr, hi) < 0:
            break
        hi *= 3.0
    for _ in range(iters):
        m = 0.5 * (lo + hi)
        if signed_err(x0, x1, lr, m) > 0:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def fit_wd(x0, x1, lr, wd_guess: float, scan: int = 41, rel: float = 3e-3, iters: int = 40,
           interval: bool = False, start: float | None = None):
    """Continuous weight decay for one schedule (lr at the fitted peak).
    Returns (wd, hits, hits_fraction), plus the interval of wd values that give
    the same hits (same factor sequence) when interval=True; wd is then its midpoint."""
    if start is not None:
        return _scan_from(x0, x1, lr, start, scan, rel, interval)
    return _scan_from(x0, x1, lr, bisect_wd(x0, x1, lr, wd_guess, iters), scan, rel, interval)


def _scan_from(x0, x1, lr, w0, scan, rel, interval):
    best_w, best_h = w0, int(hits(x0, x1, lr, [w0])[0])
    pw = plateau_width(lr, w0)
    for width in (rel, rel / 16, rel / 256, 4 * pw, pw):
        grid = np.linspace(best_w * (1 - width), best_w * (1 + width), scan)
        h = hits(x0, x1, lr, grid)
        i = int(np.argmax(h))
        if h[i] > best_h:
            best_w, best_h = float(grid[i]), int(h[i])
    if not interval:
        return best_w, best_h, best_h / x1.numel()
    a_ = _edge(x0, x1, lr, best_w, best_h, -1)
    b_ = _edge(x0, x1, lr, best_w, best_h, +1)
    return 0.5 * (a_ + b_), best_h, best_h / x1.numel(), (a_, b_)


ROUNDNESS = (1, 5, 2, 2.5, 3, 4, 1.5, 6, 7, 8, 9)      # tie order: rounder first (as v1)


def round_in(interval):
    """A round value (m x 10^k) inside the interval, if any. When several are
    (a wide plateau, e.g. a constant schedule with a tiny factor), the roundest
    wins; the interval itself is reported alongside, since the weights cannot
    tell those values apart."""
    a_, b_ = interval
    mid = 0.5 * (a_ + b_)
    k = math.floor(math.log10(mid))
    c = [(ROUNDNESS.index(m), abs(m * 10.0 ** kk - mid), m * 10.0 ** kk)
         for m in ROUNDNESS for kk in (k - 1, k, k + 1) if a_ <= m * 10.0 ** kk <= b_]
    return min(c)[2] if c else None


def round_tiebreak(x0, x1, lr, wd: float, h: int) -> float:
    """Return a round value (m x 10^k) if it reproduces as many weights as wd."""
    k = math.floor(math.log10(wd))
    cands = sorted((m * 10.0 ** kk for m in ROUND_M for kk in (k - 1, k, k + 1)),
                   key=lambda v: abs(v / wd - 1))[:4]
    cands = [v for v in cands if abs(v / wd - 1) < 0.02]
    if not cands:
        return wd
    hh = hits(x0, x1, lr, cands)
    for v, hv in zip(cands, hh):
        if hv >= h:
            return float(v)
    return wd


def plateau_width(lr: np.ndarray, wd: float) -> float:
    """Relative change of wd that flips about one rounded factor: fp32 factors
    below 1 are spaced 2**-24, and the total decay is sum(lr) * wd."""
    return 2.0 ** -24 / max(float(np.sum(lr[1:])) * wd, 1e-30)


def screen_wd(x0, x1, lr, w0: float, span: float = 0.03, n: int = 65):
    """Decay for one schedule shape, and the fraction of weights it reproduces.

    The replay's mean signed ulp error is zero on the plateau of the true decay
    and monotone around it, so bisection lands next to it; a grid at half the
    plateau width (which shrinks as the total decay grows) then finds it. A
    coarse grid over +-span guards against a non-monotone error."""
    wb = bisect_wd(x0, x1, lr, w0, iters=40)
    pw = plateau_width(lr, wb)
    cands = [wb * (1 + np.linspace(-20 * pw, 20 * pw, 81)),
             wb * (1 + np.linspace(-span, span, n))]
    best_w, best_h = wb, -1
    for grid in cands:
        h = hits(x0, x1, lr, grid)
        i = int(np.argmax(h))
        if h[i] > best_h:
            best_w, best_h = float(grid[i]), int(h[i])
    grid = best_w * (1 + np.linspace(-pw, pw, 33))
    h = hits(x0, x1, lr, grid)
    i = int(np.argmax(h))
    if h[i] > best_h:
        best_w, best_h = float(grid[i]), int(h[i])
    return best_w, best_h / x1.numel()


def select(*args, **kw):
    """See _select. The replay is a long chain of tiny element-wise ops, so it
    runs single-threaded (threads only add overhead on a busy machine)."""
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return _select(*args, **kw)
    finally:
        torch.set_num_threads(nt)


def _select(pool, lr_fn, x0s, x1s, x0f, x1f, wd_of, min_conf: float = 0.95,
            max_evals: int = 900, log=None, dscore_fn=None, **_):
    """pool: list of recipe dicts (kind, N, warmup, peak, min_ratio).
    lr_fn(rec) -> lr array (index 0 unused). wd_of(rec, lr) -> initial wd guess.
    x0s/x1s: screening sample; x0f/x1f: final sample (flattened fp32).

    The screening score (fraction of sampled weights reproduced bit for bit,
    with the decay re-fitted for each shape) is graded: shapes near the true one
    reproduce part of the weights, far ones fewer. So after screening the pool,
    a pattern search in (steps, warm-up, floor) climbs it from the best few
    shapes; a local box then settles ties. The final pick is re-scored on the
    larger sample. Returns dict(recipe=..., wd=..., conf=..., recovered=bool, ...)."""
    seen, scored = {}, []

    def key(r):
        return (r["kind"], int(r["N"]), int(r["warmup"]), float(r.get("min_ratio", 0.0)))

    def ev(r):
        k = key(r)
        if k in seen:
            return seen[k]
        if r["N"] < 2 or r["warmup"] < 0 or r["warmup"] >= r["N"] or len(seen) >= max_evals:
            return None
        lr = lr_fn(r)
        try:
            w, fr = screen_wd(x0s, x1s, lr, max(wd_of(r, lr), 1e-12))
        except Exception as e:  # pragma: no cover - degenerate schedules
            if log:
                log(f"skip {k}: {e}")
            seen[k] = None
            return None
        rec = dict(r, wd=w, screen_frac=fr)
        seen[k] = rec
        scored.append(rec)
        return rec

    for r in pool:
        ev(r)
    # v1 compatibility: the move fit's own best (pool[0]) gets the same +-3 step,
    # +-1 warm-up neighbourhood v1 searched by replay, so v2 never does worse there
    if pool:
        p0 = pool[0]
        for dN in range(-3, 4):
            for dW in (-1, 0, 1):
                ev(dict(p0, N=p0["N"] + dN, warmup=max(0, p0["warmup"] + dW)))
    scored.sort(key=lambda r: -r["screen_frac"])
    if scored and scored[0]["screen_frac"] < 0.5:
        # the move fit may be far off in steps (n05: 848 for 500): scan the step
        # count by replay over 0.4-1.6x of the best few shapes' steps
        for r0 in [r for r in scored[:3]]:
            for f in np.exp(np.linspace(np.log(0.4), np.log(1.6), 36)):
                N = int(round(r0["N"] * f))
                for wf in (0.0, 0.02, 0.05, 0.1):
                    ev(dict(r0, N=N, warmup=int(round(wf * N))))
        scored.sort(key=lambda r: -r["screen_frac"])
    starts, kinds_seen = [], set()
    for r in scored:
        kk = (r["kind"], r["min_ratio"])
        if kk not in kinds_seen and r["screen_frac"] > 0:
            kinds_seen.add(kk)
            starts.append(r)
        if len(starts) >= 3:
            break
    for cur in starts:
        for step_N, step_W in ((64, 16), (32, 8), (16, 8), (8, 4), (4, 2), (2, 1), (1, 1)):
            moved = True
            while moved and cur["screen_frac"] < 0.999:
                moved = False
                for dN, dW in ((step_N, 0), (-step_N, 0), (0, step_W), (0, -step_W),
                               (step_N, step_W), (-step_N, -step_W), (step_N, -step_W), (-step_N, step_W)):
                    r = ev(dict(cur, N=cur["N"] + dN, warmup=max(0, cur["warmup"] + dW)))
                    if r and r["screen_frac"] > cur["screen_frac"]:
                        cur, moved = r, True
        if cur["kind"] in ("cosine_min", "wsd"):
            ev(dict(cur, min_ratio=0.0 if cur["min_ratio"] else 0.1))
    scored.sort(key=lambda r: -r["screen_frac"])
    best = scored[0]
    if best["screen_frac"] < 0.95:
        # warm-up can sit in a separate basin (n05: 9 read for 31): scan it fully
        # at the best step count, then settle the box below
        for W in range(0, max(2, min(best["N"] // 4, 120)) + 1):
            ev(dict(best, warmup=W))
        scored.sort(key=lambda r: -r["screen_frac"])
        best = scored[0]
    if best["screen_frac"] < 0.999:
        for dN in range(-3, 4):
            for dW in range(-3, 4):
                ev(dict(best, N=best["N"] + dN, warmup=max(0, best["warmup"] + dW)))
        scored.sort(key=lambda r: -r["screen_frac"])
        best = scored[0]
    # shapes the screening sample cannot tell apart: score on the full sample;
    # remaining ties go to the move fit (dscore_fn)
    top = [r for r in scored if r["screen_frac"] >= best["screen_frac"] - 1e-9][:12]
    if len(top) < 3:
        top = scored[:3]
    finals = []
    for r in top:
        lr = lr_fn(r)
        w, fr = screen_wd(x0f, x1f, lr, r["wd"], span=0.01, n=41)
        finals.append((round(fr * x1f.numel()), -(dscore_fn(r) if dscore_fn else 0.0), r, w, fr, lr))
    finals.sort(key=lambda z: (z[0], z[1]), reverse=True)
    h, _, best, w, fr, lr = finals[0]
    w, h, fr, wint = fit_wd(x0f, x1f, lr, w, interval=True, start=w)
    ties = [dict(kind=z[2]["kind"], N=z[2]["N"], warmup=z[2]["warmup"], min_ratio=z[2]["min_ratio"])
            for z in finals if z[0] == finals[0][0]]
    rv = round_in(wint)
    w2 = rv if (rv is not None and hits(x0f, x1f, lr, [rv])[0] >= h) else w
    out = {k: best[k] for k in ("kind", "N", "warmup", "peak", "min_ratio")}
    return dict(recipe=out, wd=w2, wd_continuous=w, wd_interval=list(wint), hits=int(h), sample=int(x1f.numel()),
                conf=fr, recovered=bool(fr >= min_conf), n_candidates=len(scored), replay_ties=ties,
                screened=[{k: r[k] for k in ("kind", "N", "warmup", "min_ratio", "wd", "screen_frac")}
                          for r in scored[:8]])


def candidate_pool(Dq, logP, procs: int = 32, top: int = 60, sigma: float = 6e-4, coarse: float = 8.0,
                   N_range=(0.4, 4.0), n_grid=90):
    """Shapes for the replay to arbitrate: the `top` best distinct (kind, N,
    warm-up, floor) of the move fit's coarse grid, each refined at the fine
    tolerance over +-6% in N. Uses blind's scorer; does not change blind.fit_recipe."""
    from . import blind
    Dq = np.asarray(Dq, np.float64)
    logobs = np.log(Dq[Dq > 0])
    wd = ("logP", float(logP)) if (logP is not None and logP < 0) else 0.0
    so = np.sort(logobs)
    N_hint = int(1 + (np.diff(so) > 1.5e-3).sum())
    Ns = sorted({max(10, int(round(N_hint * f))) for f in
                 np.exp(np.linspace(np.log(N_range[0]), np.log(N_range[1]), n_grid))} |
                {int(round(x)) for x in np.exp(np.linspace(np.log(20), np.log(20000), 70))})
    jobs = [(logobs, kind, N, W, mr, wd, sigma * coarse)
            for N in Ns for kind in blind.KINDS
            for mr in ((0.1, 0.0) if kind in ("cosine_min", "wsd") else (0.0,))
            for W in blind._warmups(N)]
    cands = sorted(blind._pmap(jobs, procs), key=lambda r: r.score)
    keys, jobs2 = [], []
    for r in cands:
        k = (r.kind, r.min_ratio, r.N)
        if k in keys:
            continue
        keys.append(k)
        for N in range(max(10, int(r.N * 0.94)), int(r.N * 1.06) + 2):
            for W in {r.warmup, int(round(r.warmup * N / r.N))}:
                jobs2.append((logobs, r.kind, N, W, r.min_ratio, wd, sigma))
        if len(keys) >= top // 4:
            break
    ref = sorted(blind._pmap(jobs2, procs), key=lambda r: r.score)
    out, seen = [], set()
    for r in ref + cands:
        k = (r.kind, r.N, r.warmup, r.min_ratio)
        if k in seen:
            continue
        seen.add(k)
        out.append(dict(kind=r.kind, N=r.N, warmup=r.warmup, peak=r.peak, min_ratio=r.min_ratio, dscore=r.score))
        if len(out) >= top:
            break
    return out
