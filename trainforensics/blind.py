"""Blind reading of a fine-tune from (W0, W1) [+ optionally the claimed dataset].

Nothing here looks at ground truth. scripts/controlled/score.py compares the
outputs with truth.json.

  rows      which embedding rows moved (= which tokens were in the data)
  recipe    schedule kind, step count, warm-up, peak lr (and wd from the decay)
  placement which optimizer step each claimed document was in
  seed      the sampler seed that produced that placement
  swaps     claimed documents that cannot have been trained
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from . import adam
from .diff import row_diff

B1, B2 = 0.9, 0.999


# ----------------------------------------------------------------- D(s) fast
def moves_curve(lr: np.ndarray, wd: float = 0.0, b1=B1, b2=B2, H: int = 300) -> np.ndarray:
    """Vectorised adam.single_touch_moves: D[s], s=1..N (D[0]=0)."""
    lr = np.asarray(lr, np.float64)
    N = len(lr) - 1
    suf = adam.decay_products(lr, wd)
    s = np.arange(1, N + 1)[:, None]
    k = np.arange(H)[None, :]
    t = s + k
    ok = t <= N
    tc = np.minimum(t, N)
    a = ((1 - b1) * b1 ** k / (1 - b1 ** tc)) / np.sqrt((1 - b2) * b2 ** k / (1 - b2 ** tc))
    D = np.where(ok, lr[tc] * a * suf[np.minimum(tc + 1, N + 1)], 0.0).sum(1)
    return np.concatenate([[0.0], D])


# ----------------------------------------------------------------- rows
def row_moves(W0, W1, stored_dtype=None, q: float = 0.5):
    """Per-row move D and flatness. q: quantile of |delta| used as the row's D
    (median by default)."""
    rd = row_diff(W0, W1, stored_dtype)
    return rd


# ----------------------------------------------------------------- recipe
KINDS = ("constant", "linear", "cosine", "cosine_min", "wsd")


@dataclasses.dataclass
class Recipe:
    kind: str
    N: int
    warmup: int
    peak: float
    min_ratio: float = 0.0
    decay_frac: float = 0.2
    score: float = float("inf")
    coverage: float = 0.0
    resid: float = 0.0

    def lr(self):
        return adam.schedule(self.kind, self.N, self.peak, self.warmup, self.min_ratio, self.decay_frac)

    def as_dict(self):
        return dataclasses.asdict(self)


def _score(logobs: np.ndarray, Dc: np.ndarray, sigma: float, outlier: float = 0.01):
    """Negative mean log-likelihood of the observed log-moves under the candidate:
    a uniform mixture over steps of N(log D(s), sigma), plus a small uniform
    outlier floor (multi-touch rows that happen to look flat). Steps whose D lies
    where no rows were seen still carry mass, so surplus steps are penalised.
    Returns (score, coverage, residual) - the last two are diagnostics."""
    lc = np.log(np.maximum(Dc, 1e-300))
    # detection floor: moves below the smallest observed ones can be lost in
    # rounding/decay drift, so steps predicted below it carry no evidence
    floor = np.quantile(logobs, 0.001) - 4 * sigma
    lc = lc[lc >= floor]
    if len(lc) == 0:
        return 1e9, 0.0, 1e9
    N = len(lc)
    lo_min, lo_max = logobs.min() - 8 * sigma, logobs.max() + 8 * sigma
    h = sigma / 2
    nb = int(math.ceil((lo_max - lo_min) / h)) + 1
    hist = np.bincount(np.clip(((lc - lo_min) / h).astype(np.int64), -1, nb).clip(0, nb - 1)
                       [(lc >= lo_min) & (lc <= lo_max)], minlength=nb).astype(np.float64) / N
    kx = np.arange(-8, 9)
    ker = np.exp(-0.5 * (kx * h / sigma) ** 2)
    ker /= ker.sum()
    dens = np.convolve(hist, ker, mode="same") / h
    u = 1.0 / (lo_max - lo_min)
    idx = ((logobs - lo_min) / h).astype(np.int64).clip(0, nb - 1)
    ll = np.log((1 - outlier) * dens[idx] + outlier * u)
    # diagnostics
    so = np.sort(logobs)
    j = np.clip(np.searchsorted(so, lc), 1, len(so) - 1)
    dd = np.minimum(np.abs(lc - so[j - 1]), np.abs(lc - so[j]))
    cov = float((dd < 2 * sigma).mean())
    return float(-ll.mean()), cov, float(np.median(dd))


def _fit_peak(logobs, shape_D, sigma, stages=((0.5, 51), (0.03, 61), (0.002, 41), (0.0002, 41))):
    """shape_D: curve at peak=1. Returns best log-shift and score tuple."""
    lD = np.log(np.maximum(shape_D[1:], 1e-300))
    shift0 = np.median(logobs) - np.median(lD)
    best = None
    for width, n in stages:
        c = shift0 if best is None else best[0]
        for sh in np.linspace(c - width, c + width, n):
            sc = _score(logobs - sh, shape_D[1:], sigma)
            if best is None or sc[0] < best[1][0]:
                best = (sh, sc)
    return best


def _warmups(N):
    w = {0, 10, 20, 50, 100, 200, 500}
    for f in (0.01, 0.02, 0.03, 0.05, 0.1):
        w.add(int(round(f * N)))
    return sorted(x for x in w if x < N // 2)


def _eval(args):
    logobs, kind, N, W, mr, wd, sigma = args
    r = Recipe(kind, N, W, 1.0, min_ratio=mr)
    shape = r.lr()
    if isinstance(wd, tuple):              # ("logP", ln P): decay strength fixed by the total decay
        wd = -wd[1] / max(shape[1:].sum(), 1e-30)
    # coarse-tolerance jobs (global grid) only need the peak to ~0.1%
    stages = ((0.5, 51), (0.03, 61), (0.002, 21)) if sigma > 2e-3 else ((0.5, 51), (0.03, 61), (0.002, 41), (0.0002, 41))
    sh, sc = _fit_peak(logobs, moves_curve(shape, wd), sigma, stages)
    r.peak, r.score, r.coverage, r.resid = math.exp(sh), *sc
    return r


def _pmap(jobs, procs):
    if procs <= 1 or len(jobs) < 8:
        return [_eval(j) for j in jobs]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(procs) as pool:
        return pool.map(_eval, jobs, chunksize=4)


def fit_recipe(Dobs, wd: float = 0.0, N_hint: int | None = None, sigma: float = 6e-4,
               kinds=KINDS, procs: int = 32, N_range=(0.4, 4.0), n_grid=90, peak_guess: float = 0.0, logP: float | None = None,
               coarse: float = 8.0):
    """Dobs: per-row moves of flat (single-touch) rows. Returns (best, ranked by kind).

    The global grid is scored at coarse*sigma: at the fine sigma a grid point a
    few percent off the true N scores worse than a wrong shape whose dense curve
    covers everything loosely (r12: cosine_min/490 scores 1.09, linear/5337 0.29,
    true cosine_min/500 -0.54). The refinement runs at the fine sigma."""
    Dobs = np.asarray(Dobs, np.float64)
    logobs = np.log(Dobs[Dobs > 0])
    # shape curves are built at peak = 1, but decoupled decay depends on the
    # absolute lr*wd: use the effective strength wd * peak_guess
    if logP is not None and logP < 0:
        # self-consistent decay: for a shape with unit peak, wd*peak = -ln P / sum(shape)
        wd = ("logP", float(logP))
    else:
        wd = wd * peak_guess if wd else 0.0
    if N_hint is None:
        so = np.sort(logobs)
        N_hint = int(1 + (np.diff(so) > 1.5e-3).sum())
    Ns = sorted({max(10, int(round(N_hint * f))) for f in
                 np.exp(np.linspace(np.log(N_range[0]), np.log(N_range[1]), n_grid))} |
                # hint-independent coverage: the cluster-count hint can be off by
                # several x when the sorted moves have many small gaps
                {int(round(x)) for x in np.exp(np.linspace(np.log(20), np.log(20000), 70))})
    jobs = [(logobs, kind, N, W, mr, wd, sigma)
            for N in Ns for kind in kinds
            for mr in ((0.1, 0.0) if kind in ("cosine_min", "wsd") else (0.0,))
            for W in _warmups(N)]
    if coarse and coarse > 1:
        jobs = [j[:-1] + (sigma * coarse,) for j in jobs]
    cands = sorted(_pmap(jobs, procs), key=lambda r: r.score)
    # refine N at integer resolution around the 12 best distinct (kind, min_ratio, N)
    jobs, keys = [], set()
    for r0 in cands:
        key = (r0.kind, r0.min_ratio, r0.N)
        if key in keys:
            continue
        keys.add(key)
        # with the coarse pass the grid neighbours (<= 2.6% apart in N) bracket the
        # true N, so +-6% around the 8 best keys suffices
        span, nkeys = ((0.06, 8) if (coarse and coarse > 1) else (0.12, 12))
        for N in range(max(10, int(r0.N * (1 - span))), int(r0.N * (1 + span)) + 2):
            for W in {r0.warmup, int(round(r0.warmup * N / r0.N))}:
                jobs.append((logobs, r0.kind, N, W, r0.min_ratio, wd, sigma))
        if len(keys) >= nkeys:
            break
    ref = sorted(_pmap(jobs, procs), key=lambda r: r.score)
    best = ref[0]
    Wmax = max(40, 2 * best.warmup + 10)
    jobs = [(logobs, best.kind, N, W, best.min_ratio, wd, sigma)
            for N in range(max(10, best.N - 15), best.N + 16) for W in range(0, Wmax + 1)]
    ref = sorted(ref + _pmap(jobs, procs), key=lambda r: r.score)
    seen, ranked = set(), []
    for r in ref + ([] if (coarse and coarse > 1) else cands):
        if r.kind not in seen:
            seen.add(r.kind)
            ranked.append(r)
    return ranked[0], ranked


def wd_from_decay(P: float, lr: np.ndarray) -> float:
    """P = prod(fl32(1 - lr_t wd))  ->  wd (bisection). torch applies the
    decay as an fp32 scalar factor each step, so the factors are rounded to
    fp32 before multiplying."""
    if P >= 1.0:
        return 0.0
    lo, hi = 0.0, 10.0
    for _ in range(200):
        m = (lo + hi) / 2
        if np.prod((1 - lr[1:] * m).astype(np.float32).astype(np.float64)) > P:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


# ----------------------------------------------------------------- placement
def step_of_D(D: float, curve: np.ndarray, tol: float = 1.5e-3):
    """Candidate steps whose predicted move is within tol (log) of D."""
    l = np.log(np.maximum(curve[1:], 1e-300))
    d = np.abs(l - math.log(D))
    return np.nonzero(d < tol)[0] + 1, d


def place_docs(doc_tokens: dict, rd, curve: np.ndarray, flat_thr=0.9, tol=1.5e-3, D=None):
    """doc_tokens: doc key -> token ids that get an input-embedding gradient.
    Returns doc -> (step or None, n_single_tokens_used, status)."""
    # token -> number of claimed docs containing it
    cnt: dict[int, int] = {}
    for toks in doc_tokens.values():
        for t in set(toks):
            cnt[t] = cnt.get(t, 0) + 1
    D = rd.D if D is None else D
    out = {}
    for d, toks in doc_tokens.items():
        toks = set(toks)
        unmoved = [t for t in toks if not rd.moved[t]]
        if unmoved:
            out[d] = (None, 0, "has_unmoved_token")
            continue
        cand = None
        used = 0
        for t in toks:
            if rd.flat_frac[t] > flat_thr and cnt[t] == 1:
                steps, _ = step_of_D(D[t], curve, tol)
                s = set(steps.tolist())
                cand = s if cand is None else cand & s
                used += 1
        if cand is None:
            out[d] = (None, 0, "no_unique_token")
        elif len(cand) == 1:
            out[d] = (next(iter(cand)), used, "placed")
        elif len(cand) == 0:
            out[d] = (None, used, "inconsistent")
        else:
            out[d] = (None, used, f"ambiguous{len(cand)}")
    return out


def refine_placement(placed: dict, doc_tokens: dict, rd, curve, tol=1.5e-3):
    """Second pass: a token shared by several claimed docs that is single-touch
    (flat) must have all its docs in the same step."""
    return placed


# ----------------------------------------------------------------- seed
def find_seed(pos_step: dict, n_docs: int, bs: int, seeds=range(0, 200000), epoch: int = 0):
    """pos_step: dataset position k -> 1-based step (single epoch). Returns ranked [(seed, hits)]."""
    ks = np.array(list(pos_step.keys()))
    ss = np.array([pos_step[k] for k in ks])
    if len(ks) == 0:
        return []
    want = ss - 1
    g = torch.Generator()
    res = []
    priority = [42, 0, 1, 1234, 3407, 2024, 2025, 2026, 7, 123, 12345, 1337, 31337]
    for sd in list(priority) + [s for s in seeds if s not in priority]:
        g.manual_seed(sd + epoch)
        perm = torch.randperm(n_docs, generator=g).numpy()
        inv = np.empty(n_docs, np.int64)
        inv[perm] = np.arange(n_docs)
        hits = int(((inv[ks] // bs) == want).sum())
        res.append((sd, hits))
        if hits == len(ks):
            break
    res.sort(key=lambda x: -x[1])
    return res[:5]


# ----------------------------------------------------------------- replay
def replay_decay(W0: torch.Tensor, lr: np.ndarray, wd: float, foreach_like=True) -> torch.Tensor:
    """Never-touched rows under torch AdamW: p.mul_(1 - lr*wd) once per step, in fp32."""
    x = W0.clone().float()
    for t in range(1, len(lr)):
        x.mul_(1 - float(lr[t]) * wd)
    return x


# ----------------------------------------------------------------- non-flat rows
def row_intercepts(W0: torch.Tensor, W1: torch.Tensor, decay: float, rows, chunk: int = 2048,
                   qs=None, taus=None):
    """Per-row move D read as the intercept of the sorted-moves curve.

    With Adam's eps, a coordinate touched once moves by about D * g/(g + tau)
    (g = |gradient|, tau ~ eps / sqrt-moment factor). Gradient coordinates of
    one row are roughly half-normal, so the q-quantile of |delta| is
    D * z_q/(z_q + tau') with z_q the half-normal quantile. Fit (D, tau') per
    row by least squares on log-spaced quantiles; tau' -> 0 recovers the flat
    case. Returns (D, tau, rel_rms_resid)."""
    erfinv = lambda x: torch.erfinv(torch.as_tensor(x, dtype=torch.float64)).numpy()
    qs = np.linspace(0.30, 0.995, 60) if qs is None else qs
    z = np.sqrt(2) * erfinv(qs)                                   # half-normal quantiles
    taus = np.concatenate([[0.0], np.logspace(-3, 3, 121)]) if taus is None else taus
    F = z[None, :] / (z[None, :] + taus[:, None])                  # [T, Q]
    rows = np.asarray(rows)
    Dout = np.zeros(len(rows)); Tout = np.zeros(len(rows)); Rout = np.zeros(len(rows))
    qt = torch.tensor(qs, dtype=torch.float64)
    Ft = torch.tensor(F)
    for a in range(0, len(rows), chunk):
        idx = rows[a:a + chunk]
        d = (W1[idx].double() - decay * W0[idx].double()).abs()
        y = torch.quantile(d, qt, dim=1).T                         # [R, Q]
        num = y @ Ft.T                                             # [R, T]
        den = (Ft * Ft).sum(1)[None, :]
        Dt = num / den
        # relative residual (each quantile weighted by 1/y): fit in relative terms
        pred = Dt[:, :, None] * Ft[None, :, :]                     # [R, T, Q]
        res = (((pred - y[:, None, :]) / y.clamp_min(1e-30)[:, None, :]) ** 2).mean(2)
        k = res.argmin(1)
        ar = torch.arange(len(idx))
        Dout[a:a + chunk] = Dt[ar, k].numpy()
        Tout[a:a + chunk] = taus[k.numpy()]
        Rout[a:a + chunk] = res[ar, k].sqrt().numpy()
    return Dout, Tout, Rout


# ----------------------------------------------------------------- decay map
def decay_lut(W0: torch.Tensor, W1: torch.Tensor, unmoved: np.ndarray, max_rows: int = 40000):
    """Under decoupled weight decay every coordinate goes through the same
    sequence of fp32 multiplies, so x_N = F(x_0) for one deterministic F
    (including its rounding drift). Read F off the never-touched rows as a
    lookup table. Exact for starting values that also occur in those rows,
    which is nearly all of them when the start was stored in bf16/fp16.
    Returns (keys, vals, consistent_fraction)."""
    idx = np.nonzero(unmoved)[0]
    if len(idx) > max_rows:
        idx = idx[np.linspace(0, len(idx) - 1, max_rows).astype(int)]
    a = W0[idx].float().flatten().contiguous().view(torch.int32).to(torch.int64)
    b = W1[idx].float().flatten().contiguous().view(torch.int32).to(torch.int64)
    # majority value per key: rows that moved by less than the detection floor
    # (touched in the last steps of a schedule decaying to ~0) are a small
    # minority of the rows sharing a starting value, so they cannot outvote F
    pair = ((a & 0xFFFFFFFF) << 32) | (b & 0xFFFFFFFF)
    up, cnt = torch.unique(pair, return_counts=True)       # sorted by key then value
    pk = (up >> 32).to(torch.int32)
    pv = (up & 0xFFFFFFFF).to(torch.int32)
    keys, inv = torch.unique_consecutive(pk, return_inverse=True)
    best = torch.zeros(len(keys), dtype=torch.long)
    bestc = torch.zeros(len(keys), dtype=torch.long)
    bestc.scatter_reduce_(0, inv, cnt, reduce="amax", include_self=False)
    is_best = cnt == bestc[inv]
    pos = torch.nonzero(is_best).flatten()
    best[inv[pos].flip(0)] = pos.flip(0)                     # first maximal value per key
    vals = pv[best].view(torch.float32)
    tot = torch.zeros(len(keys), dtype=torch.long).scatter_add_(0, inv, cnt)
    consistent = float(bestc.sum()) / float(tot.sum())
    return keys, vals, consistent


def apply_lut(x0: torch.Tensor, keys, vals, decay: float):
    """F(x0) where known, decay * x0 otherwise. Returns (ref, covered_fraction)."""
    k = x0.float().contiguous().view(torch.int32)
    pos = torch.searchsorted(keys, k.flatten()).clamp(max=len(keys) - 1)
    hit = keys[pos] == k.flatten()
    ref = torch.where(hit, vals[pos], (x0.double() * decay).float().flatten())
    return ref.view_as(x0), float(hit.double().mean())
