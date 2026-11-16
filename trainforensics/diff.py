"""Row-level diff of two embedding matrices (start W0, end W1).

Every row ends up as  W1 = P * W0 + Delta,  where P is the product of the
weight-decay factors over the run (1 when there is no decoupled decay) and
Delta is what the gradients did. Rows of tokens that never had a gradient
have Delta = 0. Rows touched in exactly one optimizer step have |Delta_i|
equal to one number D in nearly every coordinate (Adam's sign-only update).
"""
from __future__ import annotations

import dataclasses
import numpy as np
import torch


def _f64(x):
    return x.detach().to(torch.float64)


def ulp(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Spacing of representable values of `dtype` at |x| (as float64)."""
    mant = {torch.float32: 23, torch.bfloat16: 7, torch.float16: 10}[dtype]
    a = x.abs().to(torch.float64).clamp_min(1e-30)
    e = torch.floor(torch.log2(a))
    if dtype == torch.float16:
        e = e.clamp_min(-14)
    elif dtype == torch.bfloat16:
        e = e.clamp_min(-126)
    return torch.pow(2.0, e - mant)


@dataclasses.dataclass
class RowDiff:
    identical: np.ndarray        # bool [V]: W1 row bitwise equal to W0 row
    moved: np.ndarray            # bool [V]: Delta clearly non-zero
    decay: float                 # P, fitted on unmoved rows (1.0 if all identical)
    D: np.ndarray                # median |Delta| per row (0 for unmoved)
    cv: np.ndarray               # coefficient of variation of |Delta| within the row
    flat_frac: np.ndarray        # fraction of coords with |Delta| within 1% of D
    resid: np.ndarray            # relative residual ||W1 - P W0|| / ||W0||

    def summary(self):
        return dict(rows=int(len(self.moved)), identical=int(self.identical.sum()),
                    moved=int(self.moved.sum()), decay=self.decay,
                    flat_rows=int(((self.flat_frac > 0.9) & self.moved).sum()))


def row_diff(W0: torch.Tensor, W1: torch.Tensor, stored_dtype: torch.dtype | None = None,
             chunk: int = 16384, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> RowDiff:
    """W0: start (as released / as loaded), W1: end, both [V, d] (any float dtype).
    stored_dtype: dtype W1 was published in (sets the rounding tolerance)."""
    assert W0.shape == W1.shape, (W0.shape, W1.shape)
    stored_dtype = stored_dtype or W1.dtype
    V = W0.shape[0]
    identical = np.zeros(V, bool)
    ratio = np.zeros(V)
    resid = np.zeros(V)
    for a in range(0, V, chunk):
        w0 = W0[a:a + chunk].to(device)
        w1 = W1[a:a + chunk].to(device)
        identical[a:a + chunk] = (w0.to(w1.dtype) == w1).all(1).cpu().numpy() if w0.dtype == w1.dtype else \
            (w0.double() == w1.double()).all(1).cpu().numpy()
        x0, x1 = _f64(w0), _f64(w1)
        n0 = (x0 * x0).sum(1).clamp_min(1e-300)
        r = (x0 * x1).sum(1) / n0
        ratio[a:a + chunk] = r.cpu().numpy()
        resid[a:a + chunk] = ((x1 - r[:, None] * x0).norm(dim=1) / n0.sqrt()).cpu().numpy()
    # unmoved rows: residual at the rounding floor. Rounding of one value to
    # `stored_dtype` leaves a relative error ~ 2^-(mant+1)/sqrt(3); moved rows sit
    # orders of magnitude above. Threshold at 8x the rounding floor.
    mant = {torch.float32: 23, torch.bfloat16: 7, torch.float16: 10}[stored_dtype]
    floor = 2.0 ** -(mant + 1) / np.sqrt(3)
    if stored_dtype == torch.float32:
        floor = max(floor, 1e-6)       # many fp32 multiplies accumulate rounding
    thr = 8 * floor
    # Decoupled decay applied over many fp32 steps leaves a deterministic,
    # mantissa-dependent rounding drift (up to ~N/2 ulp) on never-touched rows,
    # which can exceed the single-rounding floor. When log-residuals are
    # clearly bimodal, split them where they separate (Otsu in log10).
    lr_ = np.log10(np.clip(resid, 1e-12, None))
    nz = lr_[resid > 0]
    if len(nz) > 100:
        hist, edges = np.histogram(nz, bins=200)
        p = hist / hist.sum()
        mids = (edges[:-1] + edges[1:]) / 2
        w0 = np.cumsum(p)
        m0 = np.cumsum(p * mids)
        mt = m0[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            sb = (mt * w0 - m0) ** 2 / (w0 * (1 - w0))
        k = int(np.nanargmax(sb))
        cut = edges[k + 1]
        lo_m = nz[nz < cut].mean() if (nz < cut).any() else cut
        hi_m = nz[nz >= cut].mean() if (nz >= cut).any() else cut
        if hi_m - lo_m > 1.5 and cut <= -3 and 10 ** cut > thr:
            thr = 10 ** cut
    cand = resid < thr
    if cand.sum() > 0:
        decay = float(np.median(ratio[cand]))
    else:
        decay = 1.0
    if identical.sum() > 0.5 * cand.sum() and identical.sum() > 0:
        decay = 1.0
    D = np.zeros(V)
    cv = np.zeros(V)
    ff = np.zeros(V)
    moved = ~cand
    if decay == 1.0 and stored_dtype == W0.dtype == W1.dtype:
        # no decoupled decay and same storage: any bit change is a gradient
        moved = ~identical
    for a in range(0, V, chunk):
        w0 = _f64(W0[a:a + chunk].to(device))
        w1 = _f64(W1[a:a + chunk].to(device))
        d = (w1 - decay * w0).abs()
        med = d.median(dim=1).values
        D[a:a + chunk] = med.cpu().numpy()
        cv[a:a + chunk] = (d.std(1) / d.mean(1).clamp_min(1e-300)).cpu().numpy()
        ff[a:a + chunk] = ((d - med[:, None]).abs() <= 0.01 * med[:, None]).double().mean(1).cpu().numpy()
    D[~moved] = 0
    return RowDiff(identical, moved, decay, D, cv, ff, resid)


def _neighbours(q: torch.Tensor, dtype: torch.dtype):
    """Next representable values above and below q in `dtype` (float64)."""
    it = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[dtype]
    qq = q.to(dtype)
    bits = qq.contiguous().view(it).to(torch.int64)
    neg = qq.double() < 0
    up = torch.where(neg, bits - 1, bits + 1)
    dn = torch.where(neg, bits + 1, bits - 1)
    zero = qq.double() == 0
    to = lambda b: b.to(it).view(dtype).double()
    upv, dnv = to(up), to(dn)
    tiny = torch.finfo(dtype).smallest_normal if dtype != torch.float16 else 2.0 ** -24
    upv = torch.where(zero, torch.full_like(upv, tiny), upv)
    dnv = torch.where(zero, torch.full_like(dnv, -tiny), dnv)
    return upv, dnv


def interval_move(w0: torch.Tensor, w1q: torch.Tensor, dtype: torch.dtype, decay: float = 1.0):
    """16-bit release: for a single-touch row, the fp32 end value was
    x_i = decay*w0_i - sign_i * D and then rounded to `dtype` -> w1q_i.
    Each coordinate confines D to an interval; intersect them. Returns (lo, hi).
    The rounding interval of q is bounded by the midpoints to its two neighbours,
    which differ below a power of two (asymmetric spacing)."""
    x0 = _f64(w0) * decay
    q = _f64(w1q)
    upv, dnv = _neighbours(w1q, dtype)
    lo_x, hi_x = (q + dnv) / 2, (q + upv) / 2   # fp32 value lay in [lo_x, hi_x]
    s = torch.sign(x0 - q)
    keep = s != 0
    a = torch.where(s > 0, x0 - hi_x, lo_x - x0)
    b = torch.where(s > 0, x0 - lo_x, hi_x - x0)
    a, b = a[keep], b[keep]
    if a.numel() == 0:
        return 0.0, float("inf")
    return float(a.max().clamp_min(0)), float(b.min())


def round_to(x64: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """float64 -> float32 -> dtype (the path a fp32 master weight takes to a 16-bit release)."""
    return x64.to(torch.float32).to(dtype)


def fit_decay_16(W0: torch.Tensor, W1q: torch.Tensor, dtype: torch.dtype, sample: int = 4096,
                 seed: int = 0) -> tuple[float, float]:
    """Find P maximising the number of rows with round(P*W0) == W1q exactly.
    Returns (P, fraction of sampled rows matched).

    All-zero parent rows match round(P*0) == 0 for every P, so they carry no
    information about P and are left out of the sample (with them in, every P
    scored > 0 and the search never left P = 1). Both the P = 1 neighbourhood and
    the least-squares candidate P0 are always searched; the better one is kept."""
    pool = torch.nonzero((W0 != 0).any(1)).flatten()
    if pool.numel() == 0:
        return 1.0, float("nan")
    g = torch.Generator().manual_seed(seed)
    idx = pool[torch.randperm(pool.numel(), generator=g)[:sample]]
    x0 = W0[idx].double()
    q = W1q[idx].to(dtype)

    def rows_matched(P):
        return float(((round_to(x0 * P, dtype) == q).all(1)).double().mean())

    def coords_matched(P):   # smooth objective: a P off by 1e-5 already breaks most whole rows
        return float((round_to(x0 * P, dtype) == q).double().mean())

    xr = (x0 * W1q[idx].double()).sum(1) / (x0 * x0).sum(1)
    res = (W1q[idx].double() - xr[:, None] * x0).norm(dim=1) / x0.norm(dim=1)
    P0 = float(xr[res <= res.quantile(0.2)].median())
    best_P, best_s = 1.0, coords_matched(1.0)
    for c in (1.0, P0):
        for width, n in ((3e-3, 301), (3e-5, 301), (3e-7, 301)):
            lp, ls = c, coords_matched(c)
            for P in np.linspace(c * (1 - width), c * (1 + width), n):
                s_ = coords_matched(float(P))
                if s_ > ls:
                    lp, ls = float(P), s_
            c = lp
            if ls > best_s:
                best_P, best_s = lp, ls
    return best_P, rows_matched(best_P)


def moved_rows_16(W0: torch.Tensor, W1q: torch.Tensor, dtype: torch.dtype, P: float, chunk: int = 16384):
    """Number of coordinates per row that differ from the decayed, rounded start."""
    out = np.zeros(W0.shape[0], dtype=np.int64)
    for a in range(0, W0.shape[0], chunk):
        pred = round_to(W0[a:a + chunk].double() * P, dtype)
        out[a:a + chunk] = (pred != W1q[a:a + chunk].to(dtype)).sum(1).numpy()
    return out
