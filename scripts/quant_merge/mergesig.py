"""Merge signatures in a 16-bit embedding matrix.

analyse(M, O, parents) compares a (possibly merged) child M with its base O.
Blind features use M and O only; with candidate parents it also recovers the
merge weights and checks that M's token record is the union of the parents'.
"""
from __future__ import annotations

import torch


def okey(x: torch.Tensor) -> torch.Tensor:
    """Monotone integer key of a 16-bit float tensor: adjacent values differ by 1."""
    b = x.contiguous().view(torch.int16).to(torch.int32)
    mag = b & 0x7FFF
    return torch.where(b < 0, -mag, mag)


def blind(Mx: torch.Tensor, O: torch.Tensor):
    n = min(Mx.shape[0], O.shape[0])
    Mx, O = Mx[:n], O[:n]
    moved = (Mx != O).any(1)
    steps = (okey(Mx) - okey(O)).abs()
    r = dict(rows=n, moved_rows=int(moved.sum()), identical_rows=int((~moved).sum()))
    if O.dtype == torch.bfloat16:
        rt = O.to(torch.float16).to(O.dtype)
        r["moved_rows_explained_by_fp16_roundtrip"] = int((moved & (Mx == rt).all(1)).sum())
    if moved.sum() == 0:
        return r, moved
    S = steps[moved]
    eq = S == 0
    r["moved_coords_equal_frac"] = float(eq.float().mean())
    ch = S[~eq]
    r["changed_coords_step1_frac"] = float((ch == 1).float().mean()) if ch.numel() else None
    med = torch.where(eq, torch.full_like(S, 10 ** 6), S).float().median(1).values
    strong = med >= 4  # rows whose changed coordinates moved >= 4 ulp (typical)
    r["strong_rows"] = int(strong.sum())
    if strong.sum() > 0:
        Ss = S[strong]
        ef = (Ss == 0).float().mean(1)
        r["strong_rows_equal_frac_median"] = float(ef.median())
        r["strong_rows_equal_frac_p90"] = float(ef.quantile(0.9))
        d = (Mx[moved][strong].float() - O[moved][strong].float()).abs()
        nz = d > 0
        dm = torch.where(nz, d, torch.full_like(d, float("nan"))).nanmedian(1).values[:, None]
        small = (nz & (d < 0.25 * dm)).float().sum(1) / nz.float().sum(1).clamp_min(1)
        r["strong_rows_small_change_frac_median"] = float(small.median())
    # uniform scaling of rows (decay, or merge weights not summing to 1)
    Mf, Of = Mx[moved].double(), O[moved].double()
    ratio = (Mf * Of).sum(1) / (Of * Of).sum(1).clamp_min(1e-30)
    resid = ((Mf - ratio[:, None] * Of).norm(dim=1) / Of.norm(dim=1).clamp_min(1e-30))
    pure = resid < 2 ** -8  # within bf16 rounding of a pure rescale
    r["pure_rescale_rows"] = int(pure.sum())
    if pure.sum() > 0:
        r["pure_rescale_factor_median"] = float(ratio[pure].median())
    return r, moved


def score(pred, truth):
    return dict(tp=int((pred & truth).sum()), fp=int((pred & ~truth).sum()), fn=int((~pred & truth).sum()))


def with_parents(Mx, O, parents):
    n = min([Mx.shape[0], O.shape[0]] + [p.shape[0] for p in parents.values()])
    Mx, O = Mx[:n], O[:n]
    P = {k: v[:n] for k, v in parents.items()}
    touched = {k: (v != O).any(1) for k, v in P.items()}
    union = torch.zeros(n, dtype=torch.bool)
    for t in touched.values():
        union |= t
    moved = (Mx != O).any(1)
    r = dict(record_vs_union=score(moved, union), untouched_by_all=int((~union).sum()),
             untouched_by_all_identical=int(((~union) & ~moved).sum()))
    if ((~union) & moved).any():
        st = (okey(Mx[~union & moved]) - okey(O[~union & moved])).abs()
        r["untouched_but_changed_max_step"] = int(st.max())
        r["untouched_but_changed_coord_frac"] = float((st > 0).float().mean())
    # rows touched by exactly one parent: coefficient on that parent's update
    coef, keep = {}, {}
    for k, t in touched.items():
        only = t.clone()
        for k2, t2 in touched.items():
            if k2 != k:
                only &= ~t2
        if only.sum() < 5:
            continue
        D = (P[k][only].double() - O[only].double())
        E = (Mx[only].double() - O[only].double())
        c = (D * E).sum(1) / (D * D).sum(1).clamp_min(1e-30)
        big = D.abs() >= 4 * O[only].double().abs() * 2 ** -8  # coordinates the parent moved >= ~4 ulp
        kept = ((E != 0) & big).double().sum(1) / big.double().sum(1).clamp_min(1)
        ok = big.sum(1) >= 16
        coef[k] = dict(rows=int(only.sum()), median=float(c.median()), q25=float(c.quantile(.25)),
                       q75=float(c.quantile(.75)))
        if ok.any():
            coef[k]["kept_frac_median"] = float(kept[ok].median())
            # coefficient using only coordinates M kept (undoes DARE's drop; rescale shows as 1/density)
            Dk = torch.where(E != 0, D, torch.zeros_like(D))
            ck = (Dk * E).sum(1) / (Dk * Dk).sum(1).clamp_min(1e-30)
            coef[k]["kept_coord_coef_median"] = float(ck[ok].median())
    r["single_parent_rows"] = coef
    # parents whose embedding is bit-identical to the base carry no separate signal
    same_as_base = [k for k, v in P.items() if torch.equal(v, O)]
    r["parents_identical_to_base"] = same_as_base
    Pu = {k: v for k, v in P.items() if k not in same_as_base}
    # a parent that differs from the base only by a float16 round trip (tiny values lose precision) did not train
    r["parents_fp16_roundtrip_only"] = [k for k, v in Pu.items() if torch.equal(v, O.to(torch.float16).to(O.dtype))]
    if Pu:
        big = torch.zeros_like(O, dtype=torch.bool)
        for v in Pu.values():
            big |= (okey(v) - okey(O)).abs() >= 4
        r["kept_frac_global"] = float(((Mx != O) & big).sum() / max(int(big.sum()), 1))
    # least squares on update vectors: M - O ~ sum a_k (P_k - O), over coordinates of updated token vectors
    rows = torch.nonzero(moved).flatten()
    g = torch.Generator().manual_seed(0)
    if len(rows) > 20000:
        rows = rows[torch.randperm(len(rows), generator=g)[:20000]]
    names = list(Pu)
    if names and len(rows):
        Ob = O[rows].double()
        X = torch.stack([(Pu[k][rows].double() - Ob).flatten() for k in names], 1)
        y = (Mx[rows].double() - Ob).flatten()
        sol = torch.linalg.lstsq(X, y[:, None]).solution.flatten()
        r["lstsq_coef"] = {k: float(v) for k, v in zip(names, sol)}
        r["lstsq_rel_resid"] = float((X @ sol - y).norm() / y.norm().clamp_min(1e-30))
        sv = torch.linalg.svdvals(X / X.norm(dim=0).clamp_min(1e-30))
        r["lstsq_condition"] = float(sv[0] / sv[-1].clamp_min(1e-30))
        kept = (Mx[rows] != O[rows]).flatten()
        if int(kept.sum()) > len(names):
            solk = torch.linalg.lstsq(X[kept], y[kept][:, None]).solution.flatten()
            r["lstsq_coef_kept_coords"] = {k: float(v) for k, v in zip(names, solk)}
    return r


def analyse(Mx, O, parents=None):
    out = dict(stored_dtype=str(Mx.dtype).replace("torch.", ""))
    if Mx.dtype != O.dtype:
        Mx = Mx.to(O.dtype)
    b, _ = blind(Mx, O)
    out["blind"] = b
    if parents:
        out["with_parents"] = with_parents(Mx, O, parents)
    return out
