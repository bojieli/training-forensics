#!/usr/bin/env python3
"""Multi-epoch reading, joint over both visits.

A token that occurs in one claimed document is touched once per epoch, at s1 in
epoch 1 and s2 in epoch 2. Its row's coordinates are a continuum between the two
shades D(s1) +- D(s2) (the second touch's gradient ratio varies per coordinate
and the first touch's second moment has not decayed for gaps of a few hundred
steps), so splitting a row into two shades fails at short gaps. What survives
at every gap is the top of the row: coordinates where both touches pushed the
same way sit at D(s1) + D(s2). So:

  recipe  fit the schedule to the distribution of row maxima under an E-epoch
          model: the predicted maxima are D(s1) + ... + D(sE) with s_e uniform
          in epoch e (E = 1 is the usual single-touch fit; E is chosen by
          likelihood)
  seed    for a candidate seed, every claimed doc has its steps in every epoch
          (epoch e uses randperm(seed + e)); count docs whose unique tokens' row
          maxima equal the predicted sum. The true seed stands out.
  read    with recipe and seed every row's steps are predicted; a row is read
          when its maximum matches the predicted sum.

Uses only W0, W1, the claimed documents and the batch size; truth.json is read
in score() only.

  python3 scripts/gaps/epochs.py runs/ctl/s01_2ep [...]
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "controlled"))
import score as S  # noqa: E402
from trainforensics import blind  # noqa: E402
from trainforensics.diff import row_diff  # noqa: E402

MAXPAIRS = 200_000


def sums_curve(D, E, rng_seed=0):
    """Predicted row maxima for E epochs: D(s1)+...+D(sE), s_e uniform in epoch e."""
    N = len(D) - 1
    if E == 1:
        return D
    L = N // E
    parts = [D[1 + e * L: 1 + (e + 1) * L] for e in range(E)]
    if L ** E <= MAXPAIRS:
        grids = np.meshgrid(*parts, indexing="ij")
        s = sum(g.ravel() for g in grids)
    else:
        r = np.random.default_rng(rng_seed)
        s = sum(p[r.integers(0, L, MAXPAIRS)] for p in parts)
    return np.concatenate([[0.0], np.sort(s)])


def _eval2(job):
    logobs, kind, N, W, mr, E, sigma = job
    r = blind.Recipe(kind, N, W, 1.0, min_ratio=mr)
    D = blind.moves_curve(r.lr(), 0.0)
    sh, sc = blind._fit_peak(logobs, sums_curve(D, E), sigma)
    r.peak, r.score, r.coverage, r.resid = math.exp(sh), *sc
    return (r, E)


def pmap(jobs, procs=24):
    with mp.get_context("fork").Pool(procs) as pool:
        return pool.map(_eval2, jobs, chunksize=4)


def fit_recipe_epochs(mx, Es=(1, 2, 3), sigma=1.5e-3):
    logobs = np.log(mx[mx > 0])
    Ns = sorted({int(round(x)) for x in np.exp(np.linspace(np.log(20), np.log(20000), 80))})
    jobs = [(logobs, kind, N, W, mr, E, sigma) for E in Es for N in Ns if N % E == 0 or True
            for kind in blind.KINDS for mr in ((0.1, 0.0) if kind in ("cosine_min", "wsd") else (0.0,))
            for W in blind._warmups(N)]
    res = sorted(pmap(jobs), key=lambda x: x[0].score)
    # refine around the best few (kind, E, N)
    jobs, keys = [], set()
    for r0, E in res:
        key = (r0.kind, r0.min_ratio, r0.N, E)
        if key in keys:
            continue
        keys.add(key)
        for N in range(max(10, int(r0.N * 0.9)), int(r0.N * 1.1) + 2):
            for W in {r0.warmup, int(round(r0.warmup * N / r0.N))}:
                jobs.append((logobs, r0.kind, N, W, r0.min_ratio, E, sigma))
        if len(keys) >= 8:
            break
    ref = sorted(pmap(jobs) + res[:50], key=lambda x: x[0].score)
    b, E = ref[0]
    Wmax = max(40, 2 * b.warmup + 10)
    jobs = [(logobs, b.kind, N, W, b.min_ratio, E, sigma)
            for N in range(max(10, b.N - 12), b.N + 13) for W in range(0, Wmax + 1)]
    ref = sorted(ref + pmap(jobs), key=lambda x: x[0].score)
    best_by_E = {}
    for r, e in ref:
        best_by_E.setdefault(e, r)
    return ref[0], best_by_E


def perms(seed, n, E):
    out = []
    for e in range(E):
        g = torch.Generator().manual_seed(seed + e)
        p = torch.randperm(n, generator=g).numpy()
        inv = np.empty(n, np.int64); inv[p] = np.arange(n)
        out.append(inv)
    return out


def read(run: pathlib.Path):
    t0 = time.time()
    e = load_file(str(run / "emb.safetensors"))
    W0, W1 = e["W0"], e["W1"]
    rd = row_diff(W0, W1, torch.float32)
    mv = np.nonzero(rd.moved)[0]
    d = (W1[mv].double() - rd.decay * W0[mv].double()).abs()
    mx = np.zeros(len(rd.moved)); mx[mv] = d.max(1).values.numpy()
    q98 = np.zeros(len(rd.moved)); q98[mv] = torch.quantile(d, 0.98, dim=1).numpy()
    docs, args = S.doc_tokens_for(run)
    bs = args["bs"] * args["ga"]
    cnt = {}
    for toks in docs.values():
        for tk in set(toks):
            cnt[tk] = cnt.get(tk, 0) + 1
    # rows whose top is tight (both touches aligned in >2% of coordinates)
    tight = rd.moved & (mx > 0) & (mx / np.maximum(q98, 1e-30) - 1 < 0.01)
    (best, E), byE = fit_recipe_epochs(mx[tight])
    rec = best.as_dict(); rec["epochs"] = E
    from score import snap
    rec["peak_snap"] = snap(rec["peak"])
    lr = S.hf_lr(dict(rec, peak=rec["peak_snap"]), rec["N"])
    D = blind.moves_curve(lr, 0.0)
    L = rec["N"] // E
    n = len(docs)
    # seed: docs' unique tokens, max equals the predicted sum within tol
    uniq = {k: [tk for tk in set(toks) if cnt[tk] == 1 and rd.moved[tk]] for k, toks in docs.items()}
    ks = np.array([k for k in docs if uniq[k]])
    ktok = [np.array(uniq[k]) for k in ks]
    flat_k = np.concatenate([np.full(len(v), i) for i, v in enumerate(ktok)])
    flat_t = np.concatenate(ktok)
    lmx = np.log(mx[flat_t])

    def hits(seed):
        invs = perms(seed, n, E)
        steps = [np.minimum(e_ * L + inv[ks] // bs + 1, rec["N"]) for e_, inv in enumerate(invs)]
        pred = sum(D[s] for s in steps)[flat_k]
        ok = np.abs(lmx - np.log(np.maximum(pred, 1e-300))) < 3e-3
        return int(np.bincount(flat_k[ok], minlength=len(ks)).astype(bool).sum())
    pri = [42, 0, 1, 1234, 3407, 2024, 2025, 7, 99, 123, 12345, 1337, 31337]
    scores = []
    for sd in pri + [s for s in range(0, 20000) if s not in pri]:
        h = hits(sd)
        scores.append((h, sd))
        if h > 0.5 * len(ks) and len(scores) > 20:
            break
    scores.sort(reverse=True)
    seed = scores[0][1]
    return dict(run=run.name, recipe=rec, best_by_E={e: r.as_dict() for e, r in byE.items()},
                rows_tight=int(tight.sum()), seed=seed, seed_hits=scores[0][0], docs_with_unique=int(len(ks)),
                runner_up_seed=scores[1] if len(scores) > 1 else None, t=time.time() - t0), \
        dict(mx=mx, D=D, rd=rd, docs=docs, bs=bs, L=L, E=E)


def score(run: pathlib.Path, out, st):
    from trainforensics.truth import Run
    R = Run(run)
    a = R.t["args"]
    rec = out["recipe"]
    true = dict(kind=a["sched"], N=R.N, warmup=a["warmup"], peak=a["lr"], epochs=a["epochs"])
    lr_read = S.hf_lr(dict(rec, peak=rec["peak_snap"]), rec["N"])
    exact = len(lr_read) == len(R.lr) and float(np.max(np.abs(lr_read[1:] - R.lr[1:])) / R.lr.max()) < 1e-9
    res = dict(out, recipe_true=true, schedule_exact=bool(exact), seed_true=a["seed"])
    # rows touched exactly twice (truth): are both steps read?
    ts = R.token_steps()
    E, L, bs, D = st["E"], st["L"], st["bs"], st["D"]
    invs = perms(out["seed"], len(st["docs"]), E)
    pred = {}
    for k, toks in st["docs"].items():
        ss = tuple(int(e_ * L + invs[e_][k] // bs + 1) for e_ in range(E))
        for tk in set(toks):
            pred.setdefault(tk, set()).update(ss)
    twice = [(tk, s) for tk, s in ts.items() if len(s) == 2]
    both_right = sum(1 for tk, s in twice if pred.get(tk) == set(s))
    consistent = sum(1 for tk, s in twice if pred.get(tk) == set(s) and
                     abs(math.log(st["mx"][tk] / (D[s[0]] + D[s[1]]))) < 3e-3)
    res["rows_touched_twice"] = len(twice)
    res["both_steps_exact_with_seed"] = both_right / max(len(twice), 1)
    res["both_steps_exact_and_row_max_matches"] = consistent / max(len(twice), 1)
    return res


def main():
    out_all = {}
    dst = ROOT / "results" / "gaps" / "epochs_v2.json"
    if dst.exists():
        out_all = json.load(open(dst))
    for r in sys.argv[1:]:
        run = pathlib.Path(r)
        o, st = read(run)
        res = score(run, o, st)
        out_all[run.name] = res
        print(json.dumps(res, default=float), flush=True)
        json.dump(out_all, open(dst, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
