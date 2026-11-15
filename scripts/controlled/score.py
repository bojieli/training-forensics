#!/usr/bin/env python3
"""Blind reading of a controlled run, then scoring against its truth.json.

The reading stage only uses: W0 (the public starting model), W1 (the release),
and for the dataset stages the claimed documents (text -> token ids) plus the
batch size. truth.json is opened only in score().

  python3 scripts/controlled/score.py runs/ctl/r01_lin [--release fp32|bf16|fp16]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import blind  # noqa: E402
from trainforensics.diff import row_diff, fit_decay_16, moved_rows_16  # noqa: E402

DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def snap(x: float, rel=5e-3):
    """Nearest 'round' hyper-parameter (m x 10^k, m in a short list) within rel."""
    if x <= 0:
        return x
    k = math.floor(math.log10(x))
    best = x
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10):
        v = m * 10 ** k
        if abs(v / x - 1) < rel and abs(v / x - 1) < abs(best / x - 1) + (best == x):
            best = v
    return best


def hf_lr(rec: dict, steps: int) -> np.ndarray:
    """lr used at 1-based steps via the transformers scheduler (exact float path)."""
    from transformers import get_scheduler
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=rec["peak"])
    kind = rec["kind"]
    kw = {}
    if kind == "cosine_min":
        kind, kw = "cosine_with_min_lr", {"min_lr_rate": rec["min_ratio"]}
    elif kind == "wsd":
        nd = int(0.2 * steps)
        kind, kw = "warmup_stable_decay", {"num_decay_steps": nd, "num_stable_steps": steps - rec["warmup"] - nd,
                                           "min_lr_ratio": rec["min_ratio"]}
    sch = get_scheduler(kind, opt, num_warmup_steps=rec["warmup"], num_training_steps=steps,
                        scheduler_specific_kwargs=kw)
    out = [0.0]
    for _ in range(steps):
        out.append(float(sch.get_last_lr()[0]))
        sch.step()
    return np.array(out)


def row_q(W0, W1, decay, rows, q=0.99):
    d = (W1[rows].double() - decay * W0[rows].double()).abs()
    return torch.quantile(d, q, dim=1).numpy()


# ---------------------------------------------------------------- reading
def read(run: pathlib.Path, release="fp32", with_data=True, bs=None):
    e = load_file(str(run / "emb.safetensors"))
    W0 = e["W0"]
    W1 = {"fp32": e["W1"], "bf16": e["W1_bf16"], "fp16": e["W1_fp16"]}[release]
    out = {"release": release}
    t0 = time.time()
    if release == "fp32":
        rd = row_diff(W0, W1, torch.float32)
        decay = rd.decay
    else:
        P, frac = fit_decay_16(W0, W1, DT[release])
        nd = moved_rows_16(W0, W1, DT[release], P)
        rd = row_diff(W0, W1.float(), DT[release])
        rd.moved = nd > 0
        decay = P
        out["decay_fit_frac"] = frac
    out["moved_rows"] = int(rd.moved.sum())
    out["decay"] = decay
    # Per-row move read as the intercept of the sorted-moves curve (handles
    # both flat rows and eps-limited ones); rows whose sorted moves fit that
    # one-touch curve closely are taken as single-touch.
    out["flat_rows_classic"] = int((rd.moved & (rd.flat_frac > 0.9)).sum())
    mv = np.nonzero(rd.moved)[0]
    if release == "fp32" and len(mv):
        if decay < 1.0:
            # replace decay*W0 by the exact decay map read off the unmoved rows
            keys, vals, cons = blind.decay_lut(W0, W1, ~rd.moved)
            ref, cov = blind.apply_lut(W0[mv], keys, vals, decay)
            out["decay_lut"] = dict(distinct_values=int(len(keys)), consistent=cons, coverage_moved=cov)
            W0ref = W0.clone()
            W0ref[mv] = ref
            Dm, Tm, Rm = blind.row_intercepts(W0ref, W1, 1.0, mv)
        else:
            Dm, Tm, Rm = blind.row_intercepts(W0, W1, decay, mv)
        single = np.zeros(len(rd.moved), bool)
        single[mv] = Rm < 3e-3
        Dfull0 = np.zeros(len(rd.moved)); Dfull0[mv] = Dm
        rd.flat_frac = np.where(single, 1.0, 0.0)          # downstream code keys on flat_frac > 0.9
        rd.D = Dfull0
        rows = np.nonzero(single)[0]
        Dq = Dfull0[rows]
    else:
        rows = np.nonzero(rd.moved & (rd.flat_frac > 0.9))[0]
        Dq = rd.D[rows]
    out["flat_rows"] = int(len(rows))
    # recipe: first pass with wd=0, then wd from the decay, refit shape with wd
    # one pass: each candidate shape carries the decay strength implied by the
    # total decay P read off the never-touched rows
    best, ranked = blind.fit_recipe(Dq, logP=(math.log(decay) if decay < 1.0 else None))
    wd = blind.wd_from_decay(decay, best.lr()) if decay < 1.0 else 0.0
    rec = best.as_dict()
    rec["wd"] = wd
    rec["peak_snap"] = snap(rec["peak"])
    rec["wd_snap"] = snap(wd) if wd else 0.0
    out["recipe"] = rec
    if rec["wd_snap"] > 0:
        # Under decoupled decay the never-touched rows are an exact record of the
        # factor sequence: refine the step count (and warm-up) by bit-exact replay
        # on a sample of rows that did not move.
        idx = np.nonzero(~rd.moved)[0]
        idx = idx[np.linspace(0, len(idx) - 1, min(2000, len(idx))).astype(int)]
        x0s, x1s = W0[idx].float(), W1[idx]
        def hits(r):
            lr_ = hf_lr(dict(r, peak=r["peak_snap"]), r["N"])
            x = x0s.clone()
            for t in range(1, r["N"] + 1):
                torch._foreach_mul_([x], 1 - float(lr_[t]) * r["wd_snap"])
            if x1s.dtype != torch.float32:
                x = x.to(x1s.dtype)
            return int((x == x1s).sum())
        # wd: the decay ratio is biased by fp32 factor rounding and drift, so
        # pick among round values near it by bit-exact hits (rounder wins ties)
        cands = []
        for m in (1, 2, 5, 3, 4, 1.5, 2.5, 6, 7, 8, 9):
            for k in range(-6, 1):
                v = m * 10.0 ** k
                if 0.6 * rec["wd"] <= v <= 1.6 * rec["wd"]:
                    cands.append(v)
        best_wd, best_h = rec["wd_snap"], hits(rec)
        for v in cands:
            h = hits(dict(rec, wd_snap=v))
            if h > best_h:
                best_wd, best_h = v, h
        rec["wd_snap"] = best_wd
        best_r = dict(rec)
        for dN in range(-3, 4):
            for dW in ((-1, 0, 1) if rec["warmup"] else (0,)):
                if dN == 0 and dW == 0:
                    continue
                r = dict(rec, N=rec["N"] + dN, warmup=max(0, rec["warmup"] + dW))
                h = hits(r)
                if h > best_h:
                    best_r, best_h = r, h
        out["replay_refine"] = dict(wd=best_wd, before=dict(N=rec["N"], warmup=rec["warmup"]),
                                    after=dict(N=best_r["N"], warmup=best_r["warmup"]),
                                    sample_weights=int(x1s.numel()), hits=best_h)
        rec.update(N=best_r["N"], warmup=best_r["warmup"])
    Dfull = rd.D.copy()
    if len(rows):
        Dfull[rows] = Dq * (rec["peak_snap"] / rec["peak"])
    out["_D"] = Dfull
    out["runner_up"] = [r.as_dict() for r in ranked[1:3]]
    out["t_recipe"] = time.time() - t0
    out["_rd"] = rd
    out["_W"] = (W0, W1)
    return out


def replay(out, steps_shift=0):
    """Bit-exact replay of never-touched rows (needs decoupled wd > 0 to be non-trivial)."""
    W0, W1 = out["_W"]
    rd = out["_rd"]
    rec = dict(out["recipe"], peak=out["recipe"]["peak_snap"])
    N = rec["N"] + steps_shift
    lr = hf_lr(rec, N)
    wd = out["recipe"]["wd_snap"]
    idx = np.nonzero(~rd.moved)[0]
    x = W0[idx].clone().float().cuda() if torch.cuda.is_available() else W0[idx].clone().float()
    if wd > 0:
        for t in range(1, N + 1):
            torch._foreach_mul_([x], 1 - float(lr[t]) * wd)
    y = W1[idx].to(x.device)
    if y.dtype != torch.float32:
        x = x.to(y.dtype)
    eq = (x == y)
    return {"weights": int(eq.numel()), "bit_exact": int(eq.sum()), "zeros_exact": int(((y == 0) & eq).sum())}


def moved_by_replay(out):
    W0, W1 = out["_W"]
    rec = dict(out["recipe"], peak=out["recipe"]["peak_snap"])
    lr = hf_lr(rec, rec["N"])
    wd = out["recipe"]["wd_snap"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    moved = np.zeros(W0.shape[0], bool)
    for a in range(0, W0.shape[0], 16384):
        x = W0[a:a + 16384].clone().float().to(dev)
        for t in range(1, rec["N"] + 1):
            torch._foreach_mul_([x], 1 - float(lr[t]) * wd)
        moved[a:a + 16384] = (x.cpu() != W1[a:a + 16384]).any(1).numpy()
    return moved


def doc_tokens_for(run: pathlib.Path, which="claimed"):
    t = json.load(open(run / "truth.json"))      # only the claimed list + args (public claim)
    toks = {int(k): v for k, v in json.load(open(run / "tokens.json")).items()}
    ids = t[which]
    return {k: toks[d][:-1] for k, d in enumerate(ids)}, t["args"]


# ---------------------------------------------------------------- scoring
def score(run: pathlib.Path, out: dict, with_docs=True):
    from trainforensics.truth import Run
    R = Run(run)
    seen = R.seen_mask()
    rd = out["_rd"]
    res = {"run": run.name, "release": out["release"]}
    res["tokens"] = dict(seen=int(seen.sum()), moved=int(rd.moved.sum()),
                         tp=int((seen & rd.moved).sum()), fp=int((~seen & rd.moved).sum()),
                         fn=int((seen & ~rd.moved).sum()))
    a = R.t["args"]
    true = dict(kind=a["sched"], N=R.N, warmup=a["warmup"], peak=a["lr"], wd=a["wd"])
    rec = out["recipe"]
    res["recipe_true"] = true
    res["recipe_read"] = {k: rec[k] for k in ("kind", "N", "warmup", "peak", "peak_snap", "wd", "wd_snap", "min_ratio")}
    res["recipe_ok"] = dict(kind=rec["kind"] == true["kind"], N=rec["N"] == true["N"],
                            warmup=rec["warmup"] == true["warmup"],
                            peak_rel_err=rec["peak"] / true["peak"] - 1,
                            peak_snap_ok=abs(rec["peak_snap"] / true["peak"] - 1) < 1e-9,
                            wd_rel_err=(rec["wd"] / true["wd"] - 1) if true["wd"] else rec["wd"])
    # objective check: the schedule the reading implies vs the LR the trainer logged
    lr_read = hf_lr(dict(rec, peak=rec["peak_snap"]), rec["N"])
    if len(lr_read) == len(R.lr):
        dl = float(np.max(np.abs(lr_read[1:] - R.lr[1:])) / max(R.lr.max(), 1e-30))
    else:
        dl = None
    lr_unsnapped = hf_lr(rec, rec["N"]) if len(lr_read) == len(R.lr) else None
    res["recipe_ok"]["steps"] = rec["N"] == R.N
    res["recipe_ok"]["lr_max_rel_dev_snapped"] = dl
    res["recipe_ok"]["lr_max_rel_dev_raw"] = (float(np.max(np.abs(lr_unsnapped[1:] - R.lr[1:])) / R.lr.max())
                                              if lr_unsnapped is not None else None)
    res["recipe_ok"]["schedule_exact"] = dl is not None and dl < 1e-9
    res["replay_refine"] = out.get("replay_refine")
    res["decay_lut"] = out.get("decay_lut")
    res["t_recipe"] = out["t_recipe"]
    if with_docs and out["release"] == "fp32" and a["epochs"] == 1:
        docs, args = doc_tokens_for(run)
        curve = blind.moves_curve(hf_lr(dict(rec, peak=rec["peak_snap"]), rec["N"]), rec["wd_snap"])
        pl = blind.place_docs(docs, rd, curve, D=out['_D'])
        # truth: claimed position k -> step (None if swapped out)
        k2s = {}
        for s in range(1, R.N + 1):
            for k in R.t["docs_per_step"][s - 1]:
                k2s[k] = s
        swapped = set(R.t["swapped_positions"])
        placed = {k: v[0] for k, v in pl.items() if v[0] is not None}
        right = sum(1 for k, s in placed.items() if k2s.get(k) == s and k not in swapped)
        res["placement"] = dict(docs=len(docs), placed=len(placed), exact=right,
                                wrong=len(placed) - right,
                                status_counts={st: sum(1 for v in pl.values() if v[2] == st)
                                               for st in set(v[2] for v in pl.values())})
        # seed search from the placed docs (positions are the claimed list order)
        bs = args["bs"] * args["ga"]
        good = {k: s for k, s in placed.items()}
        ranked = blind.find_seed(good, len(docs), bs, seeds=range(0, 100000))
        res["seed"] = dict(true=args["seed"], found=ranked[0][0] if ranked else None,
                           hits=ranked[0][1] if ranked else 0, of=len(good),
                           second=ranked[1] if len(ranked) > 1 else None)
        # swaps: with the seed, every claimed doc has a predicted step; a doc is
        # flagged when one of its single-touch tokens (unique to it among claimed
        # docs) is unmoved or sits at a different step, or any token is unmoved.
        if ranked:
            g = torch.Generator().manual_seed(ranked[0][0])
            perm = torch.randperm(len(docs), generator=g).numpy()
            inv = np.empty(len(docs), np.int64)
            inv[perm] = np.arange(len(docs))
            flags, why = set(), {}
            Dd = out["_D"]
            W0f = out["_W"][0]
            # per-row rounding tolerance: fp32 spacing at the row's typical |value|
            # relative to its move (end-of-schedule moves are only ~100 ulp)
            ulp = (W0f.abs().max(dim=1).values.double().clamp_min(1e-30).log2().floor() - 23).exp2().numpy()
            def tol_hi(tk):
                return 3e-3 + 8 * ulp[tk] / max(Dd[tk], 1e-30)   # q99 of |delta| picks up ~+5 ulp of accumulated rounding
            # predicted touch steps of every token from the claimed docs + seed
            E = {}
            for k, toks in docs.items():
                for tk in set(toks):
                    E.setdefault(tk, set()).add(int(inv[k] // bs + 1))
            def matches(tk, s):
                if s >= len(curve):
                    return False
                l = math.log(max(Dd[tk], 1e-300) / max(curve[s], 1e-300))
                return -3e-2 < l < tol_hi(tk)
            for k, toks in docs.items():
                s_pred = int(inv[k] // bs + 1)
                for tk in set(toks):
                    if not rd.moved[tk]:
                        flags.add(k); why[k] = "unmoved_token"; break
                    if rd.flat_frac[tk] <= 0.9:
                        continue
                    steps = E[tk]
                    if len(steps) == 1 and not matches(tk, s_pred):
                        flags.add(k); why[k] = "single_touch_wrong_step"; break
                    if len(steps) == 2 and False:   # disabled: two-touch rows can look flat (17 false flags on r01)
                        # a predicted double touch that shows as ONE clean touch at
                        # the other doc's step: this doc's step is missing
                        other = next(iter(steps - {s_pred}))
                        if matches(tk, other) and not matches(tk, s_pred) and \
                                abs(math.log(max(curve[other], 1e-300) / max(curve[s_pred], 1e-300))) > 0.05:
                            flags.add(k); why[k] = "double_touch_seen_once"; break
            res["swap_reasons"] = {r: sum(1 for v in why.values() if v == r) for r in set(why.values())}
            res["placement_with_seed"] = int(sum(1 for k in docs if k2s.get(k) == inv[k] // bs + 1))
            res["swaps"] = dict(true=len(swapped), flagged=len(flags), caught=len(flags & swapped),
                                false_flags=len(flags - swapped))
    if (out["recipe"]["wd_snap"] or 0) > 0:
        # tokens under decoupled decay: a row never touched equals the exact
        # fp32 replay of the decay on its starting value, bit for bit
        mv_rep = moved_by_replay(out)
        res["tokens_replay"] = dict(moved=int(mv_rep.sum()), tp=int((seen & mv_rep).sum()),
                                    fp=int((~seen & mv_rep).sum()), fn=int((seen & ~mv_rep).sum()))
        res["replay"] = replay(out)
        res["replay_shift1"] = replay(out, steps_shift=1)
        res["replay_shiftm1"] = replay(out, steps_shift=-1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--release", default="fp32")
    ap.add_argument("--no-docs", action="store_true")
    a = ap.parse_args()
    for r in a.runs:
        run = pathlib.Path(r)
        out = read(run, a.release)
        res = score(run, out, with_docs=not a.no_docs)
        dst = ROOT / "results" / "controlled" / f"{run.name}.{a.release}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(dst, "w"), indent=1, default=float)
        print(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
