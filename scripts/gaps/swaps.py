#!/usr/bin/env python3
"""Swap detection v2: with the recovered seed every claimed document has an exact
step, so every embedding row has a predicted touch set E[t] (the steps of the
claimed documents that contain token t). A secretly swapped document leaves:

  A  a token of it whose row never moved                                  (v1)
  B  a token only it contains (E={s}), row single-touch at another step    (v1)
  C  a token shared with docs at later steps whose row is single-touch at a
     later predicted step: Adam moves a row at full size from its first touch,
     whatever |g| (above the eps floor), so a real earlier touch at s cannot
     leave the row looking like a clean later single touch                (new)
  D  (step level only: picking the document inside the step gave 0 right / 3 wrong
     on s05, so D is reported per step, not used to flag documents)
     a row that moved more than any history inside E allows:
     max_i |delta_i| <= sum_{s in E} D(s) for every Adam history (per step
     |sum m| / sqrt(sum v) <= sum |m| / sqrt(v)); a violation, or a moved row
     with E empty, means an unclaimed document touched it. When that row is
     single-touch its step s* is the swapped slot: flag the one claimed doc of
     batch s* with no positive evidence of being there                    (new)

Positive evidence for doc k at step s: a token with E={s} whose row is
single-touch at s.

Reads the cache written by scripts/gaps/cache.py; truth.json only to score.
  python3 scripts/gaps/swaps.py s05_swap [r01_lin ...]
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import types

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "controlled"))
import score as S  # noqa: E402
from trainforensics import blind  # noqa: E402

CACHE = ROOT / "runs" / "gaps" / "cache"


def run_dir(name):
    for d in ("runs/ctl", "runs/gaps"):
        p = ROOT / d / name
        if (p / "truth.json").exists():
            return p
    raise FileNotFoundError(name)


def setup(name):
    """Blind state shared by the gap studies: cache + recipe curve + seed + steps."""
    run = run_dir(name)
    c = np.load(CACHE / f"{name}.npz")
    rec = json.loads(str(c["recipe"]))
    moved, Dd, single, Q = c["moved"], c["D"], c["single"], c["Q"]
    lr = S.hf_lr(dict(rec, peak=rec["peak_snap"]), rec["N"])
    curve = blind.moves_curve(lr, rec["wd_snap"])
    docs, args = S.doc_tokens_for(run)                 # claimed docs + batch size (the public claim)
    bs = args["bs"] * args["ga"]
    e = load_file(str(run / "emb.safetensors"))
    W0 = e["W0"]
    moved_how = "change above the rounding floor"
    if rec.get("wd_snap"):
        # with weight decay, a token that was never updated equals the exact fp32
        # replay of the recovered decay schedule, bit for bit; anything else was
        # updated. This catches tokens updated so late (learning rate near zero)
        # that their change sits below the generic rounding floor.
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        x = W0.clone().float().to(dev)
        for t in range(1, rec["N"] + 1):
            torch._foreach_mul_([x], 1 - float(lr[t]) * rec["wd_snap"])
        moved = (x.cpu() != e["W1"]).any(1).numpy()
        moved_how = "differs from the bit-exact decay replay"
        del x
    ulp = (W0.abs().max(dim=1).values.double().clamp_min(1e-30).log2().floor() - 23).exp2().numpy()
    rd = types.SimpleNamespace(moved=moved, flat_frac=single.astype(float), D=Dd)
    pl = blind.place_docs(docs, rd, curve, D=Dd)
    placed = {k: v[0] for k, v in pl.items() if v[0] is not None}
    seeds = blind.find_seed(placed, len(docs), bs, seeds=range(0, 100000))
    seed = seeds[0][0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(docs), generator=g).numpy()
    inv = np.empty(len(docs), np.int64); inv[perm] = np.arange(len(docs))
    step = {k: int(inv[k] // bs + 1) for k in docs}
    return types.SimpleNamespace(run=run, rec=rec, moved=moved, D=Dd, single=single, Q=Q, lr=lr, curve=curve,
                                 docs=docs, args=args, bs=bs, ulp=ulp, seed=seed, step=step,
                                 placed=len(placed), moved_how=moved_how)


def analyse(name, rules=("A", "B", "C")):
    st = setup(name)
    moved, Dd, single, Q, curve, docs, step, ulp = st.moved, st.D, st.single, st.Q, st.curve, st.docs, st.step, st.ulp
    batch = {}
    for k, s in step.items():
        batch.setdefault(s, []).append(k)
    E = {}
    for k, toks in docs.items():
        for tk in set(toks):
            E.setdefault(tk, set()).add(step[k])

    # with weight decay the updated coordinates go through ~N fp32 multiplies that
    # the bit-exact map of the never-updated ones does not reproduce exactly:
    # allow N/4 units in the last place (r08: up to ~63 at N=500)
    drift_ulps = max(16, st.rec["N"] // 4) if st.rec.get("wd_snap") else 16

    def matches(tk, s):
        """Single update at step s: Adam's eps can only shrink a coordinate's change, so
        no coordinate may exceed D(s) (plus rounding), and the row's fitted size may not
        fall far below it. The upper test uses the largest change, not the fitted
        intercept, which overshoots by up to ~0.6% on eps-limited tokens (r12)."""
        if s >= len(curve) or s < 1:
            return False
        l = math.log(max(Dd[tk], 1e-300) / max(curve[s], 1e-300))
        return l > -3e-2 and Q[tk, 3] <= curve[s] * (1 + 2e-3) + drift_ulps * ulp[tk]

    def single_steps(tk):
        if not (moved[tk] and single[tk]):
            return []
        cand, _ = blind.step_of_D(Dd[tk], curve, 1.5e-3)
        return [int(x) for x in cand]

    why = {}
    positive = {k: 0 for k in docs}
    for k, toks in docs.items():
        s = step[k]
        for tk in set(toks):
            if moved[tk] and single[tk] and E[tk] == {s} and matches(tk, s):
                positive[k] += 1
    for k, toks in docs.items():
        s = step[k]
        for tk in set(toks):
            if "A" in rules and not moved[tk]:
                why[k] = "A_unmoved"; break
            if not single[tk]:
                continue
            if "B" in rules and E[tk] == {s} and not matches(tk, s):
                why[k] = "B_single_wrong_step"; break
            if "C" in rules and len(E[tk]) >= 2 and s == min(E[tk]) and not matches(tk, s):
                later = [u for u in E[tk] if u > s and matches(tk, u)]
                if later and abs(math.log(curve[later[0]] / curve[s])) > 0.05:
                    why[k] = "C_first_touch_missing"; break
    bound = np.zeros(len(moved))
    for tk, ss in E.items():
        bound[tk] = sum(curve[u] for u in ss)
    over = np.nonzero(moved & (Q[:, 3] > bound * (1 + 1e-2) + drift_ulps * ulp))[0]
    extra_steps = {}
    for tk in over:
        ss = single_steps(tk)
        if len(ss) == 1:
            extra_steps.setdefault(ss[0], []).append(int(tk))
    if "D" in rules:
        for s, toks in extra_steps.items():
            cands = [k for k in batch.get(s, []) if k not in why]
            weak = [k for k in cands if positive[k] == 0]
            if len(weak) == 1:
                why[weak[0]] = "D_unclaimed_touch_at_step"
    t = json.load(open(st.run / "truth.json"))
    swapped = set(t["swapped_positions"])
    swapped_steps = {step[k] for k in swapped}
    step_level = dict(steps_with_unclaimed_update=len(extra_steps),
                      of_which_hold_a_swap=sum(1 for s in extra_steps if s in swapped_steps),
                      steps_holding_a_swap=len(swapped_steps))
    flagged = set(why)
    return dict(run=name, moved_how=st.moved_how, seed=st.seed, seed_true=st.args["seed"], true=len(swapped), flagged=len(flagged),
                caught=len(flagged & swapped), false_flags=len(flagged - swapped),
                by_rule={r: dict(caught=sum(1 for k, v in why.items() if v.startswith(r) and k in swapped),
                                 false=sum(1 for k, v in why.items() if v.startswith(r) and k not in swapped))
                         for r in rules},
                rows_over_bound=int(len(over)), step_level=step_level,
                missed=[dict(pos=k, n_tokens=len(set(docs[k])), positive=positive[k],
                             in_extra=step[k] in extra_steps,
                             n_cands_in_batch=len([j for j in batch[step[k]] if positive[j] == 0]))
                        for k in sorted(swapped - flagged)],
                false=[dict(pos=k, why=why[k]) for k in sorted(flagged - swapped)])


if __name__ == "__main__":
    out = {}
    for n in sys.argv[1:]:
        r = analyse(n)
        out[n] = r
        print(json.dumps({k: v for k, v in r.items() if k not in ("missed",)}), flush=True)
        for m in r["missed"]:
            print("   missed", m)
    dst = ROOT / "results" / "gaps" / "swaps_v2.json"
    old = json.load(open(dst)) if dst.exists() else {}
    old.update(out)
    json.dump(old, open(dst, "w"), indent=1, default=float)
