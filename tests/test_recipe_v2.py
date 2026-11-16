"""Reader v2 (Fix-E): continuous weight decay and shape selection by bit-exact replay."""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trainforensics import adam, recipe_v2  # noqa: E402


def _x0(n=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g) * 0.02).float()


def _lr(kind, N, W, peak):
    return adam.schedule(kind, N, peak, W)


def test_replay_matches_foreach_scalar():
    x0 = _x0()
    lr = _lr("linear", 120, 7, 2.3e-5)
    ref = x0.clone()
    for t in range(1, len(lr)):
        torch._foreach_mul_([ref], 1 - float(lr[t]) * 0.013)
    assert torch.equal(recipe_v2.replay(x0, lr, [0.013])[0], ref)


def test_fit_wd_recovers_non_round_decay():
    x0 = _x0()
    lr = _lr("cosine", 300, 11, 3.7e-5)
    x1 = recipe_v2.replay(x0, lr, [0.0137])[0]
    w, h, fr, (a, b) = recipe_v2.fit_wd(x0, x1, lr, wd_guess=0.02, interval=True)
    assert fr == 1.0 and a <= 0.0137 <= b and a <= w <= b
    # no round value reproduces the same factors, so none is reported
    assert recipe_v2.round_in((a, b)) is None


def test_round_value_kept_on_tie():
    x0 = _x0()
    lr = _lr("linear", 200, 0, 1e-4)
    x1 = recipe_v2.replay(x0, lr, [0.1])[0]
    w, h, fr, iv = recipe_v2.fit_wd(x0, x1, lr, wd_guess=0.13, interval=True)
    assert fr == 1.0 and recipe_v2.round_in(iv) == 0.1


def test_select_picks_true_shape_and_flags_out_of_family():
    x0 = _x0()
    true = dict(kind="linear", N=150, warmup=9, peak=2.3e-5, min_ratio=0.0)
    lr_fn = lambda r: _lr(r["kind"], r["N"], r["warmup"], r["peak"])  # noqa: E731
    x1 = recipe_v2.replay(x0, lr_fn(true), [0.013])[0]
    pool = [dict(true, N=148, warmup=8), dict(true, kind="cosine"), dict(true, N=230)]
    res = recipe_v2.select(pool, lr_fn, x0[:256], x1[:256], x0, x1, lambda r, lr: 0.02,
                           dscore_fn=lambda r: abs(r["N"] - 150) + abs(r["warmup"] - 9))
    assert res["recovered"] and res["recipe"]["N"] == 150 and res["recipe"]["warmup"] == 9
    a, b = res["wd_interval"]
    assert a <= 0.013 <= b
    # weak decay over 150 steps: neighbouring shapes reproduce the same weights (reported as ties)
    assert {"kind": "linear", "N": 150, "warmup": 9, "min_ratio": 0.0} in res["replay_ties"]
    # decay from a schedule the pool cannot express: low confidence, flagged
    odd = np.concatenate([[0.0], 2.3e-5 / np.sqrt(np.arange(1, 151) / 10 + 1)])
    x1b = recipe_v2.replay(x0, odd, [0.03])[0]
    res2 = recipe_v2.select([dict(true, kind="constant", N=150, warmup=0)], lr_fn, x0[:256], x1b[:256], x0, x1b,
                            lambda r, lr: 0.03, neigh_N=1, neigh_W=1)
    assert not res2["recovered"]
