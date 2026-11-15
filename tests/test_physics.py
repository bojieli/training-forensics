"""Offline checks of the forward model the readings rely on."""
import math

import numpy as np
import pytest
import torch

from trainforensics import adam, blind


def test_saturation_constant():
    assert abs(adam.saturation_constant() - 31.77) < 0.01


def test_vectorised_curve_matches_reference():
    lr = adam.schedule("cosine", 400, 3e-5, 20)
    a = adam.single_touch_moves(lr, wd=0.05)
    b = blind.moves_curve(lr, 0.05)
    assert np.max(np.abs(a[1:] - b[1:]) / a[1:].clip(1e-30)) < 1e-10


@pytest.mark.parametrize("kind,hf", [("linear", "linear"), ("cosine", "cosine"), ("constant", "constant")])
def test_schedules_match_transformers(kind, hf):
    transformers = pytest.importorskip("transformers")
    N, W, peak = 300, 0 if kind == "constant" else 15, 4e-5
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=peak)
    sch = transformers.get_scheduler(hf, opt, num_warmup_steps=W, num_training_steps=N)
    got = []
    for _ in range(N):
        got.append(sch.get_last_lr()[0]); sch.step()
    want = adam.schedule(kind, N, peak, W)[1:]
    assert np.allclose(got, want, rtol=1e-12, atol=1e-18)


def test_single_touch_move_is_sign_only():
    """AdamW on a toy embedding: a row with one gradient moves by D(s) in every coordinate."""
    torch.manual_seed(0)
    V, d, N, s = 8, 64, 60, 17
    lr = adam.schedule("linear", N, 1e-3, 5)
    W = torch.nn.Parameter(torch.randn(V, d))
    W0 = W.detach().clone()
    opt = torch.optim.AdamW([W], lr=1e-3, weight_decay=0.0)
    for t in range(1, N + 1):
        for g in opt.param_groups:
            g["lr"] = float(lr[t])
        W.grad = torch.zeros_like(W)
        if t == s:
            W.grad[3] = torch.randn(d) * 1e-2          # one touch of row 3, nothing else
        opt.step()
    delta = (W.detach() - W0)[3].abs().double()
    D = blind.moves_curve(lr)[s]
    # fp32 rounding of |x|~1 over 60 steps is ~1e-4 relative; a few tiny-|g|
    # coordinates are eps-limited
    assert (torch.abs(delta / D - 1) < 2e-3).double().mean() > 0.99
    assert torch.equal(W.detach()[0], W0[0])            # untouched row: bit-identical


def test_replay_decay_bit_exact():
    torch.manual_seed(1)
    N, wd = 200, 0.1
    lr = adam.schedule("cosine", N, 1e-4, 10)
    W = torch.nn.Parameter(torch.randn(16, 32))
    W0 = W.detach().clone()
    opt = torch.optim.AdamW([W], lr=1e-4, weight_decay=wd, foreach=True)
    for t in range(1, N + 1):
        for g in opt.param_groups:
            g["lr"] = float(lr[t])
        W.grad = torch.zeros_like(W)
        opt.step()
    x = W0.clone()
    for t in range(1, N + 1):
        torch._foreach_mul_([x], 1 - float(lr[t]) * wd)
    assert torch.equal(x, W.detach())


def test_intercept_reads_eps_limited_rows():
    rng = np.random.default_rng(0)
    D, tau = 3e-4, 0.05   # real runs: tau ~0.001-0.01 (coarse tau grid: ~3% error at tau=0.5)
    g = np.abs(rng.standard_normal((50, 2048)))
    W0 = torch.zeros(50, 2048, dtype=torch.float64)
    W1 = torch.tensor(D * g / (g + tau))
    Dh, Th, R = blind.row_intercepts(W0, W1, 1.0, np.arange(50))
    assert np.median(np.abs(Dh / D - 1)) < 0.01 and np.median(R) < 0.01
