"""Offline tests for the readers the claims rest on (review M12): row_diff, fit_decay_16,
moved_rows_16, interval_move, sweep.fit_decay / analyse, p_interval.interval,
comparability.classify (including a negative-control regression) and the wsd /
cosine_min schedules."""
import pathlib
import sys

import numpy as np
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts" / "roster", ROOT / "scripts" / "frontier"):
    sys.path.insert(0, str(p))

from trainforensics import adam, diff  # noqa: E402

BF = torch.bfloat16


def _base(V=3000, d=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(V, d, generator=g, dtype=torch.float64) * 0.02).to(BF)


# ----------------------------------------------------------------------------- fit_decay_16
def test_fit_decay_16_recovers_shrink():
    W0 = _base()
    P_true = 0.9813377            # a generic factor: an exactly round P sits on a degenerate edge
    W1 = diff.round_to(W0.double() * P_true, BF)
    P, frac = diff.fit_decay_16(W0, W1, BF)
    assert abs(P - P_true) < 2e-5 and frac > 0.99
    assert diff.moved_rows_16(W0, W1, BF, P).max() == 0


def test_fit_decay_16_ignores_zero_rows():
    """Review M5: with all-zero rows in the sample the old search stayed at P = 1."""
    W0 = _base()
    W0[:5] = 0
    W1 = diff.round_to(W0.double() * 0.9813377, BF)
    P, frac = diff.fit_decay_16(W0, W1, BF)
    assert abs(P - 0.9813377) < 2e-5 and frac > 0.99
    assert (diff.moved_rows_16(W0, W1, BF, P) > 0).sum() == 0


def test_moved_rows_16_flags_updates_only():
    W0 = _base()
    x = W0.double() * 0.9913711
    x[:100] += 1e-3 * torch.sign(torch.randn(100, W0.shape[1], dtype=torch.float64))
    W1 = diff.round_to(x, BF)
    P, _ = diff.fit_decay_16(W0, W1, BF)
    nd = diff.moved_rows_16(W0, W1, BF, P)
    assert (nd[:100] > 50).all() and (nd[100:] == 0).all()


# ----------------------------------------------------------------------------- interval_move
def test_interval_move_contains_true_D():
    torch.manual_seed(3)
    w0 = (torch.randn(4096, dtype=torch.float64) * 0.02).to(BF)
    D = 7.3e-4
    s = torch.sign(torch.randn(4096, dtype=torch.float64))
    w1q = diff.round_to(w0.double() - s * D, BF)
    lo, hi = diff.interval_move(w0, w1q, BF)
    assert lo <= D <= hi and (hi - lo) / D < 0.05


def test_interval_move_power_of_two_asymmetry():
    """Below a power of two the spacing halves; the interval must use both neighbours."""
    q = torch.tensor([1.0], dtype=BF)          # stored value at a power of two
    up, dn = diff._neighbours(q, BF)
    assert float(up) - 1.0 == pytest.approx(2.0 ** -7)
    assert 1.0 - float(dn) == pytest.approx(2.0 ** -8)


# ----------------------------------------------------------------------------- row_diff
def test_row_diff_fp32_decay_and_moves():
    torch.manual_seed(4)
    W0 = torch.randn(2000, 128) * 0.02
    W1 = W0 * 0.995
    W1[:50] += 5e-4 * torch.sign(torch.randn(50, 128))
    rd = diff.row_diff(W0, W1, device="cpu")
    assert abs(rd.decay - 0.995) < 1e-6
    assert rd.moved[:50].all() and not rd.moved[50:].any()
    assert np.allclose(rd.D[:50], 5e-4, rtol=0.02)


def test_row_diff_no_decay_identical():
    W0 = torch.randn(500, 64)
    W1 = W0.clone()
    W1[7] += 1e-3
    rd = diff.row_diff(W0, W1, device="cpu")
    assert rd.decay == 1.0 and rd.moved.sum() == 1 and rd.moved[7]


# ----------------------------------------------------------------------------- sweep
def test_sweep_fit_and_analyse():
    sweep = pytest.importorskip("sweep")
    W0 = _base(4000, 256)
    x = W0.double() * 0.9851377
    x[:300] += 2e-3 * torch.sign(torch.randn(300, 256, dtype=torch.float64))
    W1 = diff.round_to(x, BF)
    P, frac, _, _ = sweep.fit_decay(W0, W1, BF)
    assert abs(P - 0.9851377) < 1e-4 and frac > 0.999
    st, unmoved, near, moved, _ = sweep.analyse(W0, W1, BF, None)
    assert st["unmoved"] == 3700 and st["moved"] == 300
    assert unmoved[300:].all() and moved[:300].all()


def test_sweep_fit_ignores_zero_rows():
    sweep = pytest.importorskip("sweep")
    W0 = _base(2000, 256)
    W0[:20] = 0
    W1 = diff.round_to(W0.double() * 0.9713377, BF)
    P, frac, _, _ = sweep.fit_decay(W0, W1, BF)
    assert abs(P - 0.9713377) < 1e-4


# ----------------------------------------------------------------------------- p_interval
def test_p_interval_contains_true_P():
    p_interval = pytest.importorskip("p_interval")
    W0 = _base(1000, 256)
    W1 = diff.round_to(W0.double() * 0.9876, BF)
    lo, hi, n = p_interval.interval(W0, W1)
    assert lo <= 0.9876 <= hi and hi - lo < 5e-4 and n > 0


# ----------------------------------------------------------------------------- comparability
def _rec(**e):
    base = dict(rows=150000, identical=0, P=1.0, P_exact_frac=1.0, unmoved=0)
    base.update(e)
    return dict(tied=False, embed=base)


def test_classify_negative_control_regression():
    """Review B1: Qwen2-7B vs Qwen2.5-7B (separate pretraining runs) shares 2,282 bit-identical
    vectors, all unreachable or at their random start. It must not be read as a token record."""
    comparability = pytest.importorskip("comparability")
    r = _rec(unmoved=2282, P=1.0, P_exact_frac=1.0)
    rc = dict(strict=2282, band=2282, left=0)
    assert comparability.classify(r, rc) != "token record"


def test_classify_real_record_and_zero_rows():
    comparability = pytest.importorskip("comparability")
    good = comparability.classify(_rec(unmoved=9000, P=0.987, P_exact_frac=0.9999), dict(strict=9000, band=9000, left=5000))
    assert good == "token record"
    # all-zero child rows (a hand edit) are never counted as bit-identical
    zeroed = comparability.classify(_rec(unmoved=2171, P=1.0, P_exact_frac=0.2),
                                    dict(strict=2171, band=2171, left=0, zeroed_edits=2171))
    assert zeroed != "token record"
    assert comparability.classify(dict(tied=True, embed={}), None) == "shared matrix"


# ----------------------------------------------------------------------------- schedules
@pytest.mark.parametrize("m", [0.0, 0.1])
def test_cosine_min_matches_transformers(m):
    tr = pytest.importorskip("transformers")
    N, W, peak = 300, 12, 3e-5
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=peak)
    sch = tr.optimization.get_cosine_with_min_lr_schedule_with_warmup(opt, W, N, min_lr_rate=m)
    got = []
    for _ in range(N):
        got.append(sch.get_last_lr()[0]); sch.step()
    want = adam.schedule("cosine_min", N, peak, W, min_ratio=m)[1:]
    assert np.allclose(got, want, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("m", [0.0, 0.1])
def test_wsd_matches_transformers(m):
    tr = pytest.importorskip("transformers")
    N, W, peak, frac = 300, 10, 2e-5, 0.2
    nd = int(frac * N)
    ns = N - W - nd
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=peak)
    sch = tr.optimization.get_wsd_schedule(opt, num_warmup_steps=W, num_decay_steps=nd,
                                           num_stable_steps=ns, min_lr_ratio=m)
    got = []
    for _ in range(N):
        got.append(sch.get_last_lr()[0]); sch.step()
    want = adam.schedule("wsd", N, peak, W, min_ratio=m, decay_frac=frac)[1:]
    assert np.allclose(got, want, rtol=1e-12, atol=1e-18)
