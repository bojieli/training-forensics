"""Forward model of how AdamW moves one embedding row.

A row whose token appears in a set of optimizer steps S gets a gradient only at
those steps. Between and after them its first and second moments decay, so the
row keeps moving for a few dozen steps. With decoupled weight decay lambda every
row (touched or not) is also multiplied by (1 - lr_t * lambda) each step.

Notation: steps are 1-based like PyTorch's Adam (state['step'] after increment).
lr[t] is the learning rate used at step t (lr[0] is unused).

For one touch at step s with gradient g (and eps negligible against |g|), the
update at step t >= s is
    u_t = sign(g) * a(t, s),
    a(t, s) = [(1-b1) b1^(t-s) / (1-b1^t)] / sqrt[(1-b2) b2^(t-s) / (1-b2^t)]
independent of |g|. The row ends at
    x_N = x_0 P(1,N) - sum_t lr_t u_t P(t+1,N),    P(a,b) = prod_{t=a..b} (1 - lr_t lambda)
so every coordinate moves by the same magnitude D(s) relative to the decayed
start, with sign -sign(g_i).  With constant lr and s large,
    D / lr -> (1-b1)/sqrt(1-b2) / (1 - b1/sqrt(b2)) = 31.77 for (0.9, 0.999).
"""
from __future__ import annotations

import math
import numpy as np


def saturation_constant(b1=0.9, b2=0.999) -> float:
    return (1 - b1) / math.sqrt(1 - b2) / (1 - b1 / math.sqrt(b2))


def decay_products(lr: np.ndarray, wd: float) -> np.ndarray:
    """suffix[t] = prod_{u=t..N} (1 - lr_u wd), with suffix[N+1] = 1 (float64)."""
    N = len(lr) - 1
    f = 1.0 - np.asarray(lr, dtype=np.float64) * wd
    f[0] = 1.0
    suf = np.ones(N + 2)
    for t in range(N, 0, -1):
        suf[t] = suf[t + 1] * f[t]
    return suf


def single_touch_moves(lr: np.ndarray, b1=0.9, b2=0.999, wd=0.0, horizon: int | None = None) -> np.ndarray:
    """D[s] for s = 1..N: magnitude a single-touch-at-step-s row ends up moved by
    (relative to its decayed starting value), assuming eps << |g|.
    Returns array of length N+1 (D[0] = 0)."""
    lr = np.asarray(lr, dtype=np.float64)
    N = len(lr) - 1
    suf = decay_products(lr, wd)
    if horizon is None:
        # contributions vanish like (b1/sqrt(b2))^k; 1e-12 relative after ~260 steps
        horizon = int(math.ceil(math.log(1e-13) / math.log(b1 / math.sqrt(b2)))) + 1
    t = np.arange(N + 1, dtype=np.float64)
    bc1 = 1 - b1 ** t
    bc2 = 1 - b2 ** t
    D = np.zeros(N + 1)
    for s in range(1, N + 1):
        te = min(N, s + horizon)
        k = np.arange(0, te - s + 1)
        tt = s + k
        a = ((1 - b1) * b1 ** k / bc1[tt]) / np.sqrt((1 - b2) * b2 ** k / bc2[tt])
        D[s] = np.sum(lr[tt] * a * suf[tt + 1])
    return D


def single_touch_moves_eps(lr, gabs: np.ndarray, s: int, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, horizon=400):
    """Move of each coordinate with gradient magnitude gabs (vector) touched once at s,
    keeping eps: used for the large-batch / tiny-gradient regime."""
    lr = np.asarray(lr, dtype=np.float64)
    N = len(lr) - 1
    suf = decay_products(lr, wd)
    out = np.zeros_like(gabs, dtype=np.float64)
    for t in range(s, min(N, s + horizon) + 1):
        k = t - s
        m = (1 - b1) * b1 ** k * gabs / (1 - b1 ** t)
        v = (1 - b2) * b2 ** k * gabs ** 2 / (1 - b2 ** t)
        out += lr[t] * m / (np.sqrt(v) + eps) * suf[t + 1]
    return out


# ---------------------------------------------------------------- schedules
def schedule(kind: str, N: int, peak: float, warmup: int = 0, min_ratio: float = 0.0,
             decay_frac: float = 0.2, num_cycles: float = 0.5) -> np.ndarray:
    """Learning rate used at each 1-based step t = 1..N, matching the
    transformers schedulers (LambdaLR evaluated at current_step = t-1)."""
    lr = np.zeros(N + 1)
    for t in range(1, N + 1):
        c = t - 1
        if c < warmup:
            f = c / max(1, warmup)
        else:
            p = (c - warmup) / max(1, N - warmup)
            if kind == "constant":
                f = 1.0
            elif kind == "linear":
                f = max(0.0, (N - c) / max(1, N - warmup))
            elif kind == "cosine":
                f = max(0.0, 0.5 * (1 + math.cos(math.pi * 2 * num_cycles * p)))
            elif kind == "cosine_min":
                f = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
            elif kind == "wsd":
                nd = int(decay_frac * N)
                ns = N - warmup - nd
                if c < warmup + ns:
                    f = 1.0
                else:
                    q = (c - warmup - ns) / max(1, nd)
                    f = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * q))
            else:
                raise ValueError(kind)
        lr[t] = peak * f
    return lr


if __name__ == "__main__":
    print("saturation constant (0.9, 0.999):", saturation_constant())
    lr = schedule("constant", 5000, 1e-4)
    D = single_touch_moves(lr)
    for s in (1, 10, 100, 1000, 3000, 4000, 4990, 5000):
        print(f"s={s:5d}  D/lr = {D[s] / 1e-4:.3f}")
