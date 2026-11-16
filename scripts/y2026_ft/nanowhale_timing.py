#!/usr/bin/env python3
"""fp32 release check: does the model card's recipe reproduce the never-updated token
vectors bit for bit? (HuggingFaceTB/nanowhale-100m vs nanowhale-100m-base)

Card: SFT on smol-smoltalk, 3,000 steps, cosine, peak 2e-5, 5% warm-up, batch 32.
  1. never-updated vectors = pure shrinks of the base (cos ~ 1, residual ~ fp32 noise);
  2. shrink factor P from them; weight decay wd from P under the card's schedule;
  3. replay torch AdamW's decoupled decay step by step in fp32 and count bit-exact
     vectors and weights; controls: step count +-1, other warm-ups, other schedules.
Writes results/y2026_ft/nanowhale_timing.json
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics.adam import schedule  # noqa: E402
from trainforensics.blind import wd_from_decay, replay_decay  # noqa: E402

C = pathlib.Path.home() / ".cache" / "tf_y2026"


def load(repo):
    f = [x for x in glob.glob(str(C / (repo.replace("/", "__") + "@*"))) if "embed_tokens" in x][0]
    return torch.frombuffer(bytearray(open(f, "rb").read()), dtype=torch.float32).view(-1, 320)


W0, W1 = load("HuggingFaceTB/nanowhale-100m-base"), load("HuggingFaceTB/nanowhale-100m")
V, d = W0.shape
a, b = W0.double(), W1.double()
r = (a * b).sum(1) / (a * a).sum(1).clamp_min(1e-300)
res_ = (b - r[:, None] * a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-300)
out = dict(vocab=V, dim=d)
qs = np.quantile(res_.numpy(), [0.01, 0.1, 0.25, 0.5, 0.75, 0.9])
out["shrink_residual_quantiles"] = qs.tolist()
shrink = (res_ < 2e-5).numpy()
out["pure_shrink_vectors"] = int(shrink.sum())
P = float(np.median(r.numpy()[shrink])) if shrink.sum() else float("nan")
out["P"] = P
out["shrink_pct"] = 100 * (1 - P)
out["P_spread_q01_q99"] = np.quantile(r.numpy()[shrink], [0.01, 0.99]).tolist() if shrink.sum() else None
print(out, flush=True)
idx = torch.from_numpy(np.nonzero(shrink)[0])
A, B = W0[idx], W1[idx]


def test(kind, N, peak, warm, wd=None, label=None, min_ratio=0.0):
    lr = schedule(kind, N, peak, warm, min_ratio=min_ratio)
    w = wd if wd is not None else wd_from_decay(P, lr)
    x = replay_decay(A, lr, w)
    eqv = int((x == B).all(1).sum())
    eqw = int((x == B).sum())
    ul = ((x.double() - B.double()).abs() / torch.finfo(torch.float32).eps / B.double().abs().clamp_min(1e-30))
    rec = dict(label=label or f"{kind} N={N} peak={peak:g} warm={warm}", wd=w, vectors_exact=eqv, weights_exact=eqw,
               weights_total=int(B.numel()), median_rel_err_in_eps=float(ul.median()),
               p90_rel_err_in_eps=float(torch.quantile(ul.flatten()[:2000000], 0.9)))
    print(rec, flush=True)
    return rec


tests = []
if shrink.sum():
    lr = schedule("cosine", 3000, 2e-5, 150)
    wd_raw = wd_from_decay(P, lr)
    out["wd_raw"] = wd_raw
    snaps = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    wd = min(snaps, key=lambda s: abs(math.log(max(s, 1e-12)) - math.log(max(wd_raw, 1e-12))))
    out["wd_snapped"] = wd
    tests.append(test("cosine", 3000, 2e-5, 150, wd, "card: cosine N=3000 peak=2e-5 warm=150, wd snapped"))
    tests.append(test("cosine", 3000, 2e-5, 150, None, "card schedule, wd fitted exactly"))
    for N in (2999, 3001):
        tests.append(test("cosine", N, 2e-5, 150, wd, f"control: N={N}"))
    for w in (149, 151, 0, 300):
        tests.append(test("cosine", 3000, 2e-5, w, wd, f"control: warm-up {w}"))
    tests.append(test("linear", 3000, 2e-5, 150, wd, "control: linear"))
    tests.append(test("constant", 3000, 2e-5, 150, wd, "control: constant"))
    # small N scan to locate the best step count blind
    best = []
    for N in range(2900, 3101, 10):
        lrN = schedule("cosine", N, 2e-5, math.ceil(0.05 * N))
        x = replay_decay(A, lrN, wd)
        best.append((int((x == B).all(1).sum()), N))
    out["N_scan_warm5pct"] = sorted(best, reverse=True)[:5]
out["tests"] = tests
(ROOT / "results" / "y2026_ft" / "nanowhale_timing.json").write_text(json.dumps(out, indent=1))
