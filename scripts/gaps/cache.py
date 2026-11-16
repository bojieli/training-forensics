#!/usr/bin/env python3
"""Run the blind reading (score.read) once per run and cache what the gap
studies need: moved rows, per-row single-touch move D and flatness, the read
recipe, and per-row quantiles of |delta| against the decay-mapped start.
Nothing from truth.json is used here.

  python3 scripts/gaps/cache.py runs/ctl/s05_swap [...]
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "controlled"))
import score as S  # noqa: E402
from trainforensics import blind  # noqa: E402

OUT = ROOT / "runs" / "gaps" / "cache"


def ref_start(W0, W1, moved, decay):
    """decay-mapped start: exact fp32 decay map (LUT) where the run had wd."""
    if decay >= 1.0:
        return W0.clone()
    keys, vals, _ = blind.decay_lut(W0, W1, ~moved)
    ref, _ = blind.apply_lut(W0, keys, vals, decay)
    return ref


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for r in sys.argv[1:]:
        run = pathlib.Path(r)
        dst = OUT / f"{run.name}.npz"
        if dst.exists():
            print("have", dst); continue
        out = S.read(run, "fp32")
        rd = out["_rd"]
        W0, W1 = out["_W"]
        ref = ref_start(W0, W1, rd.moved, out["decay"])
        qs = torch.tensor([0.5, 0.9, 0.99, 1.0], dtype=torch.float64)
        Q = np.zeros((W0.shape[0], 4))
        dev = "cuda"
        for a in range(0, W0.shape[0], 8192):
            d = (W1[a:a + 8192].double().to(dev) - ref[a:a + 8192].double().to(dev)).abs()
            Q[a:a + 8192] = torch.quantile(d, qs.to(dev), dim=1).T.cpu().numpy()
        rec = dict(out["recipe"])
        np.savez_compressed(dst, moved=rd.moved, D=out["_D"], single=rd.flat_frac > 0.9, Q=Q,
                            decay=out["decay"], recipe=json.dumps(rec, default=float))
        print("cached", dst, rec, flush=True)


if __name__ == "__main__":
    main()
