#!/usr/bin/env python3
"""Blind schedule estimate for an fp32 public fine-tune (no data): the same reading as
scripts/controlled/score.py read(release='fp32') without the dataset stage.

  python3 scripts/y2026_ft/timing.py NAME PARENT CHILD

Never-updated vectors (pure shrink of the base), total shrink P; per updated vector the
move D read as the intercept of its sorted-moves curve; vectors seen in exactly one
optimizer step (single-touch) give the learning rate at that step; fit_recipe over
schedule kinds / step counts / warm-ups. Writes results/y2026_ft/timing_NAME.json
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import glob
import json
import math
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import blind  # noqa: E402
from trainforensics.diff import row_diff  # noqa: E402

C = paths.cache("tf_y2026")
name, parent, child = sys.argv[1:4]
P_ = json.load(open(ROOT / "results" / "y2026_ft" / "pairs" / f"{name}.json"))


def load(repo, shape, dt):
    for f in glob.glob(str(C / (repo.replace("/", "__") + "@*"))):
        if f.endswith(".part") or "head" in f.split("__")[-2]:
            continue
        need = shape[0] * shape[1] * (4 if dt == "F32" else 2)
        if os.path.getsize(f) == need:
            t = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}[dt]
            return torch.frombuffer(bytearray(open(f, "rb").read()), dtype=t).view(*shape)
    raise FileNotFoundError(repo)


W0 = load(parent, P_["shape_parent"], P_["dtype_parent"])
W1 = load(child, P_["shape_child"], P_["dtype_child"])
V = min(len(W0), len(W1))
W0, W1 = W0[:V].float(), W1[:V].float()
out = dict(name=name, parent=parent, child=child)
rd = row_diff(W0, W1, torch.float32, device="cpu")
decay = rd.decay
out.update(identical=int(rd.identical.sum()), moved=int(rd.moved.sum()), decay=decay)
mv = np.nonzero(rd.moved)[0]
if decay < 1.0:
    keys, vals, cons = blind.decay_lut(W0, W1, ~rd.moved)
    ref, cov = blind.apply_lut(W0[mv], keys, vals, decay)
    W0ref = W0.clone()
    W0ref[mv] = ref
    Dm, Tm, Rm = blind.row_intercepts(W0ref, W1, 1.0, mv)
else:
    Dm, Tm, Rm = blind.row_intercepts(W0, W1, decay, mv)
single = Rm < 3e-3
Dq = Dm[single]
out["single_touch_vectors"] = int(single.sum())
print({k: out[k] for k in ("identical", "moved", "decay", "single_touch_vectors")}, flush=True)
if single.sum() >= 30:
    best, ranked = blind.fit_recipe(Dq, logP=(math.log(decay) if decay < 1.0 else None),
                                    procs=int(os.environ.get("TF_PROCS", "12")))
    out["best"] = best.as_dict()
    out["ranked"] = [r.as_dict() for r in ranked[:5]]
    # the single-touch moves also give the peak learning rate directly (D ~ 31.8 x lr late in a run)
    out["D_quantiles"] = np.quantile(Dq, [0.01, 0.5, 0.99]).tolist()
    if decay < 1.0:
        lr = best.lr()
        out["wd_from_decay"] = blind.wd_from_decay(decay, lr)
    print(out["best"], flush=True)
(ROOT / "results" / "y2026_ft" / f"timing_{name}.json").write_text(json.dumps(out, indent=1, default=float))
