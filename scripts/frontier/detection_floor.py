#!/usr/bin/env python3
"""Detection floor for 'bit-identical' claims in 16-bit releases.

A token vector can only be called bit-identical (never updated) if an update would have shown.
For each claimed vector we compute the smallest per-number change that would have altered at least
1, 10 and 100 of its numbers in the 16-bit release (half the 16-bit spacing at the stored value,
sorted). An Adam update moves every number of a token that appears by roughly the learning rate
(typical post-training learning rates: 1e-6 .. 1e-5), so if those floors are far below 1e-6 a token
present in the data would have changed many numbers. Writes results/frontier/detection_floor.json."""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import OUT, DT, pair, records  # noqa: E402

CASES = {
    "DeepSeek-V3 tool-output tokens": ("deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-V3", [128810, 128811, 128812, 128813]),
    "DeepSeek-R1 tool tokens": ("deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-R1", list(range(128806, 128815))),
    "Qwen3-8B vision/grounding tokens": ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B", list(range(151646, 151657))),
    "Qwen3-14B vision/grounding tokens": ("Qwen/Qwen3-14B-Base", "Qwen/Qwen3-14B", list(range(151646, 151657))),
}


def half_spacing(x, dtype):
    mant = {torch.bfloat16: 7, torch.float16: 10}[dtype]
    _, e = torch.frexp(x.abs().to(torch.float32).clamp_min(1e-38))
    return torch.pow(2.0, (e.to(torch.float64) - 1) - mant) / 2


def main():
    recs = records()
    out = {}
    for name, (par, ch, ids) in CASES.items():
        W0, W1, r = pair(par, ch, recs)
        dt = W1.dtype
        P = r["embed"]["P"]
        x = (W0[ids].double() * P).to(torch.float32).to(dt)
        same = bool((x == W1[ids]).all())
        hs = half_spacing(x, dt).sort(dim=1).values
        out[name] = dict(parent=par, release=ch, token_ids=ids, all_bit_identical_after_decay=same, P=P,
                         floor_1_number=float(hs[:, 0].max()), floor_10_numbers=float(hs[:, 9].max()),
                         floor_100_numbers=float(hs[:, 99].max()),
                         numbers_altered_by_1e6_change_min=int((hs < 1e-6).sum(1).min()),
                         dim=int(W0.shape[1]))
        print(name, out[name], flush=True)
    json.dump(out, open(OUT / "detection_floor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
