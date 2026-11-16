#!/usr/bin/env python3
"""Claim 4 follow-up: are Qwen3-30B-A3B's special-token vectors trained updates of the base
vectors, or replaced by new vectors? If a release vector is independent of the base vector with the
same size, the change D = new - old has cosine -1/sqrt(2) = -0.707 with the old vector and the
new vector has cosine ~0 with the old one. Also compares the 30B release with the 2507 releases.
Appends to results/frontier/qwen3_vision.json under "replacement_test"."""
from __future__ import annotations

import json
import sys

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import OUT, DT, load, rec_for, records  # noqa: E402

SPECIAL = list(range(151643, 151669))
NAMES = {151643: "<|endoftext|>", 151644: "<|im_start|>", 151645: "<|im_end|>", 151652: "<|vision_start|>",
         151655: "<|image_pad|>", 151657: "<tool_call>", 151659: "<|fim_prefix|>", 151667: "<think>"}


def main():
    recs = records()
    base = "Qwen/Qwen3-30B-A3B-Base"
    r = rec_for(base, "Qwen/Qwen3-30B-A3B", recs)
    sh, dt = r["shape_parent"], DT[r["dtype_parent"]]
    W0 = load(base, sh, dt).double()
    rels = {c: load(c, rec_for(base, c, recs)["shape_child"], dt).double()
            for c in ("Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507")}
    cs = torch.nn.functional.cosine_similarity
    res = {}
    rng = torch.Generator().manual_seed(0)
    ordi = torch.randperm(150000, generator=rng)[:2000]
    for c, W in rels.items():
        res[c] = dict(
            special_cos_new_vs_base={NAMES.get(i, str(i)): float(cs(W[i:i + 1], W0[i:i + 1])) for i in SPECIAL},
            special_cos_new_vs_base_median=float(cs(W[SPECIAL], W0[SPECIAL]).median()),
            special_change_cos_with_base_median=float(cs(W[SPECIAL] - W0[SPECIAL], W0[SPECIAL]).median()),
            ordinary_cos_new_vs_base_median=float(cs(W[ordi], W0[ordi]).median()),
            special_norm_ratio_median=float((W[SPECIAL].norm(dim=1) / W0[SPECIAL].norm(dim=1)).median()))
    a, b = rels["Qwen/Qwen3-30B-A3B"], rels["Qwen/Qwen3-30B-A3B-Instruct-2507"]
    res["30B vs Instruct-2507 special cos median"] = float(cs(a[SPECIAL], b[SPECIAL]).median())
    res["30B vs Instruct-2507 ordinary cos median"] = float(cs(a[ordi], b[ordi]).median())
    # same check for 8B as a control: special vectors should be base * P
    r8 = rec_for("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B", recs)
    A = load("Qwen/Qwen3-8B-Base", r8["shape_parent"], dt).double()
    B = load("Qwen/Qwen3-8B", r8["shape_child"], dt).double()
    res["control Qwen3-8B special cos new vs base median"] = float(cs(B[SPECIAL], A[SPECIAL]).median())
    d = json.load(open(OUT / "qwen3_vision.json"))
    d["replacement_test"] = res
    json.dump(d, open(OUT / "qwen3_vision.json", "w"), indent=1)
    for k, v in res.items():
        print(k, v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "special_cos_new_vs_base"})


if __name__ == "__main__":
    main()
