#!/usr/bin/env python3
"""For 16-bit embeddings in quantized releases that differ from the official release they are named after,
look for the official release they do equal (sibling versions of the same model).
Writes results/quant_merge/quant_lineage.json."""
from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from trainforensics import remote  # noqa: E402
from quant import FAMILIES, key_for  # noqa: E402
from report import load_quant  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
SIBLINGS = {
    "DeepSeek": ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324", "deepseek-ai/DeepSeek-R1",
                 "deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2-Exp-Base"],
    "Qwen3-235B": ["Qwen/Qwen3-235B-A22B", "Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Thinking-2507"],
    "Kimi": ["moonshotai/Kimi-K2-Base", "moonshotai/Kimi-K2-Thinking", "moonshotai/Kimi-K2.5"],
    "GLM-5": ["zai-org/GLM-5.2", "zai-org/GLM-5.3", "zai-org/GLM-5.3-BF16"],
}


def main():
    out = []
    for r in load_quant():
        e = r["tensors"].get("embed", {})
        if e.get("equals_release_cast") is not False or r["kind"] != "st":
            continue
        x = remote.tensor(r["repo"], key_for(r["repo"], "embed"))
        sib = next((v for k, v in SIBLINGS.items() if r["family"].startswith(k)), [FAMILIES[r["family"]]["child"]])
        res = dict(family=r["family"], repo=r["repo"], named_after=FAMILIES[r["family"]]["child"], matches=[])
        for s in sib:
            try:
                c = remote.tensor(s, key_for(s, "embed"))
            except Exception:
                continue
            if c.shape != x.shape:
                continue
            eq = int((x == c.to(x.dtype)).all(1).sum())
            res["matches"].append(dict(release=s, token_vectors_equal=eq, of=int(x.shape[0])))
        out.append(res)
        print(json.dumps(res), flush=True)
    json.dump(out, open(ROOT / "results/quant_merge/quant_lineage.json", "w"), indent=1)


if __name__ == "__main__":
    main()
