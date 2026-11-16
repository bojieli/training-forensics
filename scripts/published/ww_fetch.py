#!/usr/bin/env python3
"""Fetch the input embeddings of the WebWorld models and their Qwen3 bases (ranged reads) into
~/.cache/tf_webworld/emb as .pt files, and the bases' candidates for lineage."""
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402

remote.CACHE = pathlib.Path.home() / ".cache" / "tf_webworld" / "remote"
OUT = pathlib.Path.home() / ".cache" / "tf_webworld" / "emb"
OUT.mkdir(parents=True, exist_ok=True)
REPOS = sys.argv[1:] or ["Qwen/WebWorld-8B", "Qwen/Qwen3-8B", "Qwen/WebWorld-14B", "Qwen/Qwen3-14B",
                         "Qwen/WebWorld-32B", "Qwen/Qwen3-32B"]
for repo in REPOS:
    p = OUT / (repo.replace("/", "__") + ".pt")
    if p.exists():
        print("have", repo, flush=True)
        continue
    t = remote.tensor(repo, "model.embed_tokens.weight")
    torch.save(t.contiguous(), p)
    print(repo, t.dtype, tuple(t.shape), flush=True)
