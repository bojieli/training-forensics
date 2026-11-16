"""Shared helpers for checkpoint-series reading."""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import functools, os, pathlib, re, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
os.environ.setdefault("CKPTF_CACHE", str(paths.cache("trainforensics") / "series"))
from trainforensics import remote  # noqa
from huggingface_hub import HfApi

EMBN = ("embed_tokens.weight", "wte.weight", "embed_in.weight", "word_embeddings.weight")

def step_of(name: str):
    """(stage, substage, step) sort key, or None for non-step branches."""
    m = re.match(r"stage(\d+)-(?:ingredient(\d+)-)?step-?(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2) or 0), int(m.group(3))
    m = re.match(r"step-(\d+)k-", name)
    if m:
        return 1, 0, int(m.group(1)) * 1000
    m = re.match(r"step-?(\d+)", name)
    if m:
        return 1, 0, int(m.group(1))
    m = re.match(r"longctx-step(\d+)$", name)
    if m:
        return 2, 0, int(m.group(1))
    m = re.match(r"(\d+)B$", name)
    if m:
        return 1, 0, int(m.group(1))
    m = re.match(r"ckpt_(\d+)$", name)
    if m:
        return 1, 0, int(m.group(1))
    m = re.match(r"ministage(\d+)_ckpt_(\d+)$", name)
    if m:
        return 2, int(m.group(1)), int(m.group(2))
    m = re.match(r"CrystalCoder_phase(\d+)_checkpoint_(\d+)$", name)
    if m:
        return int(m.group(1)), 0, int(m.group(2))
    m = re.match(r"mdl_phase(\d+)_step_(\d+)$", name)
    if m:
        return 10 + int(m.group(1)), 0, int(m.group(2))
    return None

@functools.lru_cache(maxsize=None)
def branches(repo):
    out = []
    for b in HfApi().list_repo_refs(repo).branches:
        k = step_of(b.name)
        if k is not None and "from" not in b.name:
            out.append((k, b.name))
    return sorted(out)

@functools.lru_cache(maxsize=None)
def emb_name(repo, rev):
    idx = remote.index(repo, rev)
    return [k for k in idx if k.endswith(EMBN)][0]

def emb(repo, rev, rows=None):
    return remote.tensor(repo, emb_name(repo, rev), rev, rows=rows)

# --- allow the extra (code-free) reconstructors used by newer torch.save files ---
_orig_find = remote._SafeUnpickler.find_class
def _find(self, module, name):
    if (module, name) == ("torch._tensor", "_rebuild_from_type_v2"):
        return lambda func, new_type, args, state: func(*args)
    if (module, name) == ("torch._utils", "_rebuild_parameter"):
        return lambda data, requires_grad, hooks: data
    if (module, name) in (("torch", "Tensor"), ("torch.nn.parameter", "Parameter")):
        return name
    return _orig_find(self, module, name)
remote._SafeUnpickler.find_class = _find
