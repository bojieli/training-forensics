"""Where the scripts look for things outside the repo.

Two environment variables cover everything (both optional):

  CKPTF_REPO  checkout of https://github.com/bojieli/checkpoint-forensics
              (fingerprint cards and its results/summary.json).
              Default: ~/checkpoint-forensics
  TF_CACHE    root for every download cache (embedding tensors, tokenizers,
              per-token masks, dataset copies). When set, cache(name) is
              $TF_CACHE/name; when unset, the historical locations under
              ~/.cache are used (~/.cache/trainforensics, ~/.cache/tf_published,
              ~/.cache/tf_y2026, ~/.cache/tf_webworld, ...).

CKPTF_CACHE, read by trainforensics.remote and scripts/roster/sweep.py, still
overrides the embedding cache directly; its default is cache("trainforensics")/"emb".
"""
from __future__ import annotations

import os
import pathlib

HOME = pathlib.Path.home()
REPO = pathlib.Path(__file__).resolve().parents[1]
CKPTF_REPO = pathlib.Path(os.environ.get("CKPTF_REPO", HOME / "checkpoint-forensics")).expanduser()


def cache(name: str = "trainforensics") -> pathlib.Path:
    root = os.environ.get("TF_CACHE")
    if root:
        return pathlib.Path(root).expanduser() / name
    return HOME / ".cache" / name


EMB = pathlib.Path(os.environ.get("CKPTF_CACHE", cache("trainforensics") / "emb")).expanduser()
CARDS = CKPTF_REPO / "ckptforensics" / "cards"
CKPTF_SUMMARY = CKPTF_REPO / "results" / "summary.json"


def revisions() -> dict:
    """Pinned Hugging Face revisions (revisions.json at the repo root): {repo: sha}."""
    import json
    p = REPO / "revisions.json"
    if not p.exists():
        return {}
    return {k: v["sha"] for k, v in json.load(open(p)).get("repos", {}).items() if v.get("sha")}


def revision(repo: str, default=None):
    """Pinned revision for repo, unless TF_UNPINNED=1 (then default, i.e. main)."""
    if os.environ.get("TF_UNPINNED") == "1":
        return default
    return revisions().get(repo, default)
