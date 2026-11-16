"""Fetch embedding (and head) tensors of the published-data fine-tunes into a dedicated cache."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import os, sys, pathlib
sys.path.insert(0, ".")
from trainforensics import remote
remote.CACHE = paths.cache("tf_published")
for repo in sys.argv[1:]:
    for n in ("model.embed_tokens.weight",):
        t = remote.tensor(repo, n)
        print(repo, n, t.dtype, tuple(t.shape), flush=True)
