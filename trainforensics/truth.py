"""Ground-truth helpers for controlled runs (scripts/controlled/finetune.py)."""
from __future__ import annotations

import json
import pathlib

import numpy as np
from safetensors.torch import load_file


class Run:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.t = json.load(open(self.path / "truth.json"))
        self.tokens = {int(k): v for k, v in json.load(open(self.path / "tokens.json")).items()}
        self.emb = load_file(str(self.path / "emb.safetensors"))
        self.lr = np.array([0.0] + self.t["lr"])           # 1-based
        self.N = self.t["steps"]
        self.V = self.emb["W0"].shape[0]

    def doc_tokens(self, doc_id):
        """Tokens whose input-embedding row gets a gradient from this document:
        every position except the last (whose output has no label)."""
        return self.tokens[doc_id][:-1]

    def step_docs(self, s):                    # 1-based step -> source doc ids
        return [self.t["trained"][k] for k in self.t["docs_per_step"][s - 1]]

    def token_steps(self):
        """token id -> sorted list of 1-based steps whose batch contains it."""
        ts: dict[int, list[int]] = {}
        for s in range(1, self.N + 1):
            toks = set()
            for d in self.step_docs(s):
                toks.update(self.doc_tokens(d))
            for t in toks:
                ts.setdefault(t, []).append(s)
        return ts

    def seen_mask(self):
        m = np.zeros(self.V, bool)
        for t in self.token_steps():
            m[t] = True
        return m
