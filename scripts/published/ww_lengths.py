#!/usr/bin/env python3
"""Second pass over Qwen/WebWorldData (same tokenization as ww_tokens.py, Qwen3 chat template,
whole string): per token, the length of the shortest sample that contains it (min_len), and,
for samples no longer than L tokens, the token's earliest position (minpos_L). This separates
"long samples were truncated at C" from "samples longer than L were left out".

usage: ww_lengths.py PATH_TO_JSONL [n_procs]
Writes results/published/webworld/data_webworld_lengths.npz
"""
from __future__ import annotations

import json
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ww_tokens import OUT, ROLE, chunks, tok  # noqa: E402

LS = [8192, 16384, 20000, 24576, 30000, 32768, 40960, 65536]
BIG = np.iinfo(np.int64).max


def work(args):
    path, a, b = args
    t = tok()
    V = len(t)
    min_len = np.full(V, BIG, np.int64)
    minpos = {L: np.full(V, BIG, np.int64) for L in LS}
    with open(path, "rb") as f:
        f.seek(a)
        while f.tell() < b:
            line = f.readline()
            if not line.strip():
                continue
            d = json.loads(line)
            conv = d.get("conversations") or d.get("messages")
            msgs = [{"role": ROLE.get(m.get("from", m.get("role")), m.get("from", m.get("role"))),
                     "content": m.get("value", m.get("content", ""))} for m in conv]
            ids = np.asarray(t(t.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"], np.int64)
            n = len(ids)
            u, first = np.unique(ids, return_index=True)
            np.minimum.at(min_len, u, n)
            for L in LS:
                if n <= L:
                    np.minimum.at(minpos[L], u, first)
    return min_len, minpos


def main():
    path = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 28
    jobs = chunks(path, procs * 8)
    V = len(tok())
    min_len = np.full(V, BIG, np.int64)
    minpos = {L: np.full(V, BIG, np.int64) for L in LS}
    with Pool(procs) as pool:
        for i, (ml, mp) in enumerate(pool.imap_unordered(work, jobs)):
            min_len = np.minimum(min_len, ml)
            for L in LS:
                minpos[L] = np.minimum(minpos[L], mp[L])
            print(f"chunk {i + 1}/{len(jobs)}", flush=True)
    np.savez_compressed(OUT / "data_webworld_lengths.npz", min_len=min_len,
                        **{f"minpos_{L}": minpos[L] for L in LS})
    print("done")


if __name__ == "__main__":
    main()
