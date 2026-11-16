#!/usr/bin/env python3
"""Token occurrence counts of Qwen/WebWorldData (WebWorld_Training.jsonl), the published training
set of the WebWorld models, tokenized three ways:

  chat   : the Qwen3 chat template shipped with WebWorld (human -> user, gpt -> assistant,
           a leading "system" turn kept as system), rendered by apply_chat_template and
           tokenized whole; per token: count, earliest position, count as the final token
  slots  : (only with WW_SLOTS=1) every message content tokenized on its own (templates that
           encode each slot separately, e.g. LLaMA-Factory); on a 7,680-sample test it gave
           110,921 distinct tokens against 110,925 for chat, so the full run omits it
  plain  : the message contents joined by "\\n" without chat markers (the format found for
           Arete-27B); only the tokens formed across a join are counted extra, from a
           window of 64 characters on each side

Runs over byte-range chunks in parallel. Writes results/published/webworld/data_webworld.npz
and data_webworld.json (summary).

usage: ww_tokens.py PATH_TO_JSONL [n_procs] [limit_lines_per_chunk]
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
from multiprocessing import Pool

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "published" / "webworld"
TOK = "Qwen/WebWorld-8B"
ROLE = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
_tok = None


def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOK)
    return _tok


def chunks(path, n):
    size = os.path.getsize(path)
    bounds = [0]
    with open(path, "rb") as f:
        for i in range(1, n):
            f.seek(size * i // n)
            f.readline()
            bounds.append(f.tell())
    bounds.append(size)
    return [(path, bounds[i], bounds[i + 1]) for i in range(n) if bounds[i + 1] > bounds[i]]


def work(args):
    path, a, b, limit = args
    t = tok()
    V = len(t)
    cnt = {k: np.zeros(V, np.int64) for k in ("chat", "slots", "plain_join")}
    minpos = np.full(V, np.iinfo(np.int32).max, np.int64)
    last = np.zeros(V, np.int64)
    lens = []
    n = 0
    roles = {}
    with open(path, "rb") as f:
        f.seek(a)
        while f.tell() < b:
            line = f.readline()
            if not line.strip():
                continue
            d = json.loads(line)
            conv = d.get("conversations") or d.get("messages")
            msgs = []
            for m in conv:
                r = m.get("from", m.get("role"))
                roles[r] = roles.get(r, 0) + 1
                msgs.append({"role": ROLE.get(r, r), "content": m.get("value", m.get("content", ""))})
            text = t.apply_chat_template(msgs, tokenize=False)
            ids = np.asarray(t(text, add_special_tokens=False)["input_ids"], np.int64)
            cnt["chat"] += np.bincount(ids, minlength=V)[:V]
            pos = np.arange(len(ids))
            np.minimum.at(minpos, ids, pos)
            last[ids[-1]] += 1
            lens.append(len(ids))
            # slots: each content alone, plus role headers
            if os.environ.get("WW_SLOTS"):
                enc = t([m["content"] for m in msgs], add_special_tokens=False)["input_ids"]
                for e in enc:
                    if e:
                        cnt["slots"] += np.bincount(np.asarray(e, np.int64), minlength=V)[:V]
            # plain join: tokens formed across "\n" joins (windowed)
            for x, y in zip(msgs[:-1], msgs[1:]):
                w = x["content"][-64:] + "\n" + y["content"][:64]
                e = t(w, add_special_tokens=False)["input_ids"]
                if e:
                    cnt["plain_join"] += np.bincount(np.asarray(e, np.int64), minlength=V)[:V]
            n += 1
            if limit and n >= limit:
                break
    return cnt, minpos, last, np.asarray(lens, np.int64), n, roles


def main():
    path = sys.argv[1]
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    nchunks = procs * 8
    jobs = [(p, a, b, limit) for p, a, b in chunks(path, nchunks)]
    V = len(tok())
    tot = {k: np.zeros(V, np.int64) for k in ("chat", "slots", "plain_join")}
    minpos = np.full(V, np.iinfo(np.int32).max, np.int64)
    last = np.zeros(V, np.int64)
    lens = []
    n = 0
    roles = {}
    with Pool(procs) as pool:
        for i, (c, mp, la, ln, k, ro) in enumerate(pool.imap_unordered(work, jobs)):
            for key in tot:
                tot[key] += c[key]
            minpos = np.minimum(minpos, mp)
            last += la
            lens.append(ln)
            n += k
            for r, v in ro.items():
                roles[r] = roles.get(r, 0) + v
            print(f"chunk {i + 1}/{len(jobs)} samples {n:,}", flush=True)
    lens = np.concatenate(lens) if lens else np.zeros(0, np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_sample" if limit else ""
    np.savez_compressed(OUT / f"data_webworld{tag}.npz", chat=tot["chat"], slots=tot["slots"],
                        plain_join=tot["plain_join"], minpos=minpos, last=last, lens=lens)
    summ = dict(dataset="Qwen/WebWorldData", file="WebWorld_Training.jsonl", tokenizer=TOK, vocab=V,
                samples=n, roles=roles, tokens_chat=int(lens.sum()),
                len_quantiles={q: int(np.quantile(lens, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)} if len(lens) else {},
                over={L: int((lens > L).sum()) for L in (8192, 16384, 24576, 32768, 40960)},
                distinct={k: int((v > 0).sum()) for k, v in tot.items()})
    json.dump(summ, open(OUT / f"data_webworld{tag}.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
