#!/usr/bin/env python3
"""WebWorld models vs their published training set (Qwen/WebWorldData).

Per model: input token vectors of the release vs its Qwen3 base (P = 1: no uniform shrink is
checked first and kept if it reproduces the never-updated vectors exactly), per token the
fraction of numbers that differ; states bit-identical (0), barely changed (<1% of numbers),
updated. Compared with the token counts of the published set (ww_tokens.py), with truncation
at a context length C (a token is in the trained text if its earliest position in any sample is
< C) and the last-token rule (a token occurring only as a sample's final token gets no gradient).

Wrong-data control: the same comparison against another published set the models were not
trained on (s1K-1.1 token counts, results/published/data_s1.1.json), restricted to the same
vocabulary.

usage: ww_compare.py DATA_NPZ [tag]
Writes results/published/webworld/compare<tag>.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "published" / "webworld"
EMB = pathlib.Path.home() / ".cache" / "tf_webworld" / "emb"
PAIRS = [("Qwen/Qwen3-8B", "Qwen/WebWorld-8B"), ("Qwen/Qwen3-14B", "Qwen/WebWorld-14B"),
         ("Qwen/Qwen3-32B", "Qwen/WebWorld-32B")]
CUTS = [4096, 8192, 16384, 20000, 24576, 30000, 32768, 40960, 65536, 131072, 10 ** 9]


def emb(repo):
    p = EMB / (repo.replace("/", "__") + ".pt")
    return torch.load(p) if p.exists() else None


def frac_changed(W0, W1, chunk=8192):
    out = np.zeros(W0.shape[0])
    for a in range(0, W0.shape[0], chunk):
        out[a:a + chunk] = (W0[a:a + chunk] != W1[a:a + chunk]).float().mean(1).numpy()
    return out


def decode(tok, ids, k=25):
    return [[int(i), tok.convert_ids_to_tokens(int(i))] for i in ids[:k]]


def main():
    npz = pathlib.Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    d = np.load(npz)
    chat, minpos, last, pj = d["chat"], d["minpos"], d["last"], d["plain_join"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/WebWorld-8B")
    V = len(chat)
    special = np.zeros(V, bool)
    special[[i for i in tok.all_special_ids if i < V]] = True
    for t, i in tok.get_added_vocab().items():
        if i < V:
            special[i] = True
    res = dict(data=str(npz.name), vocab=V, models={})
    s11 = json.load(open(ROOT / "results" / "published" / "data_s1.1.json"))["counts"]
    s11 = s11[next(iter(s11))]
    wrong = np.zeros(V, bool)
    for k, v in s11.items():
        if int(k) < V and v > 0:
            wrong[int(k)] = True
    for parent, child in PAIRS:
        W0, W1 = emb(parent), emb(child)
        if W0 is None or W1 is None:
            res["models"][child] = dict(parent=parent, status="embedding not fetched")
            continue
        n = min(W0.shape[0], W1.shape[0], V)
        fc = frac_changed(W0[:n], W1[:n])
        bit = fc == 0
        barely = (fc > 0) & (fc < 0.01)
        upd = fc >= 0.01
        reg = ~special[:n]
        m = dict(parent=parent, rows=int(W0.shape[0]), compared=n, bit_identical=int(bit.sum()),
                 barely=int(barely.sum()), updated=int(upd.sum()),
                 special_updated=[t for i, t in decode(tok, np.where(special[:n] & ~bit)[0], 40)],
                 special_bit_identical=[t for i, t in decode(tok, np.where(special[:n] & bit)[0], 40)])
        ch = (fc > 0)
        scan = []
        for C in CUTS:
            present = (chat[:n] > 0) & (minpos[:n] < C) & ((chat[:n] - last[:n]) > 0)
            a = present & reg
            scan.append(dict(cutoff=C, present=int(a.sum()), present_changed=int((a & ch).sum()),
                             present_bit_identical=int((a & bit).sum()),
                             changed_absent=int((~present & reg & ch).sum()),
                             changed_absent_updated=int((~present & reg & upd).sum())))
        m["cutoff_scan"] = scan
        best = min(scan, key=lambda x: x["present_bit_identical"] + x["changed_absent"])
        m["best_cutoff"] = best
        # bracket from earliest positions (regular tokens in the data)
        inds = (chat[:n] > 0) & reg
        upd_mp = minpos[:n][inds & ch]
        nup_mp = minpos[:n][inds & bit]
        m["bracket"] = dict(max_earliest_pos_of_changed=int(upd_mp.max()) if len(upd_mp) else None,
                            min_earliest_pos_of_bit_identical=int(nup_mp.min()) if len(nup_mp) else None,
                            bit_identical_in_data_earliest_pos_quantiles=(
                                {q: int(np.quantile(nup_mp, q)) for q in (0, 0.1, 0.5, 0.9)} if len(nup_mp) else {}))
        C = best["cutoff"]
        present = (chat[:n] > 0) & (minpos[:n] < C) & ((chat[:n] - last[:n]) > 0)
        ca = np.where(~present & reg & ch)[0]
        pb = np.where(present & reg & bit)[0]
        m["changed_absent_examples"] = decode(tok, ca[np.argsort(-fc[ca])], 30)
        m["changed_absent_in_plain_joins"] = int((pj[ca] > 0).sum())
        m["changed_absent_change_frac_median"] = float(np.median(fc[ca])) if len(ca) else None
        m["present_bit_identical_examples"] = [[i, t, int(chat[i]), int(minpos[i])] for i, t in decode(tok, pb[np.argsort(-chat[pb])], 30)]
        m["present_bit_identical_count_max"] = int(chat[pb].max()) if len(pb) else 0
        m["in_data_change_frac_median"] = float(np.median(fc[present & reg & ch])) if (present & reg & ch).any() else None
        # wrong-data control: s1K-1.1 vocabulary as if it were the training set
        w = wrong[:n] & reg
        # samples left out by length (needs ww_lengths.py): present iff some sample of <= L tokens
        # contains the token (drop rule), optionally combined with truncation at C (minpos_L < C)
        lp = npz.parent / "data_webworld_lengths.npz"
        if lp.exists():
            ld = np.load(lp)
            notlast = (chat[:n] - last[:n]) > 0
            drop = []
            for L in [int(k.split("_")[1]) for k in ld.files if k.startswith("minpos_")]:
                mp = ld[f"minpos_{L}"][:n]
                for C in [c for c in CUTS if c <= L] + [10 ** 9]:
                    pr = (mp < C) & notlast & reg
                    drop.append(dict(max_len=L, cutoff=C, present=int(pr.sum()),
                                     present_bit_identical=int((pr & bit).sum()),
                                     changed_absent=int((~pr & reg & ch).sum())))
            m["drop_scan"] = drop
            m["best_drop"] = min(drop, key=lambda x: x["present_bit_identical"] + x["changed_absent"])
        m["wrong_data_control"] = dict(dataset="simplescaling/s1K-1.1 (not WebWorld's data)",
                                       present=int(w.sum()), present_bit_identical=int((w & bit).sum()),
                                       changed_absent=int((~w & reg & ch).sum()))
        res["models"][child] = m
        print(child, {k: m[k] for k in ("bit_identical", "barely", "updated")}, "best", best, "bracket", m["bracket"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / f"compare{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
