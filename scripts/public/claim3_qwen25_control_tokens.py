#!/usr/bin/env python3
"""Claim 3: which of Qwen2.5's control tokens did instruct tuning touch?

For each size, read model.embed_tokens.weight of Qwen/Qwen2.5-<S> and
Qwen/Qwen2.5-<S>-Instruct (bf16), and compare rows bit for bit:
  * the 22 added control tokens (ids from tokenizer_config.json added_tokens_decoder),
  * every vocabulary row (how many moved at all),
  * untouched rows: bit-identical, or uniformly rescaled (weight decay)?
Small sizes (0.5B/1.5B/3B) tie embeddings to the LM head, so every row gets
gradient through the softmax; we check them briefly.
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402

OUT = ROOT / "results" / "public"
EMB = "model.embed_tokens.weight"
BIG = ["7B", "14B", "32B", "72B"]
SMALL = ["0.5B", "1.5B", "3B"]
CACHE = pathlib.Path.home() / ".cache" / "trainforensics"


def added_tokens(repo):
    from huggingface_hub import hf_hub_download
    tc = json.loads(pathlib.Path(hf_hub_download(repo, "tokenizer_config.json")).read_text())
    return {int(k): v["content"] for k, v in tc["added_tokens_decoder"].items()}


def tokenizer_len(repo):
    from huggingface_hub import hf_hub_download
    tj = json.loads(pathlib.Path(hf_hub_download(repo, "tokenizer.json")).read_text())
    return len(tj["model"]["vocab"]), tj


def row_report(Wb, Wi, ids):
    out = {}
    for i in ids:
        a, b = Wb[i], Wi[i]
        same = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        d = (b.float() - a.float())
        out[i] = {"bit_identical": same, "n_coords_changed": int((d != 0).sum()),
                  "rel_change": float(d.norm() / a.float().norm())}
    return out


def analyse(size):
    base, inst = f"Qwen/Qwen2.5-{size}", f"Qwen/Qwen2.5-{size}-Instruct"
    Wb = remote.tensor(base, EMB)
    Wi = remote.tensor(inst, EMB)
    assert Wb.shape == Wi.shape, (Wb.shape, Wi.shape)
    V = Wb.shape[0]
    same = (Wb.view(torch.int16) == Wi.view(torch.int16)).all(1)
    n_vocab, _ = tokenizer_len(base)
    at = added_tokens(inst)
    ctrl = sorted(at)
    rep = row_report(Wb, Wi, ctrl)
    moved_ctrl = [at[i] for i in ctrl if not rep[i]["bit_identical"]]
    # untouched rows: bit-identical or rescaled?  Fit Wi = c*Wb on identical-looking rows
    real = torch.arange(V) < n_vocab
    n_moved_real = int((~same & real).sum())
    pad = torch.arange(V) >= max(ctrl) + 1
    # scale estimate over all rows (if WD were applied, untouched rows would be scaled by c<1)
    Wbf, Wif = Wb.float(), Wi.float()
    c_untouched = float((Wbf[same] * Wif[same]).sum() / (Wbf[same] ** 2).sum()) if same.any() else None
    delta_norm = (Wif - Wbf).norm(dim=1) / Wbf.norm(dim=1).clamp_min(1e-12)
    res = {
        "base": base, "instruct": inst, "rows": V, "hidden": Wb.shape[1], "dtype": str(Wb.dtype),
        "bpe_vocab_size": n_vocab, "n_added_tokens": len(ctrl),
        "added_token_ids": [ctrl[0], ctrl[-1]],
        "control_tokens": {at[i]: dict(id=i, **rep[i]) for i in ctrl},
        "control_tokens_moved": moved_ctrl,
        "n_control_tokens_moved": len(moved_ctrl),
        "rows_moved_total": int((~same).sum()),
        "rows_moved_bpe_vocab": n_moved_real,
        "rows_bit_identical_total": int(same.sum()),
        "padding_rows": [max(ctrl) + 1, V], "padding_rows_moved": int((~same & pad).sum()),
        "lsq_scale_of_identical_rows": c_untouched,
        "moved_rows_rel_change_quantiles": {q: float(delta_norm[~same].quantile(q))
                                            for q in (0.01, 0.1, 0.5, 0.9, 0.99)} if (~same).any() else None,
    }
    if size == "7B":  # keep for claim 4
        CACHE.mkdir(parents=True, exist_ok=True)
        torch.save({"base": Wb, "inst": Wi}, CACHE / "qwen25_7b_embed_pair.pt")
    return res


def analyse_small(size):
    base, inst = f"Qwen/Qwen2.5-{size}", f"Qwen/Qwen2.5-{size}-Instruct"
    idx = remote.index(base)
    tied = "lm_head.weight" not in idx
    Wb = remote.tensor(base, EMB)
    Wi = remote.tensor(inst, EMB)
    same = (Wb.view(torch.int16) == Wi.view(torch.int16)).all(1)
    at = added_tokens(inst)
    ctrl = sorted(at)
    rep = row_report(Wb, Wi, ctrl)
    return {"base": base, "tied_embeddings": tied, "rows": Wb.shape[0],
            "rows_bit_identical": int(same.sum()), "rows_moved": int((~same).sum()),
            "control_tokens_moved": [at[i] for i in ctrl if not rep[i]["bit_identical"]],
            "control_tokens_rel_change": {at[i]: rep[i]["rel_change"] for i in ctrl}}


def main():
    res = {"big": {}, "small": {}}
    for s in BIG:
        r = analyse(s)
        res["big"][s] = r
        print(f"\n=== Qwen2.5-{s}: rows {r['rows']} (BPE {r['bpe_vocab_size']}, added "
              f"{r['n_added_tokens']} ids {r['added_token_ids']})")
        print(f"  control tokens moved ({r['n_control_tokens_moved']}): {r['control_tokens_moved']}")
        for t, d in r["control_tokens"].items():
            print(f"    {d['id']} {t:22s} identical={d['bit_identical']} changed_coords={d['n_coords_changed']} "
                  f"rel={d['rel_change']:.4f}")
        print(f"  rows moved total={r['rows_moved_total']} (BPE vocab {r['rows_moved_bpe_vocab']}), "
              f"bit-identical={r['rows_bit_identical_total']}, padding moved={r['padding_rows_moved']}, "
              f"LSQ scale of identical rows={r['lsq_scale_of_identical_rows']}")
        print(f"  moved-row rel change quantiles: {r['moved_rows_rel_change_quantiles']}")
    sets = {s: tuple(res["big"][s]["control_tokens_moved"]) for s in BIG}
    res["same_set_across_sizes"] = len(set(sets.values())) == 1
    print("\nsame moved control-token set at all big sizes:", res["same_set_across_sizes"], sets)
    for s in SMALL:
        try:
            r = analyse_small(s)
        except Exception as e:  # noqa: BLE001
            r = {"error": str(e)}
        res["small"][s] = r
        print(f"\n--- Qwen2.5-{s}: {r}")
    (OUT / "claim3_qwen25_control_tokens.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / "claim3_qwen25_control_tokens.json")


if __name__ == "__main__":
    main()
