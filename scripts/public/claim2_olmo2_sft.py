#!/usr/bin/env python3
"""Claim 2: OLMo 2 first SFT release left embedding rows untouched that should have trained.

For base -> SFT (first "preview" release and the retrained Jan-2025 release), count the
rows of model.embed_tokens.weight that are bit-identical to the base row (after the
base's fp32 -> bf16 rounding), then tokenize the SFT mixture with
  (a) the base tokenizer (tokenizer.json: cl100k-style pre-tokenizer regex), and
  (b) the preview SFT's tokenizer (vocab.json + merges.txt only -> GPT-2 byte-level regex),
apply the chat template, and explain which untouched rows "should have trained".

Usage: claim2_olmo2_sft.py [--sizes 7B,13B] [--max-examples N]
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402

OUT = ROOT / "results" / "public"
EMB = "model.embed_tokens.weight"
DATA = "allenai/tulu-3-sft-olmo-2-mixture"
REPOS = {
    "7B": {"base": "allenai/OLMo-2-1124-7B",
           "preview": "allenai/OLMo-2-1124-7B-SFT-Preview",
           "retrained": "allenai/OLMo-2-1124-7B-SFT"},
    "13B": {"base": "allenai/OLMo-2-1124-13B",
            "preview": "allenai/OLMo-2-1124-13B-SFT-Preview",
            "retrained": "allenai/OLMo-2-1124-13B-SFT"},
}
CACHE = paths.cache("trainforensics")


TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
             "special_tokens_map.json"]


def local_tokenizer_dir(repo):
    """Fetch tokenizer files via plain file resolves (avoids /api rate limits) into a
    local dir; files the repo does not ship are simply absent."""
    from huggingface_hub import hf_hub_download
    d = CACHE / "tok" / repo.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    for f in TOK_FILES:
        if (d / f).exists():
            continue
        for k in range(20):
            try:
                p = hf_hub_download(repo, f)
                (d / f).write_bytes(pathlib.Path(p).read_bytes())
                break
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError"):
                    break           # file not shipped by this repo
                time.sleep(30)      # rate limit / transient: retry
    return str(d)


def load_tokenizers():
    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained(local_tokenizer_dir(REPOS["7B"]["base"]))
    # The preview repo ships no tokenizer.json; transformers builds a GPT2 byte-level
    # tokenizer from vocab.json/merges.txt, whose pre-tokenizer is GPT-2's regex.
    prev = AutoTokenizer.from_pretrained(local_tokenizer_dir(REPOS["7B"]["preview"]))
    retr = AutoTokenizer.from_pretrained(local_tokenizer_dir(REPOS["7B"]["retrained"]))
    return base, prev, retr


def pretok_desc(tok):
    try:
        return json.loads(tok.backend_tokenizer.to_str())["pre_tokenizer"]
    except Exception as e:  # noqa: BLE001
        return str(e)


def token_counts(tok, tmpl_tok, name, max_examples=None, bs=2000):
    """Count token ids over the SFT mixture with the chat template applied as text by
    `tmpl_tok` and encoded by `tok` (no extra special tokens: the template adds BOS)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"olmo2_sft_counts_{name}_{max_examples}.npy"
    if cp.exists():
        return np.load(cp), json.loads(cp.with_suffix(".json").read_text())
    from datasets import load_dataset
    ds = load_dataset(DATA, split="train")
    n_total = len(ds)
    if max_examples and max_examples < n_total:
        ds = ds.shuffle(seed=0).select(range(max_examples))
    counts = np.zeros(len(tok) + 512, dtype=np.int64)
    n_tok, n_trunc, t0 = 0, 0, time.time()
    for i in range(0, len(ds), bs):
        msgs = ds[i:i + bs]["messages"]
        texts = [tmpl_tok.apply_chat_template(m, tokenize=False) for m in msgs]
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        for ids in enc:
            n_trunc += len(ids) > 4096
            a = np.asarray(ids, dtype=np.int64)
            counts[: a.max() + 1] += np.bincount(a, minlength=a.max() + 1)
            n_tok += len(a)
        if (i // bs) % 50 == 0:
            print(f"  [{name}] {i + len(msgs)}/{len(ds)} examples, {n_tok / 1e6:.1f}M tokens, "
                  f"{time.time() - t0:.0f}s", flush=True)
    meta = {"n_examples_used": len(ds), "n_examples_total": n_total, "n_tokens": int(n_tok),
            "n_examples_over_4096_tokens": int(n_trunc)}
    np.save(cp, counts)
    cp.with_suffix(".json").write_text(json.dumps(meta))
    return counts, meta


def embed(repo):
    return remote.tensor(repo, EMB)


def compare(base_fp32, ft):
    b16 = base_fp32.to(torch.bfloat16) if ft.dtype == torch.bfloat16 else base_fp32.to(ft.dtype)
    same = (b16.view(torch.int16) == ft.view(torch.int16)).all(1) if ft.dtype == torch.bfloat16 \
        else (b16 == ft).all(1)
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="7B,13B")
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()

    base_tok, prev_tok, retr_tok = load_tokenizers()
    V_tok = len(base_tok)
    res = {"tokenizers": {
        "base_pre_tokenizer": pretok_desc(base_tok),
        "preview_pre_tokenizer": pretok_desc(prev_tok),
        "retrained_pre_tokenizer": pretok_desc(retr_tok),
        "len_base": len(base_tok), "len_preview": len(prev_tok),
        "vocab_identical": base_tok.get_vocab() == prev_tok.get_vocab(),
        "chat_template_preview": prev_tok.chat_template,
    }}
    # chat-turn markers under both tokenizers
    probe = prev_tok.apply_chat_template(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "Hi"},
         {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "x"},
         {"role": "assistant", "content": "y"}], tokenize=False)
    res["template_probe_text"] = probe
    for nm, tk in [("base", base_tok), ("preview", prev_tok)]:
        ids = tk(probe, add_special_tokens=False)["input_ids"]
        res[f"template_probe_{nm}"] = [[i, tk.convert_ids_to_tokens(i)] for i in ids]
        print(nm, [tk.convert_ids_to_tokens(i) for i in ids])

    # data token counts (template applied with the preview chat template; same text for all)
    cb, mb = token_counts(base_tok, prev_tok, "basetok", args.max_examples)
    cp_, mp = token_counts(prev_tok, prev_tok, "previewtok", args.max_examples)
    res["data"] = {"dataset": DATA, "base_tok": mb, "preview_tok": mp}
    in_base = set(np.nonzero(cb)[0].tolist())
    in_prev = set(np.nonzero(cp_)[0].tolist())
    res["data"]["distinct_ids_base_tok"] = len(in_base)
    res["data"]["distinct_ids_preview_tok"] = len(in_prev)
    res["data"]["ids_only_under_base_tok"] = len(in_base - in_prev)
    res["data"]["ids_only_under_preview_tok"] = len(in_prev - in_base)

    # identify the turn-closing token(s): tokens covering '|>' in the base tokenization of markers
    marker_ids = {}
    for role in ["system", "user", "assistant"]:
        s = f"<|{role}|>\n"
        for nm, tk in [("base", base_tok), ("preview", prev_tok)]:
            marker_ids[f"{role}:{nm}"] = [[i, tk.convert_ids_to_tokens(i)]
                                          for i in tk(s, add_special_tokens=False)["input_ids"]]
    res["marker_tokenization"] = marker_ids
    print(json.dumps(marker_ids))
    close_ids = sorted({i for k, v in marker_ids.items() if k.endswith(":base") for i, t in v
                        if "|>" in t})

    for size in args.sizes.split(","):
        R = REPOS[size]
        print(f"\n=== OLMo-2 {size}")
        Wb = embed(R["base"]).float()
        V = Wb.shape[0]
        out = {"repos": R, "rows": V, "tokenizer_len": V_tok}
        for kind in ["preview", "retrained"]:
            Wf = embed(R[kind])
            same = compare(Wb, Wf)
            ids_same = set(torch.nonzero(same).flatten().tolist())
            pad_rows = set(range(V_tok, V))
            real_same = ids_same - pad_rows
            should = sorted(real_same & in_base)                 # present in data (base tok)
            explained_prev = real_same - in_prev                # absent under preview tokenization
            anomalies_moved_but_absent = sorted(set(range(V_tok)) - ids_same - in_prev) \
                if kind == "preview" else sorted(set(range(V_tok)) - ids_same - in_base)
            d = {
                "dtype": str(Wf.dtype),
                "rows_identical_to_base_bf16": len(ids_same),
                "of_which_padding_rows(>= len(tokenizer))": len(ids_same & pad_rows),
                "padding_rows_total": len(pad_rows),
                "real_token_rows_identical": len(real_same),
                "identical_and_present_in_data_base_tok (should have trained)": len(should),
                "identical_and_absent_under_preview_tok": len(explained_prev),
                "identical_but_present_under_preview_tok": len(real_same & in_prev),
                "moved_but_absent_from_data_under_matching_tok": len(anomalies_moved_but_absent),
                "base_token_occurrences_in_never_moved_rows": int(cb[should].sum()),
                "turn_close_tokens": {base_tok.convert_ids_to_tokens(i): {
                    "id": i, "moved": i not in ids_same,
                    "count_base_tok": int(cb[i]), "count_preview_tok": int(cp_[i])}
                    for i in close_ids},
                "eos_endoftext_moved": 100257 not in ids_same,
                "top_should_have_trained": [
                    [int(i), base_tok.convert_ids_to_tokens(int(i)), int(cb[i])]
                    for i in sorted(should, key=lambda i: -cb[i])[:40]],
            }
            # change magnitude of moved rows (for context)
            mv = ~same
            delta = (Wf.float() - Wb.to(Wf.dtype).float())[mv]
            d["moved_rows_median_rel_change"] = float(
                (delta.norm(dim=1) / Wb[mv].norm(dim=1)).median())
            out[kind] = d
            for k, v in d.items():
                if k != "top_should_have_trained":
                    print(f"  {kind}: {k}: {v}")
            print(f"  {kind}: top should-have-trained:", d["top_should_have_trained"][:15])
        res[size] = out
    (OUT / "claim2_olmo2_sft.json").write_text(json.dumps(res, indent=1, default=str))
    print("wrote", OUT / "claim2_olmo2_sft.json")


if __name__ == "__main__":
    main()
