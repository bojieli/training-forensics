#!/usr/bin/env python3
"""Does the token record survive quantized releases?

For each quantized release of a roster fine-tune we ask
  1. is the embedding (and lm_head) still 16-bit, and is it the 16-bit release
     bit for bit (after the cast to the stored dtype)?
  2. can the moved / never-trained rows still be read: against the bf16 base
     (16-bit releases), against the base quantized with the same tool
     (quantized releases), and against the bf16 base re-quantized by us
     (when we can reproduce the quantizer bit for bit)?
Ground truth is the moved-row set read from the bf16 child vs the bf16 base.
Writes results/quant_merge/quant.json.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from trainforensics import remote  # noqa: E402
from gguf_remote import RemoteGGUF  # noqa: E402

remote._DT.setdefault("U32", (torch.int32, 4))

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "quant_merge" / "quant.json"

FAMILIES = {
    "Qwen2.5-7B-Instruct": dict(child="Qwen/Qwen2.5-7B-Instruct", base="Qwen/Qwen2.5-7B", quants=[
        ("st", "Qwen/Qwen2.5-7B-Instruct-AWQ"), ("st", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"),
        ("st", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"), ("st", "RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic"),
        ("st", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"),
        ("mlx", "mlx-community/Qwen2.5-7B-Instruct-4bit", "mlx-community/Qwen2.5-7B-4bit"),
        ("mlx", "mlx-community/Qwen2.5-7B-Instruct-8bit", "mlx-community/Qwen2.5-7B-8bit"),
        ("gguf", "mradermacher/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct.Q8_0.gguf",
         "mradermacher/Qwen2.5-7B-GGUF:Qwen2.5-7B.Q8_0.gguf"),
        ("gguf", "mradermacher/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct.Q4_K_M.gguf",
         "mradermacher/Qwen2.5-7B-GGUF:Qwen2.5-7B.Q4_K_M.gguf"),
        ("gguf", "mradermacher/Qwen2.5-7B-Instruct-i1-GGUF:Qwen2.5-7B-Instruct.i1-Q4_K_M.gguf",
         "mradermacher/Qwen2.5-7B-i1-GGUF:Qwen2.5-7B.i1-Q4_K_M.gguf"),
        ("gguf", "mradermacher/Qwen2.5-7B-Instruct-i1-GGUF:Qwen2.5-7B-Instruct.i1-IQ2_M.gguf",
         "mradermacher/Qwen2.5-7B-i1-GGUF:Qwen2.5-7B.i1-IQ2_M.gguf"),
        ("gguf", "Qwen/Qwen2.5-7B-Instruct-GGUF:qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf", None),
        ("gguf", "bartowski/Qwen2.5-7B-Instruct-GGUF:Qwen2.5-7B-Instruct-Q8_0.gguf", None),
    ]),
    "Llama-3.1-8B-Instruct": dict(child="unsloth/Meta-Llama-3.1-8B-Instruct", base="unsloth/Meta-Llama-3.1-8B", quants=[
        ("st", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"),
        ("st", "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"),
        ("st", "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8"), ("st", "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16"),
        ("st", "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"), ("st", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"),
        ("mlx", "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", "mlx-community/Meta-Llama-3.1-8B-4bit"),
        ("mlx", "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit", "mlx-community/Meta-Llama-3.1-8B-8bit"),
        ("gguf", "mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF:Meta-Llama-3.1-8B-Instruct.Q8_0.gguf",
         "mradermacher/Meta-Llama-3.1-8B-GGUF:Meta-Llama-3.1-8B.Q8_0.gguf"),
        ("gguf", "QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF:Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf",
         "QuantFactory/Meta-Llama-3.1-8B-GGUF:Meta-Llama-3.1-8B.Q4_K_M.gguf"),
    ]),
    "Qwen3-8B": dict(child="Qwen/Qwen3-8B", base="Qwen/Qwen3-8B-Base", quants=[
        ("st", "Qwen/Qwen3-8B-FP8"), ("st", "Qwen/Qwen3-8B-AWQ"), ("st", "nvidia/Qwen3-8B-NVFP4"),
        ("mlx", "mlx-community/Qwen3-8B-4bit", None),
    ]),
    "Llama-3.3-70B-Instruct": dict(child="unsloth/Llama-3.3-70B-Instruct", base="unsloth/Meta-Llama-3.1-70B", quants=[
        ("st", "casperhansen/llama-3.3-70b-instruct-awq"), ("st", "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"),
        ("st", "nvidia/Llama-3.3-70B-Instruct-FP8"), ("st", "nvidia/Llama-3.3-70B-Instruct-NVFP4"),
        ("mlx", "mlx-community/Llama-3.3-70B-Instruct-4bit", "mlx-community/Meta-Llama-3.1-70B-4bit"),
    ]),
    "Qwen2.5-72B-Instruct": dict(child="Qwen/Qwen2.5-72B-Instruct", base="Qwen/Qwen2.5-72B", quants=[
        ("st", "Qwen/Qwen2.5-72B-Instruct-AWQ"), ("st", "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"),
    ]),
}
from frontier_families import FRONTIER  # noqa: E402
FAMILIES.update(FRONTIER)
KEYS = {"embed": "model.embed_tokens.weight", "head": "lm_head.weight"}
GKEYS = {"embed": "token_embd.weight", "head": "output.weight"}
TD = {"F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def key_for(repo, which):
    """Tensor name of the input embedding / output matrix in this repo (None if absent)."""
    suf = {"embed": ("embed_tokens.weight", "embed_in.weight", "tok_embeddings.weight", "wte.weight"),
           "head": ("lm_head.weight", "embed_out.weight", "output.weight")}[which]
    names = [k for k in remote.index(repo) if k.endswith(suf) and not any(
        t in k for t in ("vision", "visual", "audio", "mtp", "draft", ".layers."))]
    return min(names, key=len) if names else None


def bf16(repo, key):
    if repo is None:
        return None
    k = key_for(repo, "embed" if key == KEYS["embed"] else "head")
    return remote.tensor(repo, k) if k else None


def score(moved, truth):
    moved, truth = moved.bool(), truth.bool()
    return dict(true_moved=int(truth.sum()), true_unmoved=int((~truth).sum()), read_moved=int(moved.sum()),
                tp=int((moved & truth).sum()), fp=int((moved & ~truth).sum()), fn=int((~moved & truth).sum()),
                exact=bool((moved == truth).all()))


# ---- mlx affine quantization (two published variants) -------------------------------------
def mlx_quant(w: torch.Tensor, bits: int, gs: int, variant: str, dtype=torch.float16):
    """Return (packed uint32 [R, C*bits/32], scales, biases) as mlx would store them."""
    w = w.to(dtype).float().view(w.shape[0], -1, gs)
    n_bins = 2 ** bits - 1
    wmax, wmin = w.amax(-1, keepdim=True), w.amin(-1, keepdim=True)
    if variant == "minmax":
        scales = ((wmax - wmin) / n_bins).clamp_min(1e-7)
        biases = wmin
    else:  # 'edge': mlx >= 0.21 (zero exactly representable)
        mask = wmin.abs() > wmax.abs()
        scales = ((wmax - wmin) / n_bins).clamp_min(1e-7)
        scales = torch.where(mask, scales, -scales)
        edge = torch.where(mask, wmin, wmax)
        q0 = torch.round(edge / scales)
        scales = torch.where(q0 != 0, edge / q0, scales)
        biases = torch.where(q0 == 0, torch.zeros_like(edge), edge)
    q = torch.clamp(torch.round((w - biases) / scales), 0, n_bins).to(torch.int64).view(w.shape[0], -1)
    per = 32 // bits
    q = q.view(q.shape[0], -1, per)
    packed = torch.zeros(q.shape[:2], dtype=torch.int64)
    for i in range(per):
        packed |= q[..., i] << (bits * i)
    return packed.to(torch.int64), scales.squeeze(-1).to(dtype), biases.squeeze(-1).to(dtype)


def mlx_rows(repo, which):
    if repo is None:
        return None
    k = key_for(repo, which)
    if k is None:
        return None
    try:
        w = remote.tensor(repo, k)
        s = remote.tensor(repo, k.replace(".weight", ".scales"))
        b = remote.tensor(repo, k.replace(".weight", ".biases"))
    except KeyError:
        return None
    return w, s, b


def row_sig(parts):
    """bytes view per row -> int64 hash-free comparison helper: concat as uint8."""
    return torch.cat([p.contiguous().view(torch.uint8).view(p.shape[0], -1) for p in parts], 1)


# ---- GGUF Q8_0 -------------------------------------------------------------------------------
def q8_0(w: torch.Tensor, via_f16=True):
    x = w.to(torch.float16).float().numpy() if via_f16 else w.float().numpy()
    R, C = x.shape
    b = x.reshape(R, C // 32, 32)
    amax = np.abs(b).max(-1)
    d = (amax / 127).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, 0).astype(np.float32)
    q = np.round(b * idv[..., None])  # roundf: half away from zero
    q = np.where(np.abs(b * idv[..., None] - np.trunc(b * idv[..., None])) == 0.5,
                 np.trunc(b * idv[..., None]) + np.sign(b), q).astype(np.int8)
    out = np.empty((R, C // 32, 34), dtype=np.uint8)
    out[..., :2] = d.astype(np.float16).view(np.uint8).reshape(R, C // 32, 2)
    out[..., 2:] = q.view(np.uint8)
    return out.reshape(R, -1)


def gguf_pick(repo, qt):
    from huggingface_hub import HfApi
    fs = sorted(f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf"))
    cand = [f for f in fs if qt.lower() in f.lower() and ("-of-" not in f or "00001-of" in f)]
    if not cand:
        raise FileNotFoundError(f"no {qt} gguf in {repo}")
    return min(cand, key=len)


def gguf_open(spec):
    if "::" in spec:
        repo, qt = spec.split("::")
        return RemoteGGUF(repo, gguf_pick(repo, qt))
    repo, fn = spec.split(":")
    return RemoteGGUF(repo, fn)


def run_family(name, fam):
    res = []
    truth, ref = {}, {}
    import os
    WHICH = ("embed",) if os.environ.get("QM_EMBED_ONLY") else ("embed", "head")
    done = set()
    for f in OUT.parent.glob("quant_*.jsonl"):
        done |= {json.loads(l)["repo"] for l in open(f) if l.strip()}
    for w in WHICH:
        c, b = bf16(fam["child"], KEYS[w]), bf16(fam["base"], KEYS[w])
        ref[w] = (c, b)
        truth[w] = None if c is None or b is None or c.shape != b.shape else (c != b).any(1)
        print(name, w, "true moved", None if truth[w] is None else int(truth[w].sum()), flush=True)
    for q in fam["quants"]:
        kind, repo = q[0], q[1]
        if repo in done:
            continue
        rec = dict(family=name, kind=kind, repo=repo, base_quant=(q[2] if len(q) > 2 else None), tensors={})
        try:
            for w in WHICH:
                c16, b16 = ref[w]
                if c16 is not None and c16.dtype != torch.bfloat16 and c16.dtype != torch.float16:
                    c16 = c16.to(torch.bfloat16)
                if c16 is None:
                    continue
                t = {}
                if kind == "st":
                    kk = key_for(repo, w)
                    try:
                        dt, sh = remote.tensor_info(repo, kk)
                    except (KeyError, TypeError):
                        t["stored"] = "absent (tied or renamed)"
                        rec["tensors"][w] = t
                        continue
                    t["stored"] = dt
                    if dt in TD:
                        x = remote.tensor(repo, kk)
                        if x.shape != c16.shape:
                            t["note"] = f"shape {list(x.shape)} vs {list(c16.shape)}"
                            n = min(x.shape[0], c16.shape[0])
                            x, cc, bb = x[:n], c16[:n], b16[:n]
                        else:
                            cc, bb = c16, b16
                        cast = cc.to(x.dtype)
                        t["rows_equal_release"] = int((x == cast).all(1).sum())
                        t["rows"] = int(x.shape[0])
                        t["coords_equal_release_frac"] = float((x == cast).float().mean())
                        t["equals_release_cast"] = bool((x == cast).all())
                        if truth[w] is not None:
                            t["read_vs_bf16_base"] = score((x != bb.to(x.dtype)).any(1), truth[w][:x.shape[0]])
                elif kind == "mlx":
                    m = mlx_rows(repo, w)
                    if m is None:
                        t["stored"] = "absent"
                        rec["tensors"][w] = t
                        continue
                    wq, s, bq = m
                    bits = 32 * wq.shape[1] // c16.shape[1]
                    gs = c16.shape[1] // s.shape[1]
                    t["stored"] = f"mlx affine {bits}-bit g{gs}"
                    sig = row_sig([wq, s, bq])
                    if q[2]:
                        mb = mlx_rows(q[2], w)
                        if mb is not None and mb[0].shape == wq.shape:
                            if truth[w] is not None:
                                t["read_vs_same_tool_base"] = score((sig != row_sig(list(mb))).any(1), truth[w])
                    # can we reproduce the quantizer from the bf16 child?
                    best = None
                    for variant in ("minmax", "edge"):
                        for dt in (torch.float16, torch.bfloat16):
                            pk, ss, bb2 = mlx_quant(c16, bits, gs, variant, dt)
                            eq = (pk == wq.to(torch.int64) & 0xFFFFFFFF).all(1) & (ss.view(torch.int16) == s.view(torch.int16)).all(1) \
                                & (bb2.view(torch.int16) == bq.view(torch.int16)).all(1)
                            frac = float(eq.float().mean())
                            if best is None or frac > best[0]:
                                best = (frac, variant, str(dt))
                    t["requant_from_bf16_rows_exact"] = dict(frac=best[0], variant=best[1], compute_dtype=best[2])
                    if best[0] > 0.999 and b16 is not None:
                        pkb, sb, bbb = mlx_quant(b16, bits, gs, best[1], getattr(torch, best[2].split(".")[-1]))
                        # compare by re-quantized values directly
                        moved = ~((pkb == wq.to(torch.int64) & 0xFFFFFFFF).all(1) & (sb.view(torch.int16) == s.view(torch.int16)).all(1)
                                  & (bbb.view(torch.int16) == bq.view(torch.int16)).all(1))
                        t["read_vs_requantized_bf16_base"] = score(moved, truth[w])
                elif kind == "gguf":
                    g = gguf_open(repo)
                    if GKEYS[w] not in g.tensors:
                        t["stored"] = "not in this file (tied, or in another shard)"
                        rec["tensors"][w] = t
                        continue
                    typ, dims = g.info(GKEYS[w])
                    t["stored"] = f"gguf {typ}"
                    if typ in ("F16", "BF16", "F32"):
                        rec["tensors"][w] = t
                        continue
                    if not ((q[2] and truth[w] is not None) or typ == "Q8_0"):
                        rec["tensors"][w] = t
                        continue
                    x = g.raw_rows(GKEYS[w])
                    if q[2] and truth[w] is not None:
                        gb = gguf_open(q[2])
                        if GKEYS[w] in gb.tensors and gb.info(GKEYS[w]) == (typ, dims):
                            xb = gb.raw_rows(GKEYS[w])
                            t["read_vs_same_tool_base"] = score(torch.from_numpy((x != xb).any(1)), truth[w])
                        elif GKEYS[w] in gb.tensors:
                            t["base_quant_type"] = gb.info(GKEYS[w])[0]
                    if typ == "Q8_0":
                        best = None
                        for via in (True, False):
                            mine = q8_0(c16, via)
                            frac = float((mine == x).all(1).mean())
                            if best is None or frac > best[0]:
                                best = (frac, via)
                        t["requant_from_bf16_rows_exact"] = dict(frac=best[0], via_f16=best[1])
                        if best[0] > 0.99 and b16 is not None:
                            mb = q8_0(b16, best[1])
                            t["read_vs_requantized_bf16_base"] = score(torch.from_numpy((x != mb).any(1)), truth[w])
                rec["tensors"][w] = t
        except Exception as e:  # keep going; record why
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(json.dumps(rec), flush=True)
        res.append(rec)
        import os
        with open(OUT.parent / ("quant_" + os.environ.get("QM_SHARD", "main") + ".jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return res


def main():
    only = sys.argv[1:]
    old = json.load(open(OUT)) if OUT.exists() else []
    out = [r for r in old if only and r["family"] not in only]
    for name, fam in FAMILIES.items():
        if only and name not in only:
            continue
        try:
            out += run_family(name, fam)
        except Exception as e:  # a family whose reference cannot be loaded
            print(name, "FAMILY FAILED", type(e).__name__, str(e)[:200], flush=True)
            import os
            with open(OUT.parent / ("quant_" + os.environ.get("QM_SHARD", "main") + ".jsonl"), "a") as fh:
                fh.write(json.dumps(dict(family=name, kind="-", repo=fam["child"], tensors={},
                                         error=f"reference not loaded: {type(e).__name__}: {str(e)[:150]}")) + "\n")


if __name__ == "__main__":
    main()
