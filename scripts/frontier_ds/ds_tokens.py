#!/usr/bin/env python3
"""DeepSeek special tokens and reserved placeholder slots, release by release.

Part A: anatomy of the base model (DeepSeek-V3-Base): which token vectors are still
        at their regenerated random start (checkpoint-forensics card), the size of
        the reserved placeholder vectors, and whether they are the start plus an
        update or something else.
Part B: for every DeepSeek release vs its base, the status of each special token
        and placeholder slot: bit-identical / barely changed / updated, plus tests
        that separate a training update from an overwrite (zeroed, copied, rescaled,
        replaced by an unrelated vector).

All DeepSeek post-training comparisons in the sweep fit no weight decay on the
embedding (P = 1), so "changed" here means "differs from the released base".

Writes results/frontier_ds/deepseek.json.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import glob
import json
import os
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = pathlib.Path(os.environ.get("CKPTF_CACHE", paths.cache("trainforensics") / "emb"))
OUT = ROOT / "results" / "frontier_ds"
TOK = OUT / "tok"
V, D = 129280, 7168

PAIRS = [
    ("deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-V3"),
    ("deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-V3-0324"),
    ("deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-R1"),
    ("deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-R1-Zero"),
    ("deepseek-ai/DeepSeek-V3-Base", "nex-agi/DeepSeek-V3.1-Nex-N1"),
    ("deepseek-ai/DeepSeek-V3.2-Exp-Base", "deepseek-ai/DeepSeek-V3.2"),
    ("deepseek-ai/DeepSeek-V4-Flash-Base", "deepseek-ai/DeepSeek-V4-Flash"),
    ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-0731"),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", "deepseek-ai/DeepSeek-V4-Pro"),
    ("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro-0813"),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", "deepseek-ai/DeepSeek-V4-Pro-0813"),
    ("deepseek-ai/DeepSeek-V4-Pro-Base", "nex-agi/Nex-N2.5-Max"),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek-ai/DeepSeek-V4.1-Flash-1017"),
]

BARELY = 0.10   # under 10% of a vector's numbers changed = "barely changed" (detection floor)


def emb_path(repo):
    pref = repo.replace("/", "__") + "@"
    fs = [f for f in glob.glob(str(CACHE / (pref + "*")))
          if ("embed" in f.split("__")[-2] or "embed" in f) and "head" not in f.split("__")[-2]]
    fs = [f for f in fs if os.path.getsize(f) % (V * 2) == 0]
    if not fs:
        raise FileNotFoundError(repo)
    return fs[0]


def load(repo):
    f = emb_path(repo)
    n = os.path.getsize(f) // 2
    d = n // V
    t = torch.frombuffer(bytearray(open(f, "rb").read()), dtype=torch.bfloat16).view(V, d)
    return t, f


def tokens(tokfile):
    t = json.load(open(tokfile))
    return {a["id"]: a["content"] for a in t["added_tokens"]}


def groups(added):
    named = sorted(i for i, s in added.items() if "place" not in s.lower() and i >= 128000)
    named += [i for i in (0, 1, 2)]
    ph = sorted(i for i, s in added.items() if "place" in s.lower())
    pad = [i for i in range(V) if i >= 128000 and i not in added]
    return dict(named=sorted(named), placeholder=ph, unused=pad, ordinary=list(range(3, 128000)))


def anatomy_v3base():
    """Part A."""
    sys.path.insert(0, str(paths.CKPTF_REPO))
    from ckptforensics import emu
    card = json.load(open(paths.CARDS / "DeepSeek-V3-Base.json"))
    W, f = load("deepseek-ai/DeepSeek-V3-Base")
    init = emu.normal((V, D), 0.0, card["std"], seed=card["seed"], offset=card["offset"],
                      device_threads=card["threads"]).cpu()
    initb = init.to(torch.bfloat16)
    eq = (initb == W).all(1)
    Wf = W.float()
    norm = Wf.norm(dim=1)
    cos_init = torch.nn.functional.cosine_similarity(Wf, init, dim=1)
    g = groups(tokens(TOK / "DeepSeek-V3-Base__tokenizer.json"))
    mean_vec = Wf[3:128000].mean(0)
    cos_mean = torch.nn.functional.cosine_similarity(Wf, mean_vec[None], dim=1)
    # update = W - init (what pretraining added to each vector)
    upd = Wf - init
    upd_norm = upd.norm(dim=1)
    res = dict(source=f, card=dict(seed=card["seed"], std=card["std"], threads=card["threads"],
                                   row_ranges=card["row_ranges"]),
               expected_norm_at_start=float(card["std"] * D ** 0.5))
    for k, ids in g.items():
        ids_t = torch.tensor(ids)
        q = lambda x: [float(v) for v in np.percentile(x[ids_t].numpy(), [5, 50, 95])]
        res[k] = dict(n=len(ids), at_start_bit_exact=int(eq[ids_t].sum()),
                      norm_p5_50_95=q(norm), cos_with_start_p5_50_95=q(cos_init),
                      update_norm_p5_50_95=q(upd_norm), cos_with_mean_ordinary_p5_50_95=q(cos_mean))
    # named tokens one by one
    added = tokens(TOK / "DeepSeek-V3-Base__tokenizer.json")
    res["named_detail"] = [dict(id=i, token=added.get(i), at_start=bool(eq[i]), norm=float(norm[i]),
                                cos_with_start=float(cos_init[i])) for i in g["named"]]
    # placeholders that are still at start
    ph = torch.tensor(g["placeholder"])
    res["placeholder_ids_at_start"] = [int(i) for i in ph[eq[ph]]]
    # do the placeholder updates share a direction with each other (trained together) ?
    U = upd[ph]
    Un = U / U.norm(dim=1, keepdim=True)
    C = Un @ Un.T
    off = C[~torch.eye(len(ph), dtype=bool)]
    res["placeholder_update_pairwise_cos_p5_50_95"] = [float(v) for v in np.percentile(off.numpy(), [5, 50, 95])]
    Uo = upd[torch.randint(3, 128000, (800,), generator=torch.Generator().manual_seed(0))]
    Uon = Uo / Uo.norm(dim=1, keepdim=True)
    Co = Uon @ Uon.T
    offo = Co[~torch.eye(800, dtype=bool)]
    res["ordinary_update_pairwise_cos_p5_50_95"] = [float(v) for v in np.percentile(offo.numpy(), [5, 50, 95])]
    del init, initb
    return res


def row_hash(T):
    b = T.contiguous().view(torch.int16).numpy()
    return {bytes(r.tobytes()): i for i, r in enumerate(b)}


def compare(parent, child, tokfile):
    A, fa = load(parent)
    B, fb = load(child)
    added = tokens(tokfile)
    g = groups(added)
    ne = (A != B)
    frac = ne.float().mean(1)
    Af, Bf = A.float(), B.float()
    dlt = Bf - Af
    cosAB = torch.nn.functional.cosine_similarity(Af, Bf, dim=1)
    nA, nB, nd = Af.norm(dim=1), Bf.norm(dim=1), dlt.norm(dim=1)
    ordinary = torch.arange(3, 128000)
    upd_ord = ordinary[frac[ordinary] >= BARELY]
    drift = dlt[upd_ord].mean(0)
    cos_drift = torch.nn.functional.cosine_similarity(dlt, drift[None], dim=1)
    # scaled-copy residual: || B - sA || / ||B||, s = <A,B>/<A,A>
    s = (Af * Bf).sum(1) / (Af * Af).sum(1).clamp_min(1e-30)
    resid = (Bf - s[:, None] * Af).norm(dim=1) / nB.clamp_min(1e-30)
    meanB = Bf[ordinary].mean(0)
    cos_meanB = torch.nn.functional.cosine_similarity(Bf, meanB[None], dim=1)
    # reference distribution: ordinary tokens updated in this comparison
    def pct(x, ids):
        return [float(v) for v in np.percentile(x[ids].numpy(), [5, 50, 95])] if len(ids) else None
    ref = dict(n_updated_ordinary=int(len(upd_ord)),
               cos_base_child=pct(cosAB, upd_ord), change_norm=pct(nd, upd_ord),
               cos_change_with_common_drift=pct(cos_drift, upd_ord), norm_child=pct(nB, upd_ord))
    hA = row_hash(A)
    hB = row_hash(B)
    Bi16 = B.contiguous().view(torch.int16).numpy()

    def status(i):
        f = float(frac[i])
        if f == 0:
            return "bit-identical"
        return "barely changed" if f < BARELY else "updated"

    def detail(i):
        key = bytes(Bi16[i].tobytes())
        copy_of_B = hB.get(key)
        copy_of_A = hA.get(key)
        d = dict(id=i, token=added.get(i, "(unused slot)"), status=status(i), frac_changed=round(float(frac[i]), 4),
                 norm_base=round(float(nA[i]), 4), norm_release=round(float(nB[i]), 4),
                 change_norm=round(float(nd[i]), 4), cos_base_release=round(float(cosAB[i]), 4),
                 cos_change_with_common_drift=round(float(cos_drift[i]), 4),
                 scaled_copy_residual=round(float(resid[i]), 4), cos_with_mean_vector=round(float(cos_meanB[i]), 4))
        flags = []
        if d["status"] != "bit-identical":
            if bool((B[i] == 0).all()):
                flags.append("zeroed")
            if copy_of_B is not None and copy_of_B != i:
                flags.append(f"exact copy of release token {copy_of_B}")
            if copy_of_A is not None and copy_of_A != i:
                flags.append(f"exact copy of base token {copy_of_A}")
            if resid[i] < 0.01 and abs(float(s[i]) - 1) > 1e-3:
                flags.append(f"rescaled copy of its base vector (x{float(s[i]):.4f})")
            if cos_meanB[i] > 0.99:
                flags.append("equals the mean vector")
            if d["status"] == "updated" and cosAB[i] < 0.05 and nA[i] > 0:
                flags.append("unrelated to its base vector (cos < 0.05): replaced, not updated")
        d["overwrite_flags"] = flags
        return d

    out = dict(parent=parent, child=child, files=[fa, fb], dim=int(A.shape[1]), reference_ordinary=ref)
    for k in ("named", "placeholder", "unused"):
        ids = torch.tensor(g[k])
        st = [status(int(i)) for i in ids]
        out[k] = dict(n=len(ids), counts={c: st.count(c) for c in ("bit-identical", "barely changed", "updated")})
    out["named_detail"] = [detail(i) for i in g["named"]]
    ph_changed = [i for i in g["placeholder"] if frac[i] > 0]
    out["placeholder_changed"] = [detail(i) for i in ph_changed]
    out["unused_changed"] = [detail(i) for i in g["unused"] if frac[i] > 0]
    # histogram of changed fraction for ordinary tokens (detection floor)
    h, edges = np.histogram(frac[ordinary].numpy(), bins=[0, 1e-9, 0.01, 0.1, 0.5, 1.0001])
    out["ordinary_frac_changed_hist"] = dict(edges=["0", "(0,1%)", "[1%,10%)", "[10%,50%)", "[50%,100%]"],
                                             counts=[int(x) for x in h])
    out["_B_named"] = {i: B[i].clone() for i in g["named"]}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = dict(barely_threshold=BARELY)
    res["anatomy_v3base"] = anatomy_v3base()
    print(json.dumps({k: v for k, v in res["anatomy_v3base"].items() if k != "named_detail"}, indent=1)[:4000], flush=True)
    comps = []
    keep = {}
    for p, c in PAIRS:
        tokfile = TOK / ("DeepSeek-V3-Base__tokenizer.json" if ("V3" in p) else
                         "DeepSeek-V4.1-Flash__tokenizer.json" if "V4.1" in p else "DeepSeek-V4-Pro-Base__tokenizer.json")
        try:
            r = compare(p, c, tokfile)
        except FileNotFoundError as e:
            comps.append(dict(parent=p, child=c, error=f"not cached: {e}"))
            continue
        keep[c] = (r.pop("_B_named"), r["dim"])
        comps.append(r)
        print(p, "->", c, {k: r[k]["counts"] for k in ("named", "placeholder", "unused")}, flush=True)
    # identical named vectors across releases of the same line
    cross = []
    for a, b in (("deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Pro-0813"),
                 ("deepseek-ai/DeepSeek-V4-Flash", "deepseek-ai/DeepSeek-V4-Flash-0731"),
                 ("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324"),
                 ("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1")):
        if a in keep and b in keep:
            same = [i for i in keep[a][0] if bool((keep[a][0][i] == keep[b][0][i]).all())]
            cross.append(dict(a=a, b=b, named_identical_ids=same))
    res["comparisons"] = comps
    res["named_identical_across_releases"] = cross
    json.dump(res, open(OUT / "deepseek.json", "w"), indent=1)


if __name__ == "__main__":
    main()
