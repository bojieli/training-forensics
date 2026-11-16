#!/usr/bin/env python3
"""Blind reading of a training-stack run (scripts/stacks/finetune_stack.py), scored vs truth.

Runs the unchanged reader (scripts/controlled/score.py) with two stack-aware steps
that use only public information:

  start point   the optimizer may start from the base cast to the training dtype
                (DeepSpeed bf16 builds its fp32 master from bf16 params). The reader
                tries {W0 as published, bf16(W0)} and keeps the one the never-moved
                rows agree with.
  decay form    the bit pattern of decoupled weight decay depends on the kernel:
                  mul      torch AdamW / Muon        p * (1 - lr*wd)
                  fused    DeepSpeed FusedAdam (CUDA) fma(-lr, wd*p, p)
                  cpuadam  DeepSpeed CPUAdam (AVX)    fma(p, -lr*wd, p)
                The reader replays all three on never-moved rows and keeps the one
                that is bit-exact; tokens and replay are then re-read with it.

  python3 scripts/stacks/score_stack.py runs/stacks/t01 [--release fp32|bf16]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "controlled"))
import score as S  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FORMS = ("mul", "fused", "cpuadam")


def decay_step(x: torch.Tensor, lr: float, wd: float, form: str) -> torch.Tensor:
    if form == "mul":
        return x * float(np.float32(1 - lr * wd))
    lr32, wd32 = np.float32(lr), np.float32(wd)
    xd = x.double()
    if form == "fused":
        t = (xd * float(wd32)).float().double()            # update = wd*p (rounded)
        return (xd - float(lr32) * t).float()              # fma(-lr, update, p)
    if form == "cpuadam":
        w = float(np.float32(-lr32 * wd32))                # w_decay = -lr*wd in fp32
        return (xd + w * xd).float()                       # fma(p, w_decay, p)
    raise ValueError(form)


def replay_rows(x0, rec, form, N=None):
    lr = S.hf_lr(dict(rec, peak=rec["peak_snap"]), N or rec["N"])
    x = x0.clone().float().to(DEV)
    wd = rec["wd_snap"]
    for t in range(1, (N or rec["N"]) + 1):
        x = decay_step(x, float(lr[t]), wd, form)
    return x


def refine_by_replay(out, n_rows=2000):
    """Warm-up and step count are read from noisy single-step moves; with weight decay
    on, the never-updated vectors pin them exactly. Try warm-up +-3 and steps +-1 under
    each decay kernel on a sample of never-updated vectors; keep the bit-exact best."""
    rec = out["recipe"]
    if not (rec.get("wd_snap") or 0) > 0:
        return None
    W0s, W1f = out["_W"]
    idx = np.nonzero(~out["_rd"].moved)[0]
    if len(idx) == 0:
        return None
    idx = idx[np.linspace(0, len(idx) - 1, min(n_rows, len(idx))).astype(int)]
    x0, y = W0s[idx], W1f[idx].to(DEV)
    best = None
    for w in range(max(0, rec["warmup"] - 3), rec["warmup"] + 4):
        for N in (rec["N"] - 1, rec["N"], rec["N"] + 1):
            r = dict(rec, warmup=w, N=N)
            for f in FORMS:
                k = int((replay_rows(x0, r, f) == y).sum())
                if best is None or k > best[0]:
                    best = (k, w, N, f)
    k, w, N, f = best
    changed = (w, N) != (rec["warmup"], rec["N"])
    info = dict(before=dict(warmup=rec["warmup"], N=rec["N"]), after=dict(warmup=w, N=N), form=f,
                bit_exact_sample=k, sample_weights=int(y.numel()), changed=changed)
    if changed:
        rec["warmup"], rec["N"] = w, N
    return info


def pick_start(W0, W1, n=4000):
    """Start point the never-moved rows agree with (public base as published or bf16-cast)."""
    idx = torch.linspace(0, W0.shape[0] - 1, n).long()
    best, best_s = "published", -1
    for name, c in (("published", W0), ("bf16-cast", W0.bfloat16().float())):
        a, b = c[idx].double(), W1[idx].double()
        r = b / a.where(a != 0, torch.ones_like(a))
        ok = (a != 0)
        med = r.where(ok, torch.nan).nanmedian(dim=1).values
        dev = ((r - med[:, None]).abs().where(ok, torch.zeros_like(r))).amax(dim=1)
        s = int((dev < 1e-5).sum())   # rows that are a uniform rescale of the start up to fp32 rounding (a bf16 cast error is ~4e-3)
        if s > best_s:
            best, best_s = name, s
    return best, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--release", default="fp32", choices=["fp32", "bf16", "param"])
    ap.add_argument("--no-docs", action="store_true")
    a = ap.parse_args()
    orig_load = S.load_file
    for r in a.runs:
        run = pathlib.Path(r)
        e = orig_load(str(run / "emb.safetensors"))
        rel = a.release
        W1 = e["W1"] if rel == "fp32" else (e["W1_param"].float() if rel == "param" and "W1_param" in e else e["W1_bf16"].float())
        start, n_ok = pick_start(e["W0"], W1)
        e2 = dict(e)
        if start == "bf16-cast":
            e2["W0"] = e["W0"].bfloat16().float()
        if rel == "param" and "W1_param" in e:
            e2["W1_bf16"] = e["W1_param"]
        S.load_file = lambda p, _e=e2: _e
        out = S.read(run, "fp32" if rel == "fp32" else "bf16")
        refine = refine_by_replay(out) if rel == "fp32" else None
        res = S.score(run, out, with_docs=not a.no_docs)
        res["start_point"] = dict(chosen=start, rows_consistent=n_ok)
        if refine:
            res["replay_refine"] = refine
        tr = json.load(open(run / "truth.json"))
        res["stack"] = tr.get("stack")
        gn = [g for g in tr.get("grad_norms", []) if g is not None]
        res["grad_norm_logged"] = dict(n=len(gn), min=min(gn) if gn else None, max=max(gn) if gn else None,
                                       clip=tr["args"].get("max_grad_norm"))
        if "W1_param" in e:
            res["param_equals_master_bf16"] = bool((e["W1_param"] == e["W1"].bfloat16()).all())
        rec = out["recipe"]
        if rel == "fp32" and (rec.get("wd_snap") or 0) > 0:
            W0s, W1f = out["_W"]
            rd = out["_rd"]
            idx = np.nonzero(~rd.moved)[0]
            y = W1f[idx].to(DEV)
            forms = {}
            for f in FORMS:
                x = replay_rows(W0s[idx], rec, f)
                forms[f] = int((x == y).sum())
            best = max(forms, key=forms.get)
            res["decay_forms"] = dict(bit_exact=forms, weights=int(y.numel()), best=best)
            if best != "mul":
                # re-read tokens and replay with the kernel the bits point to
                seen = None
                from trainforensics.truth import Run
                seen = Run(run).seen_mask()
                moved = np.zeros(W0s.shape[0], bool)
                for c in range(0, W0s.shape[0], 16384):
                    x = replay_rows(W0s[c:c + 16384], rec, best).cpu()
                    moved[c:c + 16384] = (x != W1f[c:c + 16384]).any(1).numpy()
                res["tokens_replay"] = dict(moved=int(moved.sum()), tp=int((seen & moved).sum()),
                                            fp=int((~seen & moved).sum()), fn=int((seen & ~moved).sum()), form=best)
                for k, sh in (("replay", 0), ("replay_shift1", 1), ("replay_shiftm1", -1)):
                    x = replay_rows(W0s[idx], rec, best, N=rec["N"] + sh)
                    res[k] = {"weights": int(y.numel()), "bit_exact": int((x == y).sum()), "form": best}
        S.load_file = orig_load
        dst = ROOT / os.environ.get("TF_OUTDIR", "results/stacks") / f"{run.name}.{rel}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(dst, "w"), indent=1, default=float)
        brief = {k: res.get(k) for k in ("tokens", "tokens_replay", "recipe_ok", "seed", "replay", "decay_forms",
                                         "start_point", "placement")}
        print(run.name, rel, json.dumps(brief, default=float))


if __name__ == "__main__":
    main()
