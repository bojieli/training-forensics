#!/usr/bin/env python3
"""Membership without memorisation: does a document's own gradient explain
the sign pattern of a single-touch embedding row?

For every candidate document (claimed-and-trained + outside), run the PUBLIC
starting model on that document alone, take the input-embedding gradient
G_t for each token t it contains, and compare sign(-G_t) with sign(W1_t - W0_t)
on the rows that moved exactly once (flat rows). A document's score is the
best agreement over its tokens. Threshold = the largest outside score
(zero false alarms).

Baseline: reference-model loss attack, score = loss_base(d) - loss_ft(d)
(needs the full fine-tuned model, finetune.py --save-model), and plain loss.

  python3 scripts/controlled/membership.py runs/ctl/m01
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics.diff import row_diff  # noqa: E402


def main(run):
    run = pathlib.Path(run)
    t = json.load(open(run / "truth.json"))
    toks = {int(k): v for k, v in json.load(open(run / "tokens.json")).items()}
    e = load_file(str(run / "emb.safetensors"))
    W0, W1 = e["W0"], e["W1"]
    rd = row_diff(W0, W1, torch.float32)
    flat = rd.moved & (rd.flat_frac > 0.9)
    sgn = torch.sign(W1.double() - rd.decay * W0.double())
    dev = "cuda"
    base = AutoModelForCausalLM.from_pretrained(t["args"]["model"], torch_dtype=torch.float32).to(dev)
    base.eval()
    emb = base.get_input_embeddings()
    ft = None
    if (run / "model").exists():
        ft = AutoModelForCausalLM.from_pretrained(str(run / "model"), torch_dtype=torch.float32).to(dev).eval()

    members = [d for d in t["trained"]]
    outside = t["outside"]
    rows = []
    for label, pool in ((1, members), (0, outside)):
        for d in pool:
            ids = toks[d]
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], device=dev)
            base.zero_grad(set_to_none=True)
            out = base(input_ids=x, labels=x)
            out.loss.backward()
            G = emb.weight.grad
            best, n_elig = -1.0, 0
            for tk in set(ids[:-1]):
                if not flat[tk]:
                    continue
                n_elig += 1
                g = G[tk]
                nz = g != 0
                if nz.sum() == 0:
                    continue
                agree = (torch.sign(-g[nz]).double().cpu() == sgn[tk][nz.cpu()]).double().mean().item()
                best = max(best, agree)
            r = dict(doc=d, member=label, score=best, n_elig=n_elig, loss_base=out.loss.item())
            if ft is not None:
                with torch.no_grad():
                    r["loss_ft"] = ft(input_ids=x, labels=x).loss.item()
            rows.append(r)
    json.dump(rows, open(run / "membership.json", "w"))

    def tpr_at_zero_fp(score_key, sign=1.0):
        s_out = [sign * r[score_key] for r in rows if r["member"] == 0]
        s_in = [sign * r[score_key] for r in rows if r["member"] == 1]
        thr = max(s_out)
        return sum(1 for v in s_in if v > thr) / len(s_in), thr

    res = {"members": sum(r["member"] for r in rows), "outside": sum(1 - r["member"] for r in rows)}
    res["grad_sign"] = dict(zip(("tpr_at_0fp", "threshold"), tpr_at_zero_fp("score")))
    res["members_with_eligible_token"] = sum(1 for r in rows if r["member"] and r["n_elig"] > 0)
    res["outside_score_quantiles"] = np.quantile([r["score"] for r in rows if r["member"] == 0],
                                                 [0.5, 0.9, 0.99, 1.0]).tolist()
    res["member_score_quantiles"] = np.quantile([r["score"] for r in rows if r["member"] == 1],
                                                [0.1, 0.5, 0.9]).tolist()
    res["loss_attack"] = dict(zip(("tpr_at_0fp", "threshold"), tpr_at_zero_fp("loss_base", -1.0)))
    if ft is not None:
        for r in rows:
            r["ref"] = r["loss_base"] - r["loss_ft"]
            r["neg_loss_ft"] = -r["loss_ft"]
        res["ref_model_attack"] = dict(zip(("tpr_at_0fp", "threshold"), tpr_at_zero_fp("ref")))
        res["ft_loss_attack"] = dict(zip(("tpr_at_0fp", "threshold"), tpr_at_zero_fp("neg_loss_ft")))
    dst = ROOT / "results" / "controlled" / f"{run.name}.membership.json"
    json.dump(res, open(dst, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
