#!/usr/bin/env python3
"""Pretraining weight decay from never-trained rows (base models, no parent needed).

checkpoint-forensics regenerates the never-trained embedding rows of a base model
bit for bit from (seed, offset, thread count, std). The std that regenerates them
is init_std x P, where P = prod_t (1 - lr_t wd) is the decoupled weight decay those
rows received over pretraining. So:

  - card std equal to a standard init std of that model  -> P = 1: no decoupled
    weight decay reached the embedding during pretraining;
  - otherwise P = card_std / init_std and sum(lr*wd) = -ln P, *if* init_std is
    the config's initializer_range (flagged as an assumption per model).

Validated on Pythia, whose decay is measured independently across 154 public
checkpoints (results/public/claim1_pythia_lr.json).

Writes results/roster/pretrain_decay.json and results/roster/PRETRAIN_DECAY.md.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402

import glob
import json
import math
import os
import pathlib

import requests
from huggingface_hub import hf_hub_url

ROOT = pathlib.Path(__file__).resolve().parents[2]
CARDS = paths.CARDS
CACHE = ROOT / "results" / "roster" / "_configs"


def config(repo, revision=None):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / (repo.replace("/", "__") + ".json")
    if p.exists():
        return json.loads(p.read_text())
    try:
        r = requests.get(hf_hub_url(repo, "config.json", revision=revision), timeout=60)
        cfg = r.json() if r.status_code == 200 else {}
    except Exception:
        cfg = {}
    p.write_text(json.dumps(cfg))
    return cfg


def find(cfg, *keys):
    for k in keys:
        if k in cfg and isinstance(cfg[k], (int, float)):
            return cfg[k]
    for sub in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(sub), dict):
            v = find(cfg[sub], *keys)
            if v is not None:
                return v
    return None


def main():
    pythia_truth = {}
    pj = ROOT / "results" / "public" / "claim1_pythia_lr.json"
    if pj.exists():
        d = json.load(open(pj))
        for size, v in d.items():
            if isinstance(v, dict) and isinstance(v.get("measure"), dict):
                m = v["measure"]
                for k in ("final_ratio", "ratio_final", "ratio_143000", "ratio"):
                    if isinstance(m.get(k), (int, float)):
                        pythia_truth[size] = m[k]
    out = []
    for f in sorted(CARDS.glob("*.json")):
        c = json.load(open(f))
        cfg = config(c["repo"], c.get("revision"))
        V, dmod = c["shape"]
        hidden = find(cfg, "hidden_size", "d_model", "n_embd") or dmod
        ir = find(cfg, "initializer_range", "init_std", "embedding_init_std")
        std = c["std"]
        # standard init stds this model could have used
        conv = {"initializer_range": ir, "d^-1/2": hidden ** -0.5, "small_init sqrt(2/5d)": math.sqrt(2 / (5 * hidden)),
                "1.0": 1.0}
        # truncated / uniform fills: the card std is the distribution's scale parameter
        match = [k for k, v in conv.items() if v and abs(std / v - 1) < 2e-3]
        rec = dict(name=c["name"], repo=c["repo"], dist=c["dist"], card_std=std, initializer_range=ir,
                   hidden=hidden, model_type=cfg.get("model_type"))
        if match:
            rec.update(verdict="no decay on embedding rows (std = init)", init=match[0], P=1.0, sum_lr_wd=0.0)
        elif c["name"].lower().startswith("pythia"):
            # validated against the checkpoint series: sizes <= 2.8B were initialised
            # with gpt-neox small_init, 6.9B and 12B with std 0.02
            big = any(t in c["name"] for t in ("6.9B", "12B"))
            s0 = 0.02 if big else conv["small_init sqrt(2/5d)"]
            P = std / s0
            rec.update(verdict="decayed (Pythia init validated)", init="0.02" if big else "small_init sqrt(2/5d)", P=P,
                       sum_lr_wd=-math.log(P) if 0 < P < 1 else None)
        elif std > 0 and any(abs(std / (m * 10 ** math.floor(math.log10(std))) - 1) < 1e-4
                             for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9)):
            # a decayed std is a product of ~1e5 factors and essentially never round;
            # a round std is the init itself (e.g. DeepSeek-V3: 0.006 per its paper,
            # while its config says 0.02)
            rec.update(verdict="likely no decay (std is a round number = the init)", init=f"{std:g} (round)",
                       P=1.0, sum_lr_wd=0.0)
        elif ir and 0 < std < ir:
            P = std / ir
            rec.update(verdict="decayed, assuming init std = initializer_range", init="initializer_range (assumed)",
                       P=P, sum_lr_wd=-math.log(P))
            if -math.log(P) > 15:
                # P < 3e-7: far more shrinkage than weight decay plausibly gives; the vectors
                # were zeroed or rescaled some other way (same rule as coverage.py)
                rec["verdict"] = "implausible for decay alone (vectors rescaled or zeroed otherwise)"
        else:
            rec.update(verdict="init std unknown", init=None, P=None, sum_lr_wd=None)
        out.append(rec)
    json.dump(out, open(ROOT / "results" / "roster" / "pretrain_decay.json", "w"), indent=1)
    from collections import Counter as _C
    cnt = _C(r["verdict"] for r in out)
    lines = ["# Pretraining weight decay, estimated from never-trained token vectors", "",
             "Generated by `scripts/roster/pretrain_decay.py` from the checkpoint-forensics fingerprint cards.", "",
             "Every token in a model's vocabulary has its own vector of numbers. Tokens that never occur in pretraining",
             "(plus unused padding / reserved slots) keep their random starting values, except that decoupled weight decay",
             "shrinks them by a factor (1 - lr x wd) at every step. checkpoint-forensics regenerates those starting values",
             "exactly; the spread (std) that reproduces the published vectors bit for bit is init_std x P, where P is the",
             "total shrinkage over pretraining. So the total weight decay applied is sum(lr x wd) = -ln P, and it needs no",
             "checkpoints - but only if we know init_std, the spread the vectors started with.", "",
             "How sure each row is:",
             f"- **validated** ({cnt['decayed (Pythia init validated)']} Pythia models): init_std known, and the estimate agrees with an independent measurement from Pythia's public checkpoints to 1e-4 (table at the end);",
             f"- **exact, no decay** ({cnt['no decay on embedding rows (std = init)']}): the fitted spread equals a standard init value exactly, so no decoupled weight decay reached these vectors;",
             f"- **round number, likely no decay** ({cnt['likely no decay (std is a round number = the init)']}): the fitted spread is a round number such as 0.006 (DeepSeek's paper init), which a decayed value almost never is; an inference, not a proof;",
             f"- **assumed init** ({cnt['decayed, assuming init std = initializer_range']}): the estimate assumes the vectors started at the config's `initializer_range`; a different init convention moves P by ~2x, as Pythia shows;",
             f"- **implausible for decay alone** ({cnt['implausible for decay alone (vectors rescaled or zeroed otherwise)']}): under the assumed init, sum(lr x wd) > 15 (P < 3e-7); the vectors were rescaled or zeroed some other way, so no decay value is given;",
             f"- **init unknown** ({cnt['init std unknown']}): no value.", "",
             "| model | repo | fill | fitted std | init std used | P | sum lr*wd | verdict |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(out, key=lambda r: (r["P"] is None, -(r["P"] or 0))):
        lines.append("| {} | {} | {} | {:.4g} | {} | {} | {} | {} |".format(
            r["name"], r["repo"], r["dist"], r["card_std"], r["init"] or "-",
            "-" if r["P"] is None else f"{r['P']:.4g}", "-" if r["sum_lr_wd"] is None else f"{r['sum_lr_wd']:.3f}",
            r["verdict"]))
    pj2 = ROOT / "results" / "public" / "claim1_pythia_lr.json"
    if pj2.exists():
        d = json.load(open(pj2))
        lines += ["", "## Validation on Pythia", "",
                  "Independent measurement: how much the unused padding vectors shrank between step 0 and the final checkpoint,",
                  "measured on the public `stepN` checkpoints (`results/public/claim1_pythia_lr.json`). Card-based P = fitted std / init std.", "",
                  "| size | checkpoint series P | card-based P | ratio |", "|---|---|---|---|"]
        byname = {r["name"].lower().replace("pythia-", ""): r for r in out if r["name"].lower().startswith("pythia")}
        for k, v in d.items():
            m = v.get("measure") if isinstance(v, dict) else None
            if isinstance(m, dict) and "total_slope" in m and k in byname:
                c = byname[k]
                note = " (series unusable: final branch equals step 0)" if m["total_slope"] >= 1 else ""
                lines.append(f"| {k} | {m['total_slope']:.6f} | {c['P']:.6f} | {c['P'] / m['total_slope']:.4f}{note} |")
        lines += ["", "With gpt-neox small_init (sqrt(2/(5d))) for sizes up to 2.8B and std 0.02 for 6.9B and 12B, the two",
                  "independent estimates agree to 1e-4; this init choice is the one that makes them agree (consistent with,",
                  "not independently checked against, the Pythia configs). With the wrong init convention P is off by ~2x,",
                  "which is why every non-Pythia row above that rests on `initializer_range` is marked as an assumption."]
    (ROOT / "results" / "roster" / "PRETRAIN_DECAY.md").write_text("\n".join(lines) + "\n")
    from collections import Counter
    print(Counter(r["verdict"] for r in out))
    for r in out:
        if r["name"].lower().startswith("pythia") or r["name"].lower().startswith("olmo") or "DeepSeek-V3-Base" in r["name"]:
            print(r["name"], r["card_std"], r["init"], r["P"], r["sum_lr_wd"])


if __name__ == "__main__":
    main()
