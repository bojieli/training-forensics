#!/usr/bin/env python3
"""Membership inference, held-out only.

Method (fixed in advance, chosen on m01 in membership2.py; no choice is made here):
  "excl + wagree@interp max"
    1. exclusion: a candidate document with a token whose vector is bit-identical
       to the base model, or whose single-update tokens imply different training
       steps, was not trained on;
    2. score: for each of the document's single-update tokens, the gradient-
       weighted fraction of coordinates where the token vector's update has the
       sign of minus the document's gradient, at the model state interpolated to
       that token's training step (membership2.feat); document score = max.
  Threshold = the largest score of any non-member (zero false alarms).

Reported numbers:
  m01        split-half: the threshold is set on half the non-members and TPR is
             measured on the other half of members and non-members (500 splits),
             so neither the method nor the threshold sees the evaluated half.
  g05 (wiki) a second dataset, never used to develop anything: same fixed method.

Baselines on the same candidates (all use the fine-tuned model, which the
attacker has, plus the public base model):
  loss       -mean token loss under the fine-tune
  ref        loss(base) - loss(fine-tune)            (reference-model attack)
  mink20     mean of the lowest 20% token log-probs under the fine-tune (Min-K% Prob)
  lira       LiRA with 8 shadow fine-tunes of the base model on random halves of the
             candidate pool (same recipe as claimed); per-document Gaussian fits of
             the in / out loss, global variance (offline-stable variant)

  python3 scripts/gaps/membership3.py baselines runs/ctl/m01
  python3 scripts/gaps/membership3.py score runs/ctl/m01
"""
from __future__ import annotations

import json
import os
import math
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "controlled"))
DEV = "cuda"
torch.cuda.set_per_process_memory_fraction(0.3)


def load_run(run):
    run = pathlib.Path(run)
    t = json.load(open(run / "truth.json"))
    toks = {int(k): v for k, v in json.load(open(run / "tokens.json")).items()}
    return run, t, toks


@torch.no_grad()
def token_logprobs(model, docs, toks, bs=32):
    """doc -> np.array of per-token log-probs (positions 1..L-1)."""
    out = {}
    order = sorted(docs, key=lambda d: len(toks[d]))
    for a in range(0, len(order), bs):
        ds = order[a:a + bs]
        L = max(len(toks[d]) for d in ds)
        ids = torch.zeros((len(ds), L), dtype=torch.long)
        att = torch.zeros((len(ds), L), dtype=torch.long)
        for r, d in enumerate(ds):
            ids[r, :len(toks[d])] = torch.tensor(toks[d]); att[r, :len(toks[d])] = 1
        ids, att = ids.to(DEV), att.to(DEV)
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[:, :-1].float(), -1)
        g = lp.gather(2, ids[:, 1:, None])[:, :, 0]
        for r, d in enumerate(ds):
            n = len(toks[d]) - 1
            out[d] = g[r, :n].cpu().numpy()
    return out


def train_shadow(base, docs, toks, a, seed):
    """Same recipe as the claimed run (AdamW, linear/cosine with warm-up, bs, lr), plain loop."""
    from transformers import AutoModelForCausalLM, get_scheduler
    torch.manual_seed(seed)
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32).to(DEV)
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=a["lr"], weight_decay=a["wd"], betas=(a["b1"], a["b2"]), eps=a["eps"])
    bs = a["bs"]
    steps = len(docs) // bs
    kind = a["sched"]
    kw = {}
    if kind == "cosine_min":
        kind, kw = "cosine_with_min_lr", {"min_lr_rate": a["min_lr_ratio"]}
    sch = get_scheduler(kind, opt, num_warmup_steps=a["warmup"], num_training_steps=steps, scheduler_specific_kwargs=kw)
    perm = np.random.default_rng(seed).permutation(docs)
    for s in range(steps):
        ds = perm[s * bs:(s + 1) * bs]
        L = max(len(toks[d]) for d in ds)
        ids = torch.full((len(ds), L), 0, dtype=torch.long)
        att = torch.zeros((len(ds), L), dtype=torch.long)
        for r, d in enumerate(ds):
            ids[r, :len(toks[d])] = torch.tensor(toks[d]); att[r, :len(toks[d])] = 1
        ids, att = ids.to(DEV), att.to(DEV)
        loss = m(input_ids=ids, attention_mask=att, labels=ids.masked_fill(att == 0, -100)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    m.eval()
    return m


def baselines(run, n_shadow=8):
    from transformers import AutoModelForCausalLM
    run, t, toks = load_run(run)
    a = t["args"]
    pool = list(t["trained"]) + list(t["outside"])
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(a["model"], torch_dtype=torch.float32).to(DEV).eval()
    lp_base = token_logprobs(base, pool, toks)
    del base
    ft = AutoModelForCausalLM.from_pretrained(str(run / "model"), torch_dtype=torch.float32).to(DEV).eval()
    lp_ft = token_logprobs(ft, pool, toks)
    del ft
    torch.cuda.empty_cache()
    rng = np.random.default_rng(12345)
    shadow_loss = np.zeros((n_shadow, len(pool)))
    shadow_in = np.zeros((n_shadow, len(pool)), bool)
    for i in range(n_shadow):
        inn = rng.permutation(len(pool))[:len(pool) // 2]
        shadow_in[i, inn] = True
        m = train_shadow(a["model"], [pool[j] for j in inn], toks, a, seed=1000 + i)
        lp = token_logprobs(m, pool, toks)
        shadow_loss[i] = [-lp[d].mean() for d in pool]
        del m; torch.cuda.empty_cache()
        print("shadow", i, round(time.time() - t0), flush=True)
    np.savez(run / "membership3_baselines.npz", pool=np.array(pool),
             loss_ft=np.array([-lp_ft[d].mean() for d in pool]),
             loss_base=np.array([-lp_base[d].mean() for d in pool]),
             mink20=np.array([np.sort(lp_ft[d])[:max(1, int(0.2 * len(lp_ft[d])))].mean() for d in pool]),
             shadow_loss=shadow_loss, shadow_in=shadow_in, seconds=time.time() - t0)


def lira_scores(B):
    """Fixed-variance LiRA on per-document mean loss (lower loss = member)."""
    L, I = B["shadow_loss"], B["shadow_in"]
    x = B["loss_ft"]
    mu_in = np.array([L[I[:, j], j].mean() if I[:, j].any() else np.nan for j in range(L.shape[1])])
    mu_out = np.array([L[~I[:, j], j].mean() if (~I[:, j]).any() else np.nan for j in range(L.shape[1])])
    r_in = np.concatenate([L[I[:, j], j] - mu_in[j] for j in range(L.shape[1])])
    r_out = np.concatenate([L[~I[:, j], j] - mu_out[j] for j in range(L.shape[1])])
    s_in, s_out = r_in.std(), r_out.std()
    gl = lambda v, m, s: -0.5 * ((v - m) / s) ** 2 - math.log(s)
    sc = gl(x, mu_in, s_in) - gl(x, mu_out, s_out)
    return np.nan_to_num(sc, nan=-1e9)


def tpr_at_0fp(mem, out):
    return float((mem > out.max()).mean())


def split_half(mem, out, n=500, seed=0):
    """Threshold from half the non-members, TPR on the other half of members; also
    count false alarms on the other half of non-members."""
    rng = np.random.default_rng(seed)
    tprs, fps, t1, f1 = [], [], [], []
    for _ in range(n):
        pm, po = rng.permutation(len(mem)), rng.permutation(len(out))
        cal = out[po[:len(po) // 2]]
        thr = cal.max()
        tprs.append(float((mem[pm[len(pm) // 2:]] > thr).mean()))
        fps.append(int((out[po[len(po) // 2:]] > thr).sum()))
        thr1 = np.sort(cal)[-3]                      # 2 of 2,000 calibration non-members above: 0.1%
        t1.append(float((mem[pm[len(pm) // 2:]] > thr1).mean()))
        f1.append(float((out[po[len(po) // 2:]] > thr1).mean()))
    return dict(tpr_median=float(np.median(tprs)), tpr_at_0p1pct_fpr_median=float(np.median(t1)),
                heldout_fpr_at_0p1pct_mean=float(np.mean(f1)),
                tpr_mean=float(np.mean(tprs)), tpr_q05=float(np.quantile(tprs, .05)),
                tpr_q95=float(np.quantile(tprs, .95)), heldout_false_alarms_mean=float(np.mean(fps)),
                heldout_false_alarms_q95=float(np.quantile(fps, .95)))


def clopper_pearson(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def operating_points(scores, nm, fprs=(0.0, 0.001, 0.01)):
    """Same operating points for every attack (tech review M2): threshold from half
    the non-members, TPR and realised FPR on the other half (500 splits); plus the
    full-sample TPR with a 95% Clopper-Pearson interval."""
    out = {}
    rng = np.random.default_rng(0)
    for n, s in scores.items():
        mem, oth = s[:nm], s[nm:]
        row = {}
        for f in fprs:
            k_allow = int(math.floor(f * len(oth)))
            srt = np.sort(oth)
            thr = srt[-1] if k_allow == 0 else srt[-(k_allow + 1)]
            tp = int((mem > thr).sum())
            lo, hi = clopper_pearson(tp, len(mem))
            t_, fp_ = [], []
            for _ in range(500):
                pm, po = rng.permutation(len(mem)), rng.permutation(len(oth))
                cal = np.sort(oth[po[:len(po) // 2]])
                kk = int(math.floor(f * len(cal)))
                th = cal[-1] if kk == 0 else cal[-(kk + 1)]
                t_.append(float((mem[pm[len(pm) // 2:]] > th).mean()))
                fp_.append(float((oth[po[len(po) // 2:]] > th).mean()))
            row[f"fpr_{f:g}"] = dict(full_sample_tpr=tp / len(mem), full_sample_ci95=[lo, hi],
                                     heldout_tpr_median=float(np.median(t_)),
                                     heldout_tpr_q05_q95=[float(np.quantile(t_, .05)), float(np.quantile(t_, .95))],
                                     heldout_fpr_realised_mean=float(np.mean(fp_)))
        out[n] = row
    return out


def our_scores(run, t, toks):
    """Fixed method: exclusion + max wagree at the interpolated state (membership2 features)."""
    import membership2 as M2
    from safetensors.torch import load_file
    run_, t_, toks_, rd, flat, sgn, curve, cfrac = M2.load(run)
    steps, Drow = M2.row_steps(run, flat, curve)
    e = load_file(str(run / "emb.safetensors"))
    F = np.load(run / ("membership2_feat_blind.npz" if os.environ.get("TF_BLIND") == "1" else "membership2_feat.npz"))
    pool = list(t["trained"]) + list(t["outside"])
    excluded = {}
    for d in pool:
        body = set(toks[d][:-1])
        fl = [x for x in body if flat[x]]
        lD = [np.log(Drow[x]) for x in fl]
        excluded[d] = any(not rd.moved[x] for x in body) or (len(fl) > 1 and (max(lD) - min(lD)) > 1.5e-3)
    tok = torch.tensor(F["tok"]); kind = F["kind"]; doc = F["doc"]
    delta = (e["W1"].double() - rd.decay * e["W0"].double())
    S = torch.sign(delta[tok]).float()
    g = torch.tensor(F["g"]).float()
    wag = (((torch.sign(-g) == S).float() * g.abs()).sum(1) / g.abs().sum(1).clamp_min(1e-30)).numpy()
    best = {}
    for i in np.nonzero(kind == 1)[0]:
        d = int(doc[i])
        best[d] = max(best.get(d, -1.0), float(wag[i]))
    return np.array([(-1e9 if (excluded[d] or d not in best) else best[d]) for d in pool]), excluded


def score(run):
    run, t, toks = load_run(run)
    members, outside = list(t["trained"]), list(t["outside"])
    nm = len(members)
    res = dict(run=run.name, source=t["args"]["source"], members=nm, outside=len(outside))
    ours, excluded = our_scores(run, t, toks)
    B = np.load(run / "membership3_baselines.npz")
    assert list(B["pool"]) == members + outside
    scores = {"ours (excl + wagree@interp max)": ours, "loss": -B["loss_ft"],
              "ref (base - ft loss)": B["loss_base"] - B["loss_ft"], "mink20": B["mink20"],
              "lira (8 shadows)": lira_scores(B)}
    res["members_with_single_update_token"] = int(sum(1 for v in ours[:nm] if v > -1e8)) + \
        int(sum(1 for d in members if excluded[d]))
    res["outside_excluded"] = int(sum(excluded[d] for d in outside))
    res["members_excluded_wrongly"] = int(sum(excluded[d] for d in members))
    res["attacks"] = {}
    for n, s in scores.items():
        mem, out = s[:nm], s[nm:]
        res["attacks"][n] = dict(full_sample_tpr_at_0fp=tpr_at_0fp(mem, out), split_half=split_half(mem, out))
    res["recipe_source"] = "blind (weights_only.json)" if os.environ.get("TF_BLIND") == "1" else "truth.json"
    res["operating_points"] = operating_points(scores, nm)
    dst = ROOT / "results" / "gaps" / (f"{run.name}.membership3" + (".blind" if os.environ.get("TF_BLIND") == "1" else "") + ".json")
    json.dump(res, open(dst, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    {"baselines": baselines, "score": score}[sys.argv[1]](sys.argv[2])
