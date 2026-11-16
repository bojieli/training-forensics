#!/usr/bin/env python3
"""Membership without memorisation, v2: sharper embedding-row test.

A single-touch ("flat") input-embedding row t moved by D(s)*sign(-g), where g is
the gradient of the step-s batch w.r.t. row t at the model state theta_{s-1}.
v1 (membership.py) compared the row's sign pattern with the candidate
document's gradient at the BASE model and took the max over tokens.

v2:
  * reads the step s off the row's D (recovered LR schedule) and evaluates the
    candidate's gradient at theta_s ~ theta_0 + c(s) (theta_N - theta_0),
    c(s) = cumulative-LR fraction before step s (fine-tuned model is public);
  * hard exclusion: a document with a token whose row never moved was not trained;
  * step consistency: all single-touch tokens of a trained doc share one step;
  * per-token calibration against reference contexts of the same token drawn
    from an independent public corpus (wikitext-103), i.e. a z-score that
    removes the context-independent part of the token's gradient;
  * |g|-weighted agreement and evidence summed over the doc's tokens.

No member labels are used except to compute TPR / FP at the end
(threshold = max outside score). The true LR recipe is used for D(s).

Findings on runs/ctl/m01 (results/controlled/m01.membership2.json):
  * the two structural exclusions do most of the work (v1 3.7% -> 23-24% TPR@0FP):
    they remove 2785 of 4000 outside docs outright and leave 397 with any evidence;
  * state matters: sign match is ~0.99 only within ~20 steps of the true state
    (steps >480 at theta_N, steps <10 at theta_0); mid-run the true state is far
    from the theta_0-theta_N line, weighted agreement stays ~0.70-0.75 for any c
    (11-point scan of c), so interpolation helps only the early/late steps
    (24% -> 41%);
  * |delta_i| carries |g_i| (Adam eps): Spearman(|delta|, |g|) = 0.84 at the right
    state. The signed-rank cosine is as good as, not better than, weighted signs;
  * per-token z-scores against wikitext contexts (K=8) hurt; projecting out the
    token's generic direction is neutral; summing over tokens hurts at 0 FP;
  * 99.9% of single-touch tokens are unique to one doc of their batch, so batch
    mixing (bs 8) is not a failure mode.

"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics.diff import row_diff  # noqa: E402
from trainforensics import blind  # noqa: E402

DEV = "cuda"
torch.cuda.set_per_process_memory_fraction(0.3)     # shared GPU: stay well under 40 GB


def load(run):
    run = pathlib.Path(run)
    t = json.load(open(run / "truth.json"))
    toks = {int(k): v for k, v in json.load(open(run / "tokens.json")).items()}
    e = load_file(str(run / "emb.safetensors"))
    W0, W1 = e["W0"], e["W1"]
    rd = row_diff(W0, W1, torch.float32)
    flat = rd.moved & (rd.flat_frac > 0.9)
    sgn = torch.sign(W1.double() - rd.decay * W0.double()).float()
    if os.environ.get("TF_BLIND") == "1":
        # tech review M2: the step curve from the blind-recovered recipe
        # (results/controlled/weights_only.json, W0/W1 only), not truth.json
        import score as _S
        _wd = pathlib.Path(__file__).resolve().parents[2] / "results" / "controlled"
        _all = {}
        for _f in ("weights_only.json", "weights_only_new.json"):
            if (_wd / _f).exists():
                _all.update(json.load(open(_wd / _f)))
        wo = _all[run.name]["read"]
        lr = _S.hf_lr(dict(kind=wo["kind"], N=wo["N"], warmup=wo["warmup"], peak=wo["peak_snap"],
                           min_ratio=wo.get("min_ratio", 0.1)), wo["N"])
        curve = blind.moves_curve(lr, wo.get("wd_snap") or 0.0)
    else:
        lr = np.array([0.0] + t["lr"])
        curve = blind.moves_curve(lr, t["args"]["wd"])
    cum = np.concatenate([[0.0], np.cumsum(lr[1:])])       # cum[s] = sum lr[1..s]
    cfrac = cum / cum[-1]                                   # state before step s: cfrac[s-1]
    return run, t, toks, rd, flat, sgn, curve, cfrac


def row_steps(run, flat, curve, lo=-3e-4, hi=9e-4, q=0.99):
    """flat token -> candidate steps from its move D.
    Adam's eps makes every coordinate move slightly LESS than the eps-free
    prediction D(s), so the row's D is read from a high quantile of |delta| and
    matched one-sidedly: lo <= log D(s) - log D_row <= hi. Returns (steps, D_row)."""
    e = load_file(str(run / "emb.safetensors"))
    idx = torch.tensor(np.nonzero(flat)[0])
    A = (e["W1"][idx].double() - e["W0"][idx].double()).abs()
    Dq = torch.quantile(A, q, dim=1).numpy()
    l = np.log(np.maximum(curve[1:], 1e-300))
    out, Drow = {}, {}
    for tk, d in zip(idx.tolist(), Dq):
        diff = l - np.log(d)
        c = np.nonzero((diff >= lo) & (diff <= hi))[0] + 1
        out[tk] = [int(x) for x in c] if len(c) else [int(np.argmin(np.abs(diff - 3e-4))) + 1]
        Drow[tk] = float(d)
    return out, Drow


class Interp:
    """theta(c) = theta0 + c (thetaN - theta0), set in place."""

    def __init__(self, base_name, ft_dir):
        self.m = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float32).to(DEV).eval()
        ft = AutoModelForCausalLM.from_pretrained(str(ft_dir), torch_dtype=torch.float32).to(DEV)
        self.p = [p for _, p in self.m.named_parameters()]
        self.p0 = [p.detach().clone() for p in self.p]
        ftp = dict(ft.named_parameters())
        self.dp = [ftp[n].detach() - p for (n, _), p in zip(self.m.named_parameters(), self.p0)]
        del ft
        for p in self.p:
            p.requires_grad_(False)
        self.emb = self.m.get_input_embeddings()

    @torch.no_grad()
    def set(self, c):
        for p, p0, dp in zip(self.p, self.p0, self.dp):
            p.copy_(p0 + c * dp)

    def tok_grads(self, items, bs=32):
        """items: list of (ids, [tokens wanted]). Returns list of dict tok -> grad [d]
        (gradient of the doc's summed LM loss w.r.t. the token's embedding row)."""
        order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
        res = [None] * len(items)
        for a in range(0, len(order), bs):
            idx = order[a:a + bs]
            L = max(len(items[i][0]) for i in idx)
            ids = torch.zeros((len(idx), L), dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, i in enumerate(idx):
                n = len(items[i][0])
                ids[r, :n] = torch.tensor(items[i][0])
                att[r, :n] = 1
            ids, att = ids.to(DEV), att.to(DEV)
            E = self.emb(ids).detach().requires_grad_(True)
            logits = self.m(inputs_embeds=E, attention_mask=att).logits
            lab = ids.masked_fill(att == 0, -100)
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                                     lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
            (G,) = torch.autograd.grad(loss, E)
            for r, i in enumerate(idx):
                d = {}
                for tk in items[i][1]:
                    m = (ids[r] == tk) & (att[r] == 1)
                    d[tk] = G[r][m].sum(0).cpu()
                res[i] = d
            del logits, loss, G, E
            torch.cuda.empty_cache()            # varying batch shapes fragment the cache badly
        return res


def cand_states(steps_tk):
    """Candidate steps used for a row: first and last (the two branches of the D(s) curve)."""
    return sorted({steps_tk[0], steps_tk[-1]})


def save_grads(path, rec, **extra):
    g = torch.stack(rec["g"])
    g = (g / g.abs().amax(1, keepdim=True).clamp_min(1e-30)).half().numpy()     # only direction / ranks are used
    np.savez(path, g=g, **{k: np.array(v) for k, v in rec.items() if k != "g"}, **extra)


def feat(run):
    """Gradients of every candidate doc w.r.t. each of its single-touch rows, at
    base (kind 0), fine-tuned (kind 2) and interpolated-at-the-row's-step (kind 1) states."""
    run, t, toks, rd, flat, sgn, curve, cfrac = load(run)
    steps, _ = row_steps(run, flat, curve)
    N = len(curve) - 1
    by_step, static = {}, []
    for d in t["trained"] + t["outside"]:              # candidate pool; labels not used
        ids = toks[d]
        fl = sorted(tk for tk in set(ids[:-1]) if flat[tk])
        if len(ids) < 2 or not fl:
            continue
        static.append((d, fl))
        per = {}
        for tk in fl:
            for s in cand_states(steps[tk]):
                per.setdefault(s, []).append(tk)
        for s, tl in per.items():
            by_step.setdefault(s, []).append((d, tl))
    M = Interp(t["args"]["model"], run / "model")
    rec = dict(doc=[], tok=[], step=[], kind=[], g=[])

    def run_items(lst, step, kind):
        out = M.tok_grads([(toks[d], tl) for d, tl in lst])
        for (d, tl), o in zip(lst, out):
            for tk in tl:
                rec["doc"].append(d); rec["tok"].append(tk); rec["step"].append(step)
                rec["kind"].append(kind); rec["g"].append(o[tk])

    t0 = time.time()
    M.set(0.0); run_items(static, 0, 0)
    M.set(1.0); run_items(static, N + 1, 2)
    for s in sorted(by_step):
        M.set(float(cfrac[s - 1])); run_items(by_step[s], s, 1)
    el = time.time() - t0
    save_grads(run / ("membership2_feat_blind.npz" if os.environ.get("TF_BLIND") == "1" else "membership2_feat.npz"),
               rec, seconds=el)
    print("feat pairs", len(rec["doc"]), "seconds", round(el, 1))


def ref(run, K=8, win_before=96, win_after=32):
    """Null calibration: gradients of each single-touch token in K contexts from an
    independent public corpus (wikitext-103), at the same interpolated state(s)."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    run, t, toks, rd, flat, sgn, curve, cfrac = load(run)
    steps, _ = row_steps(run, flat, curve)
    need = sorted({tk for d in t["trained"] + t["outside"] for tk in toks[d][:-1] if flat[tk]})
    tok = AutoTokenizer.from_pretrained(t["args"]["model"])
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    mask = np.zeros(len(flat), bool); mask[need] = True
    occ = {tk: [] for tk in need}
    pool = []
    t0 = time.time()
    for a in range(0, 400000, 10000):
        texts = [x for x in ds[a:a + 10000]["text"] if len(x) > 300]
        for ids in tok(texts)["input_ids"]:
            arr = np.array(ids[:1024])
            used = set()
            for p in np.nonzero(mask[arr])[0]:
                tk = int(arr[p])
                if p < 8 or p >= len(arr) - 1 or tk in used:
                    continue
                used.add(tk)
                occ[tk].append(arr[max(0, p - win_before):p + win_after].tolist())
                if len(occ[tk]) >= K:
                    mask[tk] = False
            if len(pool) < 4000 and len(arr) >= win_before + win_after:
                pool.append(arr[:win_before + win_after].tolist())
        if not mask.any():
            break
    n_real = {tk: len(v) for tk, v in occ.items()}
    rng = np.random.default_rng(0)
    for tk, v in occ.items():                    # token never/rarely seen in wikitext: plant it in random passages
        while len(v) < K:
            w = list(pool[rng.integers(len(pool))]); w[win_before] = tk
            v.append(w)
    print("ref contexts: tokens", len(need), "with K real", sum(v >= K for v in n_real.values()),
          "with 0 real", sum(v == 0 for v in n_real.values()), "index seconds", round(time.time() - t0, 1))
    by_step = {}
    for tk in need:
        for s in cand_states(steps[tk]):
            by_step.setdefault(s, []).extend((tk, k) for k in range(K))
    M = Interp(t["args"]["model"], run / "model")
    rec = dict(tok=[], step=[], real=[], g=[])
    t0 = time.time()
    for s in sorted(by_step):
        M.set(float(cfrac[s - 1]))
        out = M.tok_grads([(occ[tk][k], [tk]) for tk, k in by_step[s]], bs=64)
        for (tk, k), o in zip(by_step[s], out):
            rec["tok"].append(tk); rec["step"].append(s); rec["real"].append(k < n_real[tk]); rec["g"].append(o[tk])
    el = time.time() - t0
    save_grads(run / "membership2_ref.npz", rec, seconds=el)
    print("ref grads", len(rec["tok"]), "seconds", round(el, 1))


def srvec(sign, mag):
    """Signed-rank vector (unit norm): sign * rank(|.|). Used for both the row move
    (|delta_i| grows with |g_i| because of Adam's eps) and a candidate gradient."""
    r = mag.argsort(-1).argsort(-1).float() + 1
    v = sign * r
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def score(run):
    t_start = time.time()
    run, t, toks, rd, flat, sgn, curve, cfrac = load(run)
    steps, Drow = row_steps(run, flat, curve)
    e = load_file(str(run / "emb.safetensors"))
    members, outside = t["trained"], t["outside"]
    pool = members + outside
    F = np.load(run / "membership2_feat.npz")
    R = np.load(run / "membership2_ref.npz")
    dev = DEV

    # ---- per-document structural evidence (no gradients, no labels)
    info = {}
    for d in pool:
        body = set(toks[d][:-1])
        fl = [x for x in body if flat[x]]
        lD = [np.log(Drow[x]) for x in fl]
        info[d] = dict(unmoved=sum(1 for x in body if not rd.moved[x]),
                       incons=len(fl) > 1 and (max(lD) - min(lD)) > 1.5e-3, nflat=len(fl))
    excluded = {d: info[d]["unmoved"] > 0 or info[d]["incons"] for d in pool}

    # ---- pair statistics
    tok = torch.tensor(F["tok"]); kind = F["kind"]; step = F["step"]; doc = F["doc"]
    delta = (e["W1"].double() - rd.decay * e["W0"].double())
    S = torch.sign(delta[tok]).float().to(dev)
    X = srvec(S, delta[tok].abs().float().to(dev))
    g = torch.tensor(F["g"]).float().to(dev)
    match = (torch.sign(-g) == S).float()
    nz = (g != 0).float()
    st = dict(agree=(match * nz).sum(1) / nz.sum(1).clamp_min(1),
              wagree=(match * g.abs()).sum(1) / g.abs().sum(1).clamp_min(1e-30))
    Y = srvec(-torch.sign(g), g.abs())
    st["srcos"] = (X * Y).sum(1)
    st = {k: v.cpu().numpy() for k, v in st.items()}

    # ---- reference-context calibration for interpolated-state pairs
    rkey = {}
    for i, (a, b) in enumerate(zip(R["tok"].tolist(), R["step"].tolist())):
        rkey.setdefault((a, b), []).append(i)
    K = min(len(v) for v in rkey.values())
    i1 = np.nonzero(kind == 1)[0]
    ridx = torch.tensor([rkey[(int(F["tok"][i]), int(step[i]))][:K] for i in i1])       # [P, K]
    gr_all = torch.tensor(R["g"]).float()
    for nm in ("z_wagree", "z_srcos", "res_mean", "res_span", "ref_mean_wagree", "ref_mean_srcos"):
        st[nm] = np.full(len(doc), np.nan)
    for a in range(0, len(i1), 2048):
        ii = i1[a:a + 2048]
        gr = gr_all[ridx[a:a + 2048]].to(dev)                                           # [p, K, d]
        Sx, Xx, Yx = S[ii], X[ii], Y[ii]
        m = (torch.sign(-gr) == Sx[:, None]).float()
        wa = (m * gr.abs()).sum(2) / gr.abs().sum(2).clamp_min(1e-30)                   # [p, K]
        Yr = srvec(-torch.sign(gr), gr.abs())
        sc = (Yr * Xx[:, None]).sum(2)
        st["z_wagree"][ii] = ((torch.tensor(st["wagree"][ii], device=dev) - wa.mean(1)) / wa.std(1).clamp_min(0.02)).cpu().numpy()
        st["z_srcos"][ii] = ((torch.tensor(st["srcos"][ii], device=dev) - sc.mean(1)) / sc.std(1).clamp_min(0.04)).cpu().numpy()
        st["ref_mean_wagree"][ii] = wa.mean(1).cpu().numpy()
        st["ref_mean_srcos"][ii] = sc.mean(1).cpu().numpy()
        mu = Yr.mean(1); mu = mu / mu.norm(dim=1, keepdim=True)                         # generic direction of the token
        def resid(v, Q):                                                                # remove span(Q) ; Q [p, d, k] orthonormal
            return v - (Q @ (Q.transpose(1, 2) @ v[:, :, None]))[:, :, 0]
        def cosr(Q):
            x, y = resid(Xx, Q), resid(Yx, Q)
            return ((x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1)).clamp_min(1e-30)).cpu().numpy()
        st["res_mean"][ii] = cosr(mu[:, :, None])
        Q, _ = torch.linalg.qr(Yr.transpose(1, 2))
        st["res_span"][ii] = cosr(Q)

    # ---- token level: (doc, tok) -> stat ; interpolated state = max over the row's candidate steps
    def tok_level(name, k):
        out = {}
        for i in np.nonzero(kind == k)[0]:
            key = (int(doc[i]), int(F["tok"][i]))
            v = float(st[name][i])
            if key not in out or v > out[key]:
                out[key] = v
        return out

    def doc_scores(tl, agg, excl, null=0.0):
        per = {}
        for (d, _), v in tl.items():
            per.setdefault(d, []).append(v - null)
        sc = {}
        for d in pool:
            if d not in per or (excl and excluded[d]):
                sc[d] = -1e9
            else:
                sc[d] = max(per[d]) if agg == "max" else float(sum(per[d]))
        return sc

    variants = {}
    T = {(n, k): tok_level(n, k) for n in ("agree", "wagree", "srcos") for k in (0, 1, 2)}
    for n in ("z_wagree", "z_srcos", "res_mean", "res_span"):
        T[(n, 1)] = tok_level(n, 1)
    kn = {0: "base", 1: "interp", 2: "ft"}
    variants["v1 agree@base max (membership.py)"] = doc_scores(T[("agree", 0)], "max", False)
    variants["wagree@interp max, no exclusion"] = doc_scores(T[("wagree", 1)], "max", False)
    variants["z_wagree@interp max, no exclusion"] = doc_scores(T[("z_wagree", 1)], "max", False)
    variants["res_span@interp max, no exclusion"] = doc_scores(T[("res_span", 1)], "max", False)
    for n in ("agree", "wagree", "srcos"):
        for k in (0, 2, 1):
            variants[f"excl + {n}@{kn[k]} max"] = doc_scores(T[(n, k)], "max", True)
    for n in ("z_wagree", "z_srcos", "res_mean", "res_span"):
        variants[f"excl + {n}@interp max"] = doc_scores(T[(n, 1)], "max", True)
    def exclusive(tl):
        """Pool-aware: a single-touch row has one cause, so among the candidate docs that
        contain the token (and are not structurally excluded) only the best-matching keeps it.
        NOTE: a doc's score then depends on the other candidates - not a per-document test."""
        best = {}
        for (d, tk), v in tl.items():
            if not excluded[d] and (tk not in best or v > best[tk][0]):
                best[tk] = (v, d)
        return {(d, tk): v for (d, tk), v in tl.items() if not excluded[d] and best[tk][1] == d}

    for n in ("wagree", "srcos", "z_wagree", "res_span"):
        variants[f"excl + {n}@interp max, pool-exclusive"] = doc_scores(exclusive(T[(n, 1)]), "max", True)
    comb = {key: (T[("srcos", 0)][key] + T[("srcos", 1)][key] + T[("srcos", 2)][key]) / 3 for key in T[("srcos", 1)]}
    variants["excl + srcos mean(base,interp,ft) max"] = doc_scores(comb, "max", True)
    comb = {key: max(T[("srcos", 0)][key], T[("srcos", 1)][key], T[("srcos", 2)][key]) for key in T[("srcos", 1)]}
    variants["excl + srcos max(base,interp,ft) max"] = doc_scores(comb, "max", True)
    variants["excl + wagree@interp sum(T-0.5)"] = doc_scores(T[("wagree", 1)], "sum", True, 0.5)
    variants["excl + srcos@interp sum"] = doc_scores(T[("srcos", 1)], "sum", True)
    variants["excl + z_wagree@interp sum"] = doc_scores(T[("z_wagree", 1)], "sum", True)
    variants["excl + z_srcos@interp sum"] = doc_scores(T[("z_srcos", 1)], "sum", True)
    variants["excl + res_span@interp sum"] = doc_scores(T[("res_span", 1)], "sum", True)
    variants["excl + res_mean@interp sum"] = doc_scores(T[("res_mean", 1)], "sum", True)

    # ---- evaluation (only place labels are used)
    elig = [d for d in members if info[d]["nflat"] > 0]
    res = dict(run=str(run), members=len(members), outside=len(outside),
               members_eligible=len(elig),
               members_excluded_wrongly=sum(excluded[d] for d in members),
               outside_excluded_unmoved_token=sum(info[d]["unmoved"] > 0 for d in outside),
               outside_excluded_inconsistent_D=sum(info[d]["unmoved"] == 0 and info[d]["incons"] for d in outside),
               outside_not_excluded=sum(not excluded[d] for d in outside),
               outside_not_excluded_with_single_touch_token=sum(not excluded[d] and info[d]["nflat"] > 0 for d in outside),
               ref_contexts_per_token=int(K), ref_real_fraction=float(R["real"].mean()))
    mem_arr = {n: np.array([sc[d] for d in members]) for n, sc in variants.items()}
    out_arr = {n: np.array([sc[d] for d in outside]) for n, sc in variants.items()}
    table = {}
    for n in variants:
        so = np.sort(out_arr[n])
        table[n] = dict(tpr_at_0fp=float((mem_arr[n] > so[-1]).mean()),
                        tpr_at_0fp_eligible=float((mem_arr[n] > so[-1]).sum() / len(elig)),
                        tpr_at_1fp=float((mem_arr[n] > so[-2]).mean()),
                        threshold=float(so[-1]), outside_competing=int((so > -1e8).sum()))
    res["variants"] = table
    # selection-bias check: pick the best variant on a random half, evaluate on the other half
    rng = np.random.default_rng(0)
    names = [n for n in variants if n.startswith("excl") and "pool-exclusive" not in n]
    held, picks, per_var = [], {}, {n: [] for n in names}
    for _ in range(500):
        pm = rng.permutation(len(members)); po = rng.permutation(len(outside))
        A = (pm[:len(pm) // 2], po[:len(po) // 2]); B = (pm[len(pm) // 2:], po[len(po) // 2:])
        f = lambda n, h: float((mem_arr[n][h[0]] > out_arr[n][h[1]].max()).mean())
        best = max(names, key=lambda n: f(n, A))
        held.append(f(best, B)); picks[best] = picks.get(best, 0) + 1
        for n in names:
            per_var[n].append(f(n, B))
    res["split_half"] = dict(note="variant chosen on half A (2000 members / 2000 outside), TPR@0FP on half B; 500 random splits",
                             heldout_tpr_mean=float(np.mean(held)), heldout_tpr_q05=float(np.quantile(held, .05)),
                             heldout_tpr_q95=float(np.quantile(held, .95)), picks=picks,
                             per_variant_half_mean={n: float(np.mean(v)) for n, v in per_var.items()})
    # diagnostics with truth: who is found, by training step (headline = best full-sample excl variant)
    head = max(names, key=lambda n: table[n]["tpr_at_0fp"])
    dstep = {}
    for s_, ks in enumerate(t["docs_per_step"], 1):
        for k in ks:
            dstep[members[k]] = s_
    thr = table[head]["threshold"]
    bins = [1, 21, 101, 201, 301, 401, 451, 481, 501]
    res["headline"] = head
    res["headline_by_step"] = {f"{a}-{b - 1}": dict(
        eligible=sum(1 for d in elig if a <= dstep[d] < b),
        found=sum(1 for d in elig if a <= dstep[d] < b and variants[head][d] > thr)) for a, b in zip(bins[:-1], bins[1:])}
    res["headline_by_n_single_touch_tokens"] = {str(k): dict(
        members=sum(1 for d in elig if min(info[d]["nflat"], 4) == k),
        found=sum(1 for d in elig if min(info[d]["nflat"], 4) == k and variants[head][d] > thr)) for k in (1, 2, 3, 4)}
    old = ROOT / "results" / "controlled" / f"{run.name}.membership.json"
    if old.exists():
        o = json.load(open(old))
        res["baselines"] = {k: o[k]["tpr_at_0fp"] for k in ("grad_sign", "loss_attack", "ref_model_attack", "ft_loss_attack") if k in o}
    res["seconds"] = dict(feat=float(F["seconds"]), ref=float(R["seconds"]), score=time.time() - t_start)
    dst = ROOT / "results" / "controlled" / f"{run.name}.membership2.json"
    json.dump(res, open(dst, "w"), indent=1, default=lambda o: o.item())
    for k, v in res.items():
        if k not in ("variants", "split_half"):
            print(k, v)
    for n, v in table.items():
        print("%-42s TPR@0FP %.4f  elig %.4f  @1FP %.4f  thr %8.3f  out-competing %d  half-mean %s" % (
            n, v["tpr_at_0fp"], v["tpr_at_0fp_eligible"], v["tpr_at_1fp"], v["threshold"], v["outside_competing"],
            ("%.4f" % res["split_half"]["per_variant_half_mean"][n]) if n in per_var else "-"))
    print("split-half", {k: v for k, v in res["split_half"].items() if k != "per_variant_half_mean"})


if __name__ == "__main__":
    {"feat": feat, "ref": ref, "score": score}[sys.argv[1]](sys.argv[2])
