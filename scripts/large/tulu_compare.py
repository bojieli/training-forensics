#!/usr/bin/env python3
"""Published-data token audit for Tulu 3 405B: tokens updated between stages vs tokens in
the published data of that stage (counts from tulu_data.py).

  base (Llama-3.1-405B, bf16 embedding of the FP8 release) -> Tulu-3-405B-SFT : tulu-3-sft-mixture
  Tulu-3-405B-SFT -> Tulu-3-405B-DPO                                          : 405b preference mixture
Per token vector: numbers that differ from round(P x parent) (mine.per_vector, P from the
sweep). updated = more than 1% of numbers differ; barely = 1..1%; bit-identical = none.
Writes results/large/tulu_audit.json.
"""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tokens")); sys.path.insert(0, str(ROOT / "scripts" / "roster"))
import mine, sweep  # noqa
from transformers import AutoTokenizer

PAIRS = {p["key"]: p for p in json.load(open(ROOT / "results" / "large" / "pairs_large.json"))}
REC = {}
for l in open(ROOT / "results" / "large" / "sweep_large.jsonl"):
    r = json.loads(l)
    if r.get("status") == "ok":
        REC[r["key"]] = r
STAGES = [("NousResearch/Meta-Llama-3.1-405B-FP8@ -> allenai/Llama-3.1-Tulu-3-405B-SFT", "sft"),
          ("allenai/Llama-3.1-Tulu-3-405B-SFT@ -> allenai/Llama-3.1-Tulu-3-405B-DPO", "dpo")]
tok = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-405B-SFT")
added = set(tok.added_tokens_decoder)
out = {}
for key, which in STAGES:
    if key not in REC:
        out[which] = dict(status="sweep record missing"); continue
    p, r = PAIRS[key], REC[key]
    ce = p["compat"]["embed"]; r0, r1 = ce["common_rows"]
    W0 = sweep.fetch(p["parent"]); W1 = sweep.fetch(p["child"])
    W0, W1 = W0[r0:r1], W1[r0:r1]
    cdt = sweep.TD[ce["dtype_child"]]
    P = r["embed"]["P"]
    nd, n0, n1, cos = mine.per_vector(W0, W1.to(cdt), cdt, P)
    V, d = W0.shape
    thr = max(2, int(0.01 * d))
    counts = json.load(open(ROOT / "results" / "large" / f"tulu_{which}_counts.json"))
    res = dict(key=key, P=P, fit=r["embed"].get("P_exact_frac"), rows=V,
               bit_identical=int((nd == 0).sum()), barely=int(((nd > 0) & (nd <= thr)).sum()), updated=int((nd > thr).sum()))
    for var in ("cut", "nocut"):
        c = {int(k): v for k, v in counts[var].items()}
        present = np.zeros(V, bool)
        for k in c:
            if 0 <= k < V:
                present[k] = True
        reg = np.array([i not in added for i in range(V)])
        upd, ch = nd > thr, nd > 0
        pnc = np.nonzero(present & ~ch)[0]
        cnp = np.nonzero(upd & ~present & reg)[0]
        res[var] = dict(present=int(present.sum()), present_and_changed=int((present & ch).sum()),
                        present_not_changed=len(pnc), changed_not_present_regular=len(cnp),
                        present_not_changed_occ_hist={str(b): int(sum(1 for i in pnc if c.get(int(i), 0) == b)) for b in (1, 2, 3, 4, 5)},
                        present_not_changed_examples=[[int(i), tok.convert_ids_to_tokens(int(i)), c.get(int(i), 0)] for i in pnc[:25]],
                        changed_not_present_examples=[[int(i), tok.convert_ids_to_tokens(int(i)), round(float(nd[i] / d), 3)] for i in cnp[:25]])
    res["added_tokens"] = [[i, tok.convert_ids_to_tokens(i), "updated" if nd[i] > thr else ("barely" if nd[i] else "bit-identical")]
                           for i in sorted(added) if i < V]
    out[which] = res
    print(which, {k: v for k, v in res.items() if k in ("bit_identical", "barely", "updated")},
          {v: {k: res[v][k] for k in ("present", "present_not_changed", "changed_not_present_regular")} for v in ("cut", "nocut")}, flush=True)
json.dump(out, open(ROOT / "results" / "large" / "tulu_audit.json", "w"), indent=1, ensure_ascii=False)
