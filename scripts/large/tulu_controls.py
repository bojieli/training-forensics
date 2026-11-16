#!/usr/bin/env python3
"""Tulu 3 405B published-data audit, extended (tech review M8).

  - SFT stage: the 245 tokens in the published SFT mixture whose vectors stayed
    bit-identical, listed in full with their occurrence counts;
  - "changed but not in the data" counted at ANY change (not only above the 1% bar
    that tulu_compare.py uses);
  - cross-scoring: the SFT stage against the DPO preference mixture, and the DPO stage
    against the SFT mixture (wrong-data controls).
Writes results/large/tulu_controls.json.
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
counts = {w: {int(k): v for k, v in json.load(open(ROOT / "results" / "large" / f"tulu_{w}_counts.json"))["cut"].items()}
          for w in ("sft", "dpo")}
out = {}
for key, which in STAGES:
    p, r = PAIRS[key], REC[key]
    ce = p["compat"]["embed"]; r0, r1 = ce["common_rows"]
    W0 = sweep.fetch(p["parent"])[r0:r1]; W1 = sweep.fetch(p["child"])[r0:r1]
    cdt = sweep.TD[ce["dtype_child"]]
    nd, *_ = mine.per_vector(W0, W1.to(cdt), cdt, r["embed"]["P"])
    V, d = W0.shape
    reg = np.array([i not in added for i in range(V)])
    ch = nd > 0
    res = {}
    for data in ("sft", "dpo"):
        c = counts[data]
        present = np.zeros(V, bool)
        for k in c:
            if 0 <= k < V:
                present[k] = True
        pnc = np.nonzero(present & ~ch)[0]
        cnp = np.nonzero(ch & ~present & reg)[0]
        rec = dict(present=int(present.sum()), present_not_changed=len(pnc), changed_any_not_present_regular=len(cnp),
                   changed_any_not_present_by_share=np.histogram(nd[cnp] / d, bins=[0, 1e-9, 0.01, 0.1, 1.0001])[0].tolist())
        if data == which:
            rec["present_not_changed_full"] = sorted([[int(i), tok.convert_ids_to_tokens(int(i)), int(c.get(int(i), 0))] for i in pnc],
                                                     key=lambda x: -x[2])
            rec["present_not_changed_occurring_over_5"] = int(sum(1 for i in pnc if c.get(int(i), 0) > 5))
        res[f"vs_{data}_mixture"] = rec
    out[which] = res
    print(which, {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in res.items()}, flush=True)
json.dump(out, open(ROOT / "results" / "large" / "tulu_controls.json", "w"), indent=1, ensure_ascii=False)
