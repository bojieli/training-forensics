#!/usr/bin/env python3
"""Packed sequences vs the last-token rule.

Without packing, a document's final token is the last input of its sequence and
gets no gradient, so the published-data audits drop final positions. With
EOS-joined packing the same token is followed by EOS (or the next document) and
does get a gradient; only each chunk's last token does not.

For a packed run this counts the tokens that occur in the claimed documents only
in final position and checks whether they were updated. "Updated" is the chunk-level
truth, which the reader reproduces exactly (score_stack tokens_replay fp = fn = 0).

  python3 scripts/stacks/pack_lasttoken.py runs/stacks/t13_pack
"""
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics.truth import Run  # noqa: E402

run = pathlib.Path(sys.argv[1])
tr = json.load(open(run / "truth.json"))
toks = {int(k): v for k, v in json.load(open(run / "tokens.json")).items()}
a = tr["args"]
n_ds = sum(1 for _ in open(ROOT / a["data_jsonl"]))
idx = list(range(n_ds))
random.Random(a["pool_seed"]).shuffle(idx)
claimed = idx[: a["n_docs"]]
missing = sum(d not in toks for d in claimed)
assert missing == 0, f"{missing} claimed docs not reproduced"
inner, final = set(), set()
for d in claimed:
    ids = toks[d]
    inner.update(ids[:-1])
    final.add(ids[-1])
R = Run(run)
trained = R.seen_mask()
fo = sorted(t for t in final - inner if t < R.V)
eos = int(tr["stack"].get("eos_id", 0)) if isinstance(tr.get("stack"), dict) else 0
out = dict(claimed_docs=len(claimed),
           tokens_only_in_final_position=len(fo),
           of_which_updated=int(sum(bool(trained[t]) for t in fo)),
           note="an unpacked audit (last position dropped) would call these tokens untrained; packed, they are trained",
           eos_token_updated=bool(trained[eos]))
print(json.dumps(out))
json.dump(out, open(ROOT / "results" / "stacks" / "pack_lasttoken.json", "w"), indent=1)
