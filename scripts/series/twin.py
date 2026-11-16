#!/usr/bin/env python3
"""Are OLMo-7B-0424-hf and OLMo-7B-0724-hf intermediate checkpoints the same run? Compare whole
embedding matrices at shared step branches."""
import json, sys, pathlib
import torch
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from lib import branches, emb, ROOT
A, B = "allenai/OLMo-7B-0424-hf", "allenai/OLMo-7B-0724-hf"
ba = {n for _, n in branches(A)}; bb = {n for _, n in branches(B)}
common = sorted(ba & bb, key=lambda n: int(n.split("-")[0][4:]))
picks = [common[i] for i in sorted({0, 1, len(common) // 4, len(common) // 2, 3 * len(common) // 4, len(common) - 1})]
out = dict(common=len(common), only_0424=sorted(ba - bb)[:20], only_0724=sorted(bb - ba)[:20], n_only_0424=len(ba - bb), n_only_0724=len(bb - ba), checks=[])
for n in picks:
    x = emb(A, n, rows=(0, 2048)); y = emb(B, n, rows=(0, 2048))
    out["checks"].append(dict(branch=n, identical=bool(torch.equal(x, y)), max_abs_diff=float((x.double() - y.double()).abs().max())))
    print(out["checks"][-1], flush=True)
json.dump(out, open(ROOT / "results" / "series" / "olmo7b_0424_vs_0724.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "checks"})
