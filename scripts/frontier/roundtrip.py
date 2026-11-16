#!/usr/bin/env python3
"""Format round trips that look like training (review minor 10).

A release converted through another 16-bit format and back (e.g. bf16 -> fp16 -> bf16) changes
small values (fp16 subnormals) without any training. For every recounted comparison this counts
the changed vectors (not consistent with parent x shrink) that equal the parent pushed through a
float16 or bfloat16 round trip, and stores `roundtrip_explained` / `changed` in the recount record.
comparability.classify treats a comparison whose changes are >= 99% round trip as "unchanged".
"""
import glob
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "frontier"))
from recount import DT, OUT, _cached, all_records  # noqa: E402


def main():
    recs = all_records()
    extra = ROOT / "results" / "frontier" / "negctl_pairs.json"
    n = 0
    for f in sorted(glob.glob(str(OUT / "*.json"))):
        rc = json.loads(open(f).read())
        if "roundtrip_explained" in rc and not ("--redo" in sys.argv and rc["roundtrip_explained"] > 0):
            continue
        r = recs.get(rc["key"])
        if r is None:
            continue
        W0 = _cached(r["parent"], r["shape_parent"], DT[r["dtype_parent"]])
        W1 = _cached(r["child"], r["shape_child"], DT[r["dtype_child"]])
        if W0 is None or W1 is None:
            continue
        r0 = rc.get("r0", 0)
        V = rc["rows"]
        W0, W1 = W0[r0:r0 + V], W1[r0:r0 + V]
        dC = W1.dtype
        changed = ~((W0.to(torch.float32).to(dC) == W1).all(1))
        expl = torch.zeros_like(changed)
        for mid in (torch.float16, torch.bfloat16):
            if mid == dC:
                continue
            expl |= (W0.to(torch.float32).to(mid).to(torch.float32).to(dC) == W1).all(1)
        expl &= changed & ~(W1 == 0).all(1)        # zeroed rows are edits, not a format round trip
        rc["changed_vs_parent"] = int(changed.sum())
        rc["roundtrip_explained"] = int(expl.sum())
        open(f, "w").write(json.dumps(rc))
        n += 1
        if rc["roundtrip_explained"]:
            print(rc["key"], rc["changed_vs_parent"], rc["roundtrip_explained"], flush=True)
    print("updated", n)


if __name__ == "__main__":
    main()
