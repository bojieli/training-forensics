#!/usr/bin/env python3
"""Size of the changes of tokens updated but absent from the data, against tokens known to be in
the data (seen once / seen often). Per token vector: fraction of its numbers changed and relative
change |W1 - P*W0| / |W0|. Also the decay-fit residual (how many never-updated vectors the fit
reproduces exactly). Writes results/published/size_<child>.json."""
import json, pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "published"))
from trainforensics import remote  # noqa
from compare import frac_changed, EMB, PUB  # noqa
OUT = ROOT / "results" / "published"


def q(x):
    x = np.asarray(x)
    return {k: float(np.quantile(x, p)) for k, p in (("p05", .05), ("p25", .25), ("median", .5), ("p75", .75), ("p95", .95))} if len(x) else {}


def main():
    base, child, data, key = sys.argv[1], sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv) > 4 else "main")
    r = json.load(open(OUT / f"{child.replace('/', '__')}__{data}.json"))
    P = r["P"]
    remote.CACHE = EMB
    W0 = remote.tensor(base, "model.embed_tokens.weight")
    remote.CACHE = PUB
    W1 = remote.tensor(child, "model.embed_tokens.weight")
    n = min(len(W0), len(W1))
    W0, W1 = W0[:n], W1[:n]
    f = frac_changed(W0, W1, P)
    pred = (W0.double() * P).to(torch.float32).to(W1.dtype).double()
    rel = ((W1.double() - pred).norm(dim=1) / W0.double().norm(dim=1).clamp_min(1e-30)).numpy()
    counts = json.load(open(OUT / f"data_{data}.json"))["counts"][key]
    cnt = np.zeros(n, np.int64)
    for k, c in counts.items():
        if int(k) < n:
            cnt[int(k)] = c
    from transformers import AutoTokenizer
    V = len(AutoTokenizer.from_pretrained(base))
    real = np.arange(n) < V
    changed = f > 0
    groups = {"updated, not in data": changed & (cnt == 0) & real,
              "in data once": (cnt == 1), "in data 2-5 times": (cnt >= 2) & (cnt <= 5),
              "in data over 100 times": cnt > 100}
    res = dict(child=child, data=data, key=key, P=P, never_updated_reproduced=int((f == 0).sum()), groups={})
    for g, sel in groups.items():
        res["groups"][g] = dict(n=int(sel.sum()), updated=int((sel & changed).sum()),
                                frac_changed=q(f[sel & changed]), rel_change=q(rel[sel & changed]))
    json.dump(res, open(OUT / f"size_{child.replace('/', '__')}.json", "w"), indent=1)
    for g, x in res["groups"].items():
        print(g, x["n"], x["updated"], "frac med", round(x["frac_changed"].get("median", 0), 3),
              "rel p05/med/p95", [round(x["rel_change"].get(k, 0), 5) for k in ("p05", "median", "p95")])


if __name__ == "__main__":
    main()
