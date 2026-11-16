"""Probe: candidate pool rank of the true shape, and v2 selection, for one run."""
import json, math, os, pathlib, sys, time, warnings, functools
print = functools.partial(print, flush=True)
warnings.filterwarnings("ignore")
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "controlled"))
from safetensors.torch import load_file
from trainforensics import blind, recipe_v2
from trainforensics.diff import row_diff
import score as S

run = pathlib.Path(sys.argv[1])
t = json.load(open(run / "truth.json"))
a = t["args"]
e = load_file(str(run / "emb.safetensors")); W0, W1 = e["W0"], e["W1"]
rd = row_diff(W0, W1, torch.float32); decay = rd.decay
mv = np.nonzero(rd.moved)[0]
keys, vals, cons = blind.decay_lut(W0, W1, ~rd.moved)
ref, cov = blind.apply_lut(W0[mv], keys, vals, decay)
W0ref = W0.clone(); W0ref[mv] = ref
Dm, Tm, Rm = blind.row_intercepts(W0ref, W1, 1.0, mv)
Dq = Dm[Rm < 3e-3]
t0 = time.time()
pool = recipe_v2.candidate_pool(Dq, math.log(decay), procs=int(os.environ.get("TF_PROCS", "16")))
print("pool", len(pool), "%.0fs" % (time.time() - t0))
true = (a["sched"], a.get("warmup"))
for i, p in enumerate(pool[:60]):
    flag = " <== kind" if p["kind"] == a["sched"] else ""
    if i < 12 or flag:
        print(i, p["kind"], p["N"], p["warmup"], p["min_ratio"], "%.4g" % p["peak"], "%.3f" % p["dscore"], flag)
idx = np.nonzero(~rd.moved)[0]
g = np.random.default_rng(0)
s_idx = g.choice(idx, 48, replace=False); f_idx = idx[np.linspace(0, len(idx) - 1, 256).astype(int)]
x0s, x1s = W0[s_idx].float().flatten(), W1[s_idx].flatten()
x0f, x1f = W0[f_idx].float().flatten(), W1[f_idx].flatten()
def lr_fn(r):
    return S.hf_lr(dict(r, peak=S.snap(r["peak"])), r["N"])
def wd_of(r, lr):
    return blind.wd_from_decay(decay, lr)
t0 = time.time()
lo_ = np.log(Dq[Dq > 0])
res = recipe_v2.select(pool, lr_fn, x0s, x1s, x0f, x1f, wd_of, log=print,
    dscore_fn=lambda r: blind._eval((lo_, r["kind"], r["N"], r["warmup"], r["min_ratio"], ("logP", math.log(decay)), 6e-4)).score)
print("select %.0fs" % (time.time() - t0))
print(json.dumps({k: v for k, v in res.items() if k != "screened"}, default=float))
for s in res["screened"]:
    print("  ", s)
print("TRUE", a["sched"], 500 if a["max_steps"] < 0 else a["max_steps"], a["warmup"], a["lr"], a["wd"], a.get("min_lr_ratio"))
