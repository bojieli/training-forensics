"""Diagnostic (uses truth): how well do per-row statistics give D(s1), D(s2) for twice-touched rows."""
import sys, pathlib, numpy as np, torch
ROOT = pathlib.Path.home() / "training-forensics"
sys.path.insert(0, str(ROOT))
from trainforensics import blind
from trainforensics.truth import Run

run = Run(ROOT / sys.argv[1])
D = blind.moves_curve(run.lr, run.t["args"]["wd"])
W0, W1 = run.emb["W0"].double(), run.emb["W1"].double()
ts = run.token_steps()
rows = [(t, s) for t, s in ts.items() if len(s) == 2]
toks = torch.tensor([t for t, _ in rows])
d = (W1[toks] - W0[toks]).abs()
qs = torch.tensor([0.0, 0.005, 0.01, 0.02, 0.5, 0.98, 0.99, 0.995, 1.0], dtype=torch.float64)
Q = torch.quantile(d, qs, dim=1).T.numpy()
s1 = np.array([s[0] for _, s in rows]); s2 = np.array([s[1] for _, s in rows])
D1, D2 = D[s1], D[s2]
gap = s2 - s1
print("rows", len(rows), "D1>D2 frac", np.mean(D1 > D2), "gap median", np.median(gap))
def rep(name, est, true):
    e = est / true - 1
    print(f"{name:28s} med|err| {np.median(np.abs(e)):.4f}  within1% {np.mean(np.abs(e) < 0.01):.3f}  within0.3% {np.mean(np.abs(e) < 0.003):.3f}")
rep("median -> D1", Q[:, 4], D1)
for i, q in ((8, "max"), (7, "q995"), (6, "q99"), (5, "q98")):
    rep(f"{q} -> D1+D2", Q[:, i], D1 + D2)
for i, q in ((0, "min"), (1, "q005"), (2, "q01"), (3, "q02")):
    rep(f"{q} -> |D1-D2|", Q[:, i], np.abs(D1 - D2))
rep("(max-min)/2 -> D2", (Q[:, 8] - Q[:, 0]) / 2, D2)
rep("max - median -> D2", Q[:, 8] - Q[:, 4], D2)
rep("q995 - median -> D2", Q[:, 7] - Q[:, 4], D2)
for lo, hi in ((0, 100), (100, 300), (300, 600), (600, 10**9)):
    m = (gap >= lo) & (gap < hi)
    if m.sum():
        e1 = np.abs(Q[m, 4] / D1[m] - 1); e2 = np.abs((Q[m, 8] - Q[m, 4]) / D2[m] - 1)
        print(f"gap [{lo},{hi}) n={m.sum()} median within1% {np.mean(e1<0.01):.3f}  max-med within1% {np.mean(e2<0.01):.3f} within2% {np.mean(e2<0.02):.3f}")
