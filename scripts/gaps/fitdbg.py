"""Why does fit_recipe miss r12? Rebuild the single-touch observations as score.read does and
inspect the coarse grid. Truth used only for printing."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import sys, math, pathlib, numpy as np, torch
ROOT = paths.REPO
sys.path.insert(0, str(ROOT))
from trainforensics import blind
from trainforensics.diff import row_diff
from safetensors.torch import load_file

run = ROOT / sys.argv[1]
e = load_file(str(run / "emb.safetensors")); W0, W1 = e["W0"], e["W1"]
rd = row_diff(W0, W1, torch.float32); mv = np.nonzero(rd.moved)[0]
if rd.decay < 1:
    k, v, c = blind.decay_lut(W0, W1, ~rd.moved); ref, cov = blind.apply_lut(W0[mv], k, v, rd.decay)
    W0r = W0.clone(); W0r[mv] = ref
    D, T, Rr = blind.row_intercepts(W0r, W1, 1.0, mv)
else:
    D, T, Rr = blind.row_intercepts(W0, W1, rd.decay, mv)
obs = D[Rr < 3e-3]
logP = math.log(rd.decay) if rd.decay < 1 else None
np.save(f"/tmp/obs_{run.name}.npy", obs)
print("decay", rd.decay, "single rows", len(obs))
wd = ("logP", logP) if logP else 0.0
lo = np.log(obs)
for spec in sys.argv[2:]:
    kd, N, W, mr = spec.split(",")
    for sig in (6e-4, 4.8e-3):
        r = blind._eval((lo, kd, int(N), int(W), float(mr), wd, sig))
        print(spec, sig, round(r.score, 4), r.peak)
