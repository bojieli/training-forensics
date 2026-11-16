#!/usr/bin/env python3
"""Step 3: the optimizer state of the run against the released and base embeddings. -> out/optim.json"""
import json, numpy as np, torch
from common import D, save, V, H
import pins

ld = lambda n: np.concatenate([np.fromfile(f"{D}optim/r{r}_{n}.bin", dtype=np.float32) for r in range(pins.N_RANKS)]).reshape(V, H)
limo = np.fromfile(D + "limo.bin", dtype=np.uint16).reshape(V, H); base = np.fromfile(D + "q32i.bin", dtype=np.uint16).reshape(V, H)
steps = sorted({json.load(open(f"{D}optim/r{r}.done"))["adam_step"] for r in range(pins.N_RANKS)})
w = ld("w")
wb = torch.from_numpy(w).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
rows_round = int((wb == limo).all(1).sum()); del wb
basef = torch.from_numpy(base.view(np.int16)).view(torch.bfloat16).float().numpy()
dw = w - basef
w_changed = (dw != 0).any(1)
v = ld("v"); v_nz = (v != 0).any(1)
m = ld("m"); m_nz = (m != 0).any(1)
ch = np.load(D + "changed_mask.npy")
np.save(D + "v_nz.npy", v_nz)
vs = v.sum(1); ms = (m.astype(np.float64) ** 2).sum(1)
np.save(D + "row_stats.npy", np.stack([vs, ms, np.abs(dw).mean(1), np.sqrt((dw.astype(np.float64) ** 2).sum(1)), (v != 0).mean(1)], 1))
frac = (v[v_nz] != 0).mean(1)
out = {
    "source": f"{pins.LIMO}@{pins.LIMO_OPTIM_REV} {pins.OPTIM_FILE} (ranks 0-{pins.N_RANKS - 1}), embedding slice",
    "adam_step_values_across_ranks": steps,
    "fp32_master_rounds_to_release_rows": rows_round,
    "rows_master_ne_base": int(w_changed.sum()),
    "rows_exp_avg_sq_nonzero": int(v_nz.sum()),
    "rows_exp_avg_nonzero": int(m_nz.sum()),
    "rows_bf16_changed": int(ch.sum()),
    "v_nz_and_changed": int((v_nz & ch).sum()), "v_nz_not_changed": int((v_nz & ~ch).sum()), "changed_not_v_nz": int((ch & ~v_nz).sum()),
    "equals_changed_rows": bool((v_nz == ch).all()), "master_changed_equals_v_nz": bool((w_changed == v_nz).all()),
    "max_abs_dw_where_v_zero": float(np.abs(dw[~v_nz]).max()),
    "share_nonzero_elements_in_v_nz_rows": {"min": float(frac.min()), "median": float(np.median(frac))},
}
print(json.dumps(out, indent=1), flush=True)
save("optim", out)
