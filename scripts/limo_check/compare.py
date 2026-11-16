#!/usr/bin/env python3
"""Step 1: which candidate parent the LIMO embedding descends from, and the changed-row masks
(LIMO v1, LIMO-v2 and s1-32B against Qwen2.5-32B-Instruct) used by the later steps. -> out/compare.json"""
import numpy as np, torch
from common import D, save, V, H

L = lambda t: np.fromfile(D + t + ".bin", dtype=np.uint16).reshape(V, H)
limo = L("limo")
res = {"parents_of_limo": {}}
for b, name in (("q32i", "Qwen2.5-32B-Instruct"), ("q32b", "Qwen2.5-32B"), ("qc32i", "Qwen2.5-Coder-32B-Instruct"), ("qc32b", "Qwen2.5-Coder-32B")):
    rows = (L(b) != limo).any(1)
    res["parents_of_limo"][name] = {"rows_identical": int((~rows).sum()), "rows_changed": int(rows.sum())}
    if b == "q32i": np.save(D + "changed_mask.npy", rows)
    print(name, res["parents_of_limo"][name], flush=True)
base = L("q32i")
r1 = np.load(D + "changed_mask.npy")
r2 = (L("limo2") != base).any(1); np.save(D + "changed_mask_v2.npy", r2)
res["limo2_vs_q32i"] = {"rows_changed": int(r2.sum()), "rows_identical": int((~r2).sum())}
res["v1_v2_overlap"] = {"both": int((r1 & r2).sum()), "v1_only": int((r1 & ~r2).sum()), "v2_only": int((~r1 & r2).sum())}
# s1-32B is released in fp32; compare with the bf16 base value-wise
s1 = np.fromfile(D + "s1.bin", dtype=np.float32).reshape(V, H)
bf = torch.from_numpy(base.view(np.int16)).view(torch.bfloat16).float().numpy()
cs = np.zeros(V, bool)
for a in range(0, V, 8192): cs[a:a + 8192] = (s1[a:a + 8192] != bf[a:a + 8192]).any(1)
np.save(D + "changed_mask_s1.npy", cs)
res["s1_vs_q32i"] = {"rows_changed": int(cs.sum())}
res["limo_rows_changed_at_or_above_151643"] = np.nonzero(r1[151643:])[0].__add__(151643).tolist()
print(res, flush=True)
save("compare", res)
