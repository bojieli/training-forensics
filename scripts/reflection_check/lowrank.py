#!/usr/bin/env python3
"""Step 3: is (current - reference) low-rank? First ROWS rows of selected matrices; singular values of the difference."""
import json, sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import st
R = "mattshumer/Reflection-Llama-3.1-70B"; ROWS = 1024
REF = {"L3-I": "NousResearch/Meta-Llama-3-70B-Instruct", "L3.1-I": "unsloth/Meta-Llama-3.1-70B-Instruct"}
ix = {k: st.index(r) for k, r in REF.items()}; cx = st.index(R)
NAMES = [f"model.layers.{l}.{t}.weight" for l in (0, 20, 40, 60, 79) for t in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
out = {}
for name in NAMES:
    e = cx[name]; cols = e[2][1]; n = min(ROWS, e[2][0]) * cols
    W = st.f32(st.read(R, None, e, 0, n), e[1]).reshape(-1, cols).astype(np.float64)
    o = dict(shape=e[2], rows_used=W.shape[0])
    for k in REF:
        C = st.f32(st.read(REF[k], None, ix[k][name], 0, n), ix[k][name][1]).reshape(-1, cols).astype(np.float64)
        s = np.linalg.svd(W - C, compute_uv=False); en = np.cumsum(s ** 2) / (s ** 2).sum()
        o[k] = dict(rel=float(np.linalg.norm(W - C) / np.linalg.norm(C)), rank_gt_1e3=int((s > 1e-3 * s[0]).sum()), rank_gt_1e2=int((s > 1e-2 * s[0]).sum()),
                    energy_top={r: float(en[r - 1]) for r in (8, 16, 32, 64, 128, 256, 512)}, s_ratio={r: float(s[r] / s[0]) for r in (15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 511, 512)},
                    max_abs_diff=float(np.abs(W - C).max()))
    out[name] = o; print(name, json.dumps(o), flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
