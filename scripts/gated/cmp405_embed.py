"""Bitwise compare the 405B substitutes' embedding against a copy whose file is sha256-identical to Meta's official release."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import json, sys, os, pathlib
sys.path.insert(0, str(paths.REPO))
from trainforensics import remote
import torch
PAIRS = [("NousResearch/Meta-Llama-3.1-405B-FP8", "chutesai/Llama-3.1-405B-FP8", "meta-llama/Llama-3.1-405B"),
         ("RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8", "SicariusSicariiStuff/Llama-3.1-405B-Instruct", "meta-llama/Llama-3.1-405B-Instruct")]
N = "model.embed_tokens.weight"
out = []
for sub, ref, off in PAIRS:
    a_info = remote.tensor_info(sub, N); b_info = remote.tensor_info(ref, N)
    V = a_info["shape"][0] if isinstance(a_info, dict) else a_info[1][0]
    eq_rows = 0; eq_vals = 0; tot = 0; step = 8192
    for r0 in range(0, V, step):
        r1 = min(V, r0 + step)
        A = remote.tensor(sub, N, rows=(r0, r1)); B = remote.tensor(ref, N, rows=(r0, r1))
        assert A.dtype == B.dtype and A.shape == B.shape, (A.dtype, B.dtype, A.shape, B.shape)
        ia = A.view(torch.int16); ib = B.view(torch.int16)
        e = ia == ib
        eq_vals += int(e.sum()); tot += e.numel(); eq_rows += int(e.all(1).sum())
        print(sub, r0, eq_rows, flush=True)
    rec = dict(substitute=sub, reference=ref, reference_file_sha256_equals=off, rows=V, identical_rows=eq_rows,
               identical_values=eq_vals, values=tot, bit_identical=eq_vals == tot)
    out.append(rec); print("RESULT", json.dumps(rec), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
