"""Diagnose B false flags (uses truth)."""
import sys, math, pathlib, numpy as np
ROOT = pathlib.Path.home() / "training-forensics"
sys.path.insert(0, str(ROOT / "scripts" / "gaps")); sys.path.insert(0, str(ROOT))
import swaps
from trainforensics.truth import Run
name = sys.argv[1]
st = swaps.setup(name)
R = Run(st.run); ts = R.token_steps()
E = {}
for k, toks in st.docs.items():
    for tk in set(toks):
        E.setdefault(tk, set()).add(st.step[k])
r = swaps.analyse(name)
for f in r["false"][:8]:
    k = f["pos"]; s = st.step[k]
    for tk in set(st.docs[k]):
        if st.single[tk] and E[tk] == {s}:
            l = math.log(st.D[tk] / st.curve[s]); q = st.Q[tk, 3] / st.curve[s]
            if not (l > -3e-2 and st.Q[tk, 3] <= st.curve[s] * (1 + 2e-3) + 16 * st.ulp[tk]):
                print(k, "tok", tk, "pred", s, "true", ts.get(tk), "logD", round(l, 5), "max/curve", round(q, 5), "16ulp/curve", 16 * st.ulp[tk] / st.curve[s])
