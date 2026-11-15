import sys, json, numpy as np, torch
sys.path.insert(0, ".")
from trainforensics.diff import fit_decay_16, moved_rows_16
from trainforensics.truth import Run
res = {}
for run in sys.argv[1:]:
    R = Run(run); seen = R.seen_mask(); W0 = R.emb["W0"]
    for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        W1q = R.emb["W1_" + name]
        P, frac = fit_decay_16(W0, W1q, dt)
        nd = moved_rows_16(W0, W1q, dt, P)
        # never-touched rows: a few coordinates land one 16-bit step off (the
        # sequential fp32 decay drifts by up to ~N/2 fp32 ulp before rounding);
        # moved rows differ in many. Split the per-row mismatch count where it
        # separates (largest gap in log(1+count) between the two groups).
        x = np.log1p(nd.astype(np.float64))
        xs = np.sort(np.unique(x))
        if xs[-1] > 0:
            gaps = np.diff(xs); k = int(np.argmax(gaps)); cut = (xs[k] + xs[k + 1]) / 2
        else:
            cut = 0.5
        thr = int(np.expm1(cut))
        # drift floor: most rows are never touched, so the lower half of rows
        # estimates the per-coordinate drift-mismatch rate p; require counts
        # above a ~1e-6 binomial tail of Binomial(d, p)
        d = W0.shape[1]
        lower = nd[nd <= np.median(nd)]
        lam = float(lower.mean()) if len(lower) else 0.0
        binom_thr = int(np.ceil(lam + 5 * np.sqrt(max(lam, 1e-9)) + 3))
        thr = max(thr, binom_thr)
        moved = nd > thr
        r = dict(P=P, P_fit_frac=frac, mismatch_thr=thr, unseen_mismatch_max=int(nd[~seen].max()), seen_mismatch_min=int(nd[seen].min()), seen=int(seen.sum()), moved=int(moved.sum()), tp=int((moved & seen).sum()),
                 fp=int((moved & ~seen).sum()), fn=int((seen & ~moved).sum()))
        res[f"{run.split('/')[-1]}.{name}"] = r
        print(run.split('/')[-1], name, r, flush=True)
json.dump(res, open("results/controlled/tokens_16bit.json", "w"), indent=1)
