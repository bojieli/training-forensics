#!/usr/bin/env python3
"""Recompute the sweep fit score (fit_decay) from cached embeddings and compare with stored records."""
import glob, json, os, pathlib, random, sys
import torch
sys.path.insert(0, "scripts/roster"); sys.path.insert(0, ".")
import sweep  # noqa
CACHE = pathlib.Path(os.environ.get("CKPTF_CACHE", pathlib.Path.home() / ".cache/trainforensics/emb"))
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

def load(repo, shape, dt):
    need = shape[0] * shape[1] * torch.finfo(dt).bits // 8
    for f in glob.glob(str(CACHE / (repo.replace("/", "__") + "@*"))):
        if ("embed" in f or "wte" in f or "word_emb" in f or "tok_emb" in f) and "head" not in f.split("__")[-2] and os.path.getsize(f) == need:
            return torch.frombuffer(bytearray(open(f, "rb").read()), dtype=dt).view(*shape), f
    return None, None

recs = {}
for fn in sorted(glob.glob("results/roster/sweep*.jsonl")):
    for l in open(fn):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("status") == "ok" and not r.get("tied"):
            recs.setdefault(r["key"], (fn, r))
keys = [k for k in recs if "V4-Flash-Base@ -> deepseek-ai/DeepSeek-V4-Flash" in k] + ([k for k in json.load(open(os.environ["KEYS"])) if k in recs] if os.environ.get("KEYS") else [])
others = [k for k in recs if k not in keys]
random.seed(0); random.shuffle(others)
out = []
for k in keys + others:
    if k not in keys and len([o for o in out if o["key"] not in keys]) >= int(sys.argv[1] if len(sys.argv) > 1 else 10): break
    fn, r = recs[k]
    dt0, dt1 = DT.get(r["dtype_parent"]), DT.get(r["dtype_child"])
    if dt0 is None or dt1 is None or r["shape_parent"] != r["shape_child"]: continue
    A, fa = load(r["parent"], r["shape_parent"], dt0); B, fb = load(r["child"], r["shape_child"], dt1)
    if A is None or B is None: continue
    P, s, Pls, _ = sweep.fit_decay(A, B, dt1)
    ident = int((A.to(torch.float32).to(dt1) == B).all(1).sum())
    o = dict(key=k, file=fn, stored_P=r["embed"]["P"], stored_fit=r["embed"]["P_exact_frac"], P=P, fit=s,
             stored_identical=r["embed"]["identical"], identical=ident, parent_file=os.path.basename(fa), child_file=os.path.basename(fb))
    print(json.dumps(o), flush=True); out.append(o)
json.dump(out, open(os.environ.get("OUT", "results/frontier/recheck_fit.json"), "w"), indent=1)
