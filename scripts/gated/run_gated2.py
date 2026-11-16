#!/usr/bin/env python3
"""Run the comparisons that were blocked by gated licences (accepted 2026-11-16), with each lane's pipeline
(scripts/y2026_labs/run.py work(): roster sweep + change histogram). Appends to results/roster/sweep_gated.jsonl."""
import json, os, pathlib, sys, importlib.util, concurrent.futures as cf
ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("CKPTF_CACHE", str(pathlib.Path.home() / ".cache" / "trainforensics" / "emb"))
sys.path.insert(0, str(ROOT / "scripts" / "roster"))
import build_pairs as bp  # noqa
spec = importlib.util.spec_from_file_location("lr", ROOT / "scripts" / "y2026_labs" / "run.py")
lr = importlib.util.module_from_spec(spec); spec.loader.exec_module(lr)
OUT = ROOT / "results" / "roster" / "sweep_gated.jsonl"
A = "swiss-ai/Apertus-v1.5-70B"
TODO = [(A + "-base", A + "-SFT", "frontier_x", "base -> SFT"), (A + "-SFT", A + "-RL", "frontier_x", "SFT -> RL"),
        (A + "-base", A, "frontier_x", "base -> release"), (A + "-RL", A, "frontier_x", "RL -> release")]
T = "CohereLabs/tiny-aya-"
TODO += [(T + "base", T + x, "y2026_labs", "base -> release") for x in ("global", "earth", "fire", "water")]
TODO += [("zai-org/GLM-5.3-BF16", "deep-principle/Hydrogen-1.0", "y2026_ft", "card base -> fine-tune")]
done = set()
if OUT.exists():
    done = {json.loads(l)["key"] for l in open(OUT) if json.loads(l).get("status") == "ok"}
pairs = []
for par, ch, lane, rel in TODO:
    key = f"{par}@ -> {ch}"
    if key in done: continue
    try:
        a, b = bp.side(par), bp.side(ch)
        bad = [x["repo"] + (": gated" if x.get("gated") and not x.get("accessible", True) else ": no embedding" if not x.get("embed") else "")
               for x in (a, b) if not x.get("embed")]
        if bad:
            rec = dict(key=key, parent=par, child=ch, src=[f"gated:{lane}"], status="error", error="; ".join(bad))
            open(OUT, "a").write(json.dumps(rec) + "\n"); print("skip", key, bad, flush=True); continue
        cmp_ = bp.compare(a, b)
        if not (cmp_.get("embed") or {}).get("compatible"):
            rec = dict(key=key, parent=par, child=ch, src=[f"gated:{lane}"], status="not comparable", why=str(cmp_.get("embed"))[:300])
            open(OUT, "a").write(json.dumps(rec) + "\n"); print("not comparable", key, flush=True); continue
        pairs.append(dict(key=key, parent=a, child=b, src=[f"gated:{lane}"], compat=cmp_, tied=bool(a.get("tied") or b.get("tied")),
                          relation=rel, rank=0, bytes=a["embed"]["bytes"] + b["embed"]["bytes"]))
    except Exception as e:
        print("build error", key, type(e).__name__, str(e)[:200], flush=True)
pairs.sort(key=lambda p: p["bytes"])
with cf.ThreadPoolExecutor(3) as ex:
    futs = [ex.submit(lr.work, p, 0.0) for p in pairs]
    for fu, p in zip(futs, pairs):
        rec = fu.result(); rec.pop("_new_cache", None)
        rec["official_parent"] = p["parent"]["repo"]; rec["mirror_used"] = None
        with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
        e = rec.get("embed", {})
        print(rec.get("status"), rec["key"], "unmoved", e.get("unmoved"), "P", e.get("P"), "fit", e.get("P_exact_frac"),
              "hist", rec.get("change_hist", {}).get("frac"), rec.get("error", "")[:200], flush=True)
