#!/usr/bin/env python3
"""Run previously gated comparisons whose gated side has a byte-identical open mirror
(verified by LFS sha256 in results/gated/mirror_hashes_embed.json).

Writes results/roster/sweep_gated.jsonl (sweep.jsonl format + change_hist).
"""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import json, os, pathlib, sys, importlib.util
ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("CKPTF_CACHE", str(paths.cache("trainforensics") / "emb"))
sys.path.insert(0, str(ROOT / "scripts" / "roster"))
import build_pairs as bp  # noqa
spec = importlib.util.spec_from_file_location("lr", ROOT / "scripts" / "y2026_labs" / "run.py")
lr = importlib.util.module_from_spec(spec); spec.loader.exec_module(lr)
OUT = ROOT / "results" / "roster" / "sweep_gated.jsonl"
VER = {r["mirror"]: r for r in json.load(open(ROOT / "results" / "gated" / "mirror_hashes_embed.json"))}
# (official gated parent, verified open mirror, child, source lane)
TODO = [("meta-llama/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", "microsoft/UserLM-2.0", "y2026_labs"),
        ("meta-llama/Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B", "microsoft/UserLM-2.0", "y2026_labs")]
done = set()
if OUT.exists():
    done = {json.loads(l)["key"] for l in open(OUT) if json.loads(l).get("status") == "ok"}
for off, mir, ch, lane in TODO:
    v = VER[mir]
    assert v["official"] == off and v["embedding_files_identical"], (mir, v)
    a, b = bp.side(mir), bp.side(ch)
    cmp_ = bp.compare(a, b)
    key = f"{off}@ -> {ch}"
    if key in done: continue
    p = dict(key=key, parent=a, child=b, src=[f"gated:{lane}"], compat=cmp_, tied=bool(a.get("tied") or b.get("tied")),
             relation="card base (gated; byte-identical open mirror)", rank=0, bytes=a["embed"]["bytes"] + b["embed"]["bytes"])
    if not (cmp_.get("embed") or {}).get("compatible"):
        rec = dict(key=key, parent=off, child=ch, status="not comparable", why=str(cmp_.get("embed")))
    else:
        rec = lr.work(p, 0.0); rec.pop("_new_cache", None)
    rec["official_parent"] = off; rec["mirror_used"] = mir; rec["mirror_verified_sha256"] = True
    with open(OUT, "a") as f: f.write(json.dumps(rec) + "\n")
    e = rec.get("embed", {})
    print(rec.get("status"), key, "unmoved", e.get("unmoved"), "P", e.get("P"), "fit", e.get("P_exact_frac"), "hist", rec.get("change_hist", {}).get("frac"), flush=True)
