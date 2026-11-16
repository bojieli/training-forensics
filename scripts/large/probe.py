import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import json, pathlib, sys
ROOT = paths.REPO
sys.path.insert(0, str(ROOT / "scripts" / "roster")); sys.path.insert(0, str(ROOT))
import build_pairs as bp
names = sys.argv[1:]
for n in names:
    s = bp.side(n)
    e = s.get("embed") or {}
    print(n, "| exists", s.get("exists"), "gated", s.get("gated"), "| embed", e.get("dtype"), e.get("shape"), "| tied", s.get("tied"), "| base", s.get("card_base_model"), flush=True)
