"""Paths for the LIMO check. Large inputs live in $TF_CACHE/limo_check (default ~/.cache/trainforensics/limo_check);
every number that a report cites is written to results/limo_check/out/<step>.json by the script that computes it."""
import json, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = pathlib.Path(os.environ.get("TF_CACHE", pathlib.Path.home() / ".cache" / "trainforensics")) / "limo_check"
OUT = ROOT / "results" / "limo_check" / "out"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
D = str(CACHE) + "/"
V, H = 152064, 5120  # Qwen2.5-32B embedding shape


def save(name, d):
    json.dump(d, open(OUT / f"{name}.json", "w"), indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[saved out/{name}.json]", flush=True)


def load(name):
    return json.load(open(OUT / f"{name}.json"))


def gh(path):
    return D + "gh_limo/" + path
