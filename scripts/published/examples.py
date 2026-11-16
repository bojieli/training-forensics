"""Decoded examples per category of changed-but-absent tokens (excluding tokens any counted variant
explains), added to boundary_<child>.json as examples_by_category."""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "published"))
from trainforensics import remote
from compare import frac_changed, EMB, PUB
from transformers import AutoTokenizer
OUT = ROOT / "results" / "published"
base, child, data = sys.argv[1:4]
name = child.replace("/", "__")
r = json.load(open(OUT / f"{name}__{data}.json"))
bd = json.load(open(OUT / f"boundary_{name}.json"))
tok = AutoTokenizer.from_pretrained(base)
remote.CACHE = EMB; W0 = remote.tensor(base, "model.embed_tokens.weight")
remote.CACHE = PUB; W1 = remote.tensor(child, "model.embed_tokens.weight")
n = min(len(W0), len(W1)); f = frac_changed(W0[:n], W1[:n], r["P"])
allin = set()
for c in json.load(open(OUT / f"data_{data}.json"))["counts"].values():
    allin |= set(int(k) for k in c)
extra = [i for i in np.nonzero(f > 0)[0].tolist() if i not in allin and i < len(tok)]
cats = {}
for i in extra:
    s = tok.decode([i])
    if any("一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" for ch in s): k = "CJK"
    elif "�" in s or not s.strip(): k = "bytes / whitespace"
    elif any(ord(ch) > 127 for ch in s): k = "other non-ASCII"
    else: k = "ASCII"
    cats.setdefault(k, []).append(s)
rng = np.random.default_rng(0)
bd["examples_by_category"] = {k: [v[j] for j in sorted(rng.choice(len(v), min(25, len(v)), replace=False))] for k, v in cats.items()}
json.dump(bd, open(OUT / f"boundary_{name}.json", "w"), indent=1, ensure_ascii=False)
print({k: v[:12] for k, v in bd["examples_by_category"].items()})
