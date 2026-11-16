#!/usr/bin/env python3
"""AM-Thinking-v1's model.safetensors.index.json points at the wrong shards; locate the
embedding and head by scanning shard headers, then add its two pairs to pairs_large.json."""
import json, pathlib, struct, sys, requests
from huggingface_hub import hf_hub_url
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "roster"))
import build_pairs as bp  # noqa
REPO = "a-m-team/AM-Thinking-v1"
found = {}
for i in range(1, 15):
    f = f"model-{i:05d}-of-00014.safetensors"
    u = hf_hub_url(REPO, f)
    n = struct.unpack("<Q", requests.get(u, headers={"Range": "bytes=0-7"}, timeout=60).content[:8])[0]
    h = json.loads(requests.get(u, headers={"Range": f"bytes=8-{7+n}"}, timeout=60).content)
    for k, v in h.items():
        if k in ("model.embed_tokens.weight", "lm_head.weight"):
            s, e = v["data_offsets"]
            found["embed" if "embed" in k else "head"] = dict(tensor=k, file=f, dtype=v["dtype"], shape=v["shape"],
                                                              offset=8 + n + s, bytes=e - s)
print(found)
side = bp.side(REPO)
for role in ("embed", "head"):
    ref = side.get(role) or {}
    side[role] = {**{k: v for k, v in ref.items() if k != "error"}, **found[role]}
side["tied"] = False
side["note"] = "index.json lists wrong shards; tensors located by header scan"
P = ROOT / "results" / "large" / "pairs_large.json"
pairs = json.load(open(P))
pairs = [p for p in pairs if p["child"]["repo"] != REPO]
for par in ("Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct"):
    a = bp.side(par)
    cmp_ = bp.compare(a, side)
    pairs.append(dict(key=f"{par}@ -> {REPO}", parent=a, child=side, src=["large:base" if par.endswith("32B") else "large:instruct"],
                      compat=cmp_, tied=False, relation="base" if par.endswith("32B") else "instruct", rank=0,
                      bytes=a["embed"]["bytes"] + side["embed"]["bytes"]))
json.dump(pairs, open(P, "w"), indent=1, default=str)
print(len(pairs), "pairs")
