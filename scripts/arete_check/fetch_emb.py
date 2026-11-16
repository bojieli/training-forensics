#!/usr/bin/env python3
"""Ranged read of single tensors from HF safetensors shards. usage: fetch_emb.py repo shard tensor out.npy [rev]"""
import json, struct, sys, urllib.request, numpy as np
repo, shard, name, out = sys.argv[1:5]
rev = sys.argv[5] if len(sys.argv) > 5 else "main"
url = f"https://huggingface.co/{repo}/resolve/{rev}/{shard}"
def rng(a, b):
    for k in range(5):
        try:
            rq = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b-1}", "User-Agent": "python-urllib"})
            return urllib.request.urlopen(rq, timeout=600).read()
        except Exception as e:
            err = e
    raise err
n = struct.unpack("<Q", rng(0, 8))[0]
hdr = json.loads(rng(8, 8 + n))
if name == "LIST":
    for k, v in hdr.items():
        if k != "__metadata__": print(k, v["dtype"], v["shape"])
    print(hdr.get("__metadata__")); sys.exit()
t = hdr[name]
a, b = t["data_offsets"]
print(repo, name, t["dtype"], t["shape"], "bytes", b - a, "meta", hdr.get("__metadata__"), flush=True)
buf = bytearray()
CH = 256 << 20
for s in range(a, b, CH):
    buf += rng(8 + n + s, 8 + n + min(b, s + CH))
assert len(buf) == b - a
dt = {"BF16": np.uint16, "F16": np.uint16, "F32": np.uint32}[t["dtype"]]
np.save(out, np.frombuffer(bytes(buf), dtype=dt).reshape(t["shape"]))
print("saved", out)
