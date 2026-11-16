"""Minimal independent ranged reader for safetensors on the HF hub (no project imports)."""
import json, struct, time, os, pathlib, requests, numpy as np
from concurrent.futures import ThreadPoolExecutor
S = requests.Session()
UA = {"User-Agent": "python-requests"}
HC = pathlib.Path(__file__).resolve().parents[2] / "results" / "reflection_check" / "hdr"
def url(repo, rev, f):
    rev = full(repo, rev)
    return f"https://huggingface.co/{repo}/resolve/{rev or 'main'}/{f}"
def rng(u, a, b):
    for k in range(10):
        try:
            r = requests.get(u, headers={**UA, "Range": f"bytes={a}-{b}"}, timeout=300)
            if r.status_code == 206 and len(r.content) == b - a + 1: return r.content
            if r.status_code in (401, 403, 404): raise RuntimeError(f"{r.status_code} {u}")
        except requests.RequestException: pass
        time.sleep(min(60, 2 ** k))
    raise RuntimeError("range failed " + u)
def api(path):
    for k in range(8):
        r = requests.get("https://huggingface.co/api/" + path, headers=UA, timeout=60)
        if r.status_code == 200: return r.json()
        if r.status_code in (401, 403, 404): raise RuntimeError(f"{r.status_code} {path}")
        time.sleep(2 ** k)
    raise RuntimeError(path)
_full = {}
def full(repo, rev):
    """expand a short commit prefix to the full sha via the commit list"""
    if rev is None or len(rev) == 40: return rev
    if repo not in _full: _full[repo] = [c["id"] for c in api(f"models/{repo}/commits/main?limit=1000")]
    m = [c for c in _full[repo] if c.startswith(rev)]; assert len(m) == 1, (rev, m); return m[0]
def files(repo, rev=None):
    rev = full(repo, rev)
    i = api(f"models/{repo}" + (f"/revision/{rev}" if rev else "") + "?blobs=true")
    return {s["rfilename"]: s for s in i["siblings"]}
def header(repo, rev, f):
    HC.mkdir(parents=True, exist_ok=True)
    p = HC / (f"{repo}@{rev or 'main'}@{f}".replace("/", "__") + ".json")
    if p.exists(): return json.load(open(p))
    u = url(repo, rev, f)
    n = struct.unpack("<Q", rng(u, 0, 7))[0]
    h = json.loads(rng(u, 8, 8 + n - 1)); h.pop("__metadata__", None)
    for v in h.values(): v["base"] = 8 + n
    json.dump(h, open(p, "w")); return h
def index(repo, rev=None):
    """tensor name -> (file, dtype, shape, abs_start, abs_end)"""
    fs = sorted(k for k in files(repo, rev) if k.endswith(".safetensors"))
    with ThreadPoolExecutor(16) as ex: hs = list(ex.map(lambda f: header(repo, rev, f), fs))
    out = {}
    for f, h in zip(fs, hs):
        for name, v in h.items():
            out[name] = (f, v["dtype"], v["shape"], v["base"] + v["data_offsets"][0], v["base"] + v["data_offsets"][1])
    return out
ISZ = {"F32": 4, "BF16": 2, "F16": 2}
def read(repo, rev, ent, elem_start=0, elem_count=None, chunk=32 << 20, workers=8):
    """raw little-endian bytes of elements [elem_start, elem_start+elem_count) as uint32 (F32) or uint16 (BF16)"""
    f, dt, shape, a, b = ent
    sz = ISZ[dt]; n = (b - a) // sz
    if elem_count is None: elem_count = n - elem_start
    elem_count = min(elem_count, n - elem_start)
    s = a + elem_start * sz; e = s + elem_count * sz
    u = url(repo, rev, f)
    parts = [(x, min(x + chunk, e) - 1) for x in range(s, e, chunk)]
    if len(parts) == 1: buf = rng(u, *parts[0])
    else:
        with ThreadPoolExecutor(workers) as ex: buf = b"".join(ex.map(lambda p: rng(u, *p), parts))
    return np.frombuffer(buf, dtype="<u4" if sz == 4 else "<u2")
def bits32(x, dt):
    """bit pattern as fp32: bf16 upcast is exact = bits << 16"""
    if dt == "F32": return x
    assert dt == "BF16", dt
    return x.astype(np.uint32) << 16
def f32(x, dt): return bits32(x, dt).view(np.float32)
