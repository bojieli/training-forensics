"""Parallel ranged fetch of one tensor into the trainforensics.remote cache layout."""
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
from trainforensics import paths  # noqa: E402
import sys, pathlib
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, ".")
from trainforensics import remote
remote.CACHE = paths.cache("tf_published")
NAME = "model.embed_tokens.weight"
CH = 64 << 20
for repo in sys.argv[1:]:
    fn = remote.index(repo)[NAME]
    st = remote.open_file(repo, fn)
    b0, b1 = st.header[NAME]["data_offsets"]
    key = f"{repo}@None/{fn}/{NAME}/{b0}-{b1}".replace("/", "__")
    cp = remote.CACHE / key
    if cp.exists():
        print(repo, "cached", flush=True); continue
    spans = [(a, min(b1, a + CH)) for a in range(b0, b1, CH)]
    with ThreadPoolExecutor(16) as ex:
        parts = list(ex.map(lambda s: remote._range(st.url, st.data_start + s[0], st.data_start + s[1] - 1), spans))
    remote.CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_suffix(".part")
    with open(tmp, "wb") as f:
        for p in parts: f.write(p)
    tmp.rename(cp)
    print(repo, "fetched", (b1 - b0) >> 20, "MiB", flush=True)
