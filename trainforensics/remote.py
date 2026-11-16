"""Read individual tensors (or row ranges) from safetensors files on the
Hugging Face Hub with HTTP Range requests, without downloading checkpoints."""
from __future__ import annotations

import json
import os
import pathlib
import struct
import functools

import requests
import torch
from huggingface_hub import hf_hub_url, get_token

_DT = {
    "F32": (torch.float32, 4), "F16": (torch.float16, 2), "BF16": (torch.bfloat16, 2),
    "F64": (torch.float64, 8), "I64": (torch.int64, 8), "I32": (torch.int32, 4),
    "U8": (torch.uint8, 1), "I8": (torch.int8, 1),
    "F8_E4M3": (torch.float8_e4m3fn, 1), "F8_E5M2": (torch.float8_e5m2, 1),
}
CACHE = pathlib.Path(os.environ.get("CKPTF_CACHE", pathlib.Path.home() / ".cache" / "ckptforensics"))


def _headers(extra=None):
    h = {"User-Agent": "ckptforensics"}
    tok = get_token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    h.update(extra or {})
    return h


def _get(url: str, headers=None, timeout=300, tries=12):
    """requests.get with backoff on HF rate limiting (429) and transient 5xx."""
    import time
    r = None
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        except requests.RequestException:
            time.sleep(min(60, 2 ** k))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            ra = r.headers.get("Retry-After")
            time.sleep(float(ra) if ra and ra.isdigit() else min(120, 5 * 2 ** k))
            continue
        return r
    return r


def _range(url: str, start: int, end_incl: int) -> bytes:
    for _ in range(5):
        r = _get(url, headers=_headers({"Range": f"bytes={start}-{end_incl}"}))
        if r is None:
            continue
        if r.status_code in (200, 206):
            data = r.content
            if r.status_code == 200:  # server ignored Range
                data = data[start:end_incl + 1]
            if len(data) == end_incl - start + 1:
                return data
    raise IOError(f"range fetch failed: {url} {start}-{end_incl} ({r.status_code})")


def cache_prefix(repo: str) -> str:
    return f"{repo}@".replace("/", "__")


class RemoteSafetensors:
    def __init__(self, repo: str, filename: str, revision: str | None = None):
        self.repo, self.filename, self.revision = repo, filename, revision
        self.url = hf_hub_url(repo, filename, revision=revision)
        n = struct.unpack("<Q", _range(self.url, 0, 7))[0]
        self.header = json.loads(_range(self.url, 8, 8 + n - 1))
        self.header.pop("__metadata__", None)
        self.data_start = 8 + n

    def keys(self):
        return list(self.header)

    def info(self, name):
        h = self.header[name]
        return h["dtype"], h["shape"]

    def get(self, name: str, rows: tuple[int, int] | None = None) -> torch.Tensor:
        """Whole tensor, or rows [r0, r1) of its first dimension."""
        h = self.header[name]
        dt, size = _DT[h["dtype"]]
        shape = list(h["shape"])
        b0, b1 = h["data_offsets"]
        if rows is not None:
            row_bytes = size
            for d in shape[1:]:
                row_bytes *= d
            r0, r1 = rows
            b0, b1 = b0 + r0 * row_bytes, b0 + r1 * row_bytes
            shape[0] = r1 - r0
        key = f"{self.repo}@{self.revision}/{self.filename}/{name}/{b0}-{b1}".replace("/", "__")
        cp = CACHE / key
        if cp.exists():
            data = cp.read_bytes()
        else:
            data = _range(self.url, self.data_start + b0, self.data_start + b1 - 1)
            CACHE.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(data)
        return torch.frombuffer(bytearray(data), dtype=dt).reshape(shape)


def _pin(repo: str, revision: str | None) -> str | None:
    """With TF_PINNED=1, read the revision pinned in revisions.json when the caller gives none.
    Off by default: the cache key contains the revision, so switching would re-download."""
    if revision is None and os.environ.get("TF_PINNED") == "1":
        from . import paths
        return paths.revisions().get(repo)
    return revision


@functools.lru_cache(maxsize=None)
def index(repo: str, revision: str | None = None) -> dict[str, str]:
    """tensor name -> safetensors filename for a (possibly sharded) repo."""
    revision = _pin(repo, revision)
    for idx_name in ("model.safetensors.index.json", "consolidated.safetensors.index.json",
                     "pytorch_model.bin.index.json"):
        url = hf_hub_url(repo, idx_name, revision=revision)
        r = _get(url, headers=_headers(), timeout=60)
        if r is not None and r.status_code == 200:
            return r.json()["weight_map"]
    for single in ("model.safetensors", "consolidated.safetensors", "pytorch_model.bin"):
        try:
            st = open_file(repo, single, revision)
            return {k: single for k in st.keys()}
        except Exception:
            continue
    raise IOError(f"no safetensors or pytorch_model.bin found in {repo}")


@functools.lru_cache(maxsize=None)
def open_file(repo: str, filename: str, revision: str | None = None):
    revision = _pin(repo, revision)
    if filename.endswith(".bin"):
        return RemoteTorchBin(repo, filename, revision)
    return RemoteSafetensors(repo, filename, revision)


def tensor(repo: str, name: str, revision: str | None = None, rows=None) -> torch.Tensor:
    return open_file(repo, index(repo, revision)[name], revision).get(name, rows)


def tensor_rows_sample(repo: str, name: str, revision: str | None = None, tail: int = 4096,
                       head: int = 1024, blocks: int = 8, block: int = 512, seed: int = 0):
    """A cheap subset of rows: the last `tail` rows (padding / reserved tokens
    usually live there), the first `head` rows (special tokens) and a few
    random blocks. Returns (rows tensor, original row indices, total rows)."""
    _, shape = tensor_info(repo, name, revision)
    V = shape[0]
    spans = [(0, min(head, V)), (max(0, V - tail), V)]
    g = torch.Generator().manual_seed(seed)
    for _ in range(blocks):
        a = int(torch.randint(0, max(1, V - block), (1,), generator=g))
        spans.append((a, min(V, a + block)))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    parts = [tensor(repo, name, revision, rows=(a, b)) for a, b in merged]
    ids = torch.cat([torch.arange(a, b) for a, b in merged])
    return torch.cat(parts), ids, V


def tensor_info(repo: str, name: str, revision: str | None = None):
    return open_file(repo, index(repo, revision)[name], revision).info(name)


# ---------------------------------------------------------------------------
# Legacy PyTorch checkpoints (pytorch_model*.bin): a zip of uncompressed
# entries. We range-read the zip directory and the pickled metadata, then
# fetch only the storage bytes of the requested tensor -- the pickle is
# interpreted with a restricted unpickler that never executes code.

import pickle
import zipfile
import io

_STORAGE = {
    "FloatStorage": (torch.float32, 4), "HalfStorage": (torch.float16, 2),
    "BFloat16Storage": (torch.bfloat16, 2), "DoubleStorage": (torch.float64, 8),
    "LongStorage": (torch.int64, 8), "IntStorage": (torch.int32, 4),
    "ByteStorage": (torch.uint8, 1), "BoolStorage": (torch.bool, 1),
}


class _Meta:
    def __init__(self, key, dtype, size, offset, shape, stride):
        self.key, self.dtype, self.size = key, dtype, size
        self.offset, self.shape, self.stride = offset, list(shape), list(stride)


class _StorageRef:
    def __init__(self, key, dtype, size):
        self.key, self.dtype, self.size = key, dtype, size


def _rebuild_tensor_v2(storage, offset, shape, stride, *args):
    return _Meta(storage.key, storage.dtype, storage.size, offset, shape, stride)


class _SafeUnpickler(pickle.Unpickler):
    ALLOWED = {("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"),
               ("torch", "Size")} | {("torch", n) for n in _STORAGE}

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if module == "torch" and name in _STORAGE:
            return name
        if (module, name) == ("collections", "OrderedDict"):
            import collections
            return collections.OrderedDict
        if (module, name) == ("torch", "Size"):
            return tuple
        raise pickle.UnpicklingError(f"refusing {module}.{name}")

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, st, key, _, numel = pid
        st = st if isinstance(st, str) else getattr(st, "__name__", str(st))
        return _StorageRef(key, _STORAGE[st][0], numel)


class _RangeFile(io.RawIOBase):
    """Seekable file-like view of a remote file, for zipfile's directory parsing."""

    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self._cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = _range(self.url, self.pos, self.pos + n - 1)
        b[:n] = data
        self.pos += n
        return n


class RemoteTorchBin:
    def __init__(self, repo, filename, revision=None):
        self.repo, self.filename, self.revision = repo, filename, revision
        self.url = hf_hub_url(repo, filename, revision=revision)
        r = requests.head(self.url, headers=_headers(), allow_redirects=True, timeout=60)
        size = int(r.headers["Content-Length"])
        self.raw = _RangeFile(self.url, size)
        self.zf = zipfile.ZipFile(io.BufferedReader(self.raw, buffer_size=1 << 16))
        names = self.zf.namelist()
        pkl = next(n for n in names if n.endswith("data.pkl"))
        self.prefix = pkl[: -len("data.pkl")]
        meta = _SafeUnpickler(io.BytesIO(self.zf.read(pkl))).load()
        self.header = {k: v for k, v in meta.items() if isinstance(v, _Meta)}

    def keys(self):
        return list(self.header)

    def info(self, name):
        m = self.header[name]
        rev = {v[0]: k for k, v in _DT.items()}
        return rev.get(m.dtype, str(m.dtype)), m.shape

    def get(self, name, rows=None):
        m = self.header[name]
        zi = self.zf.getinfo(f"{self.prefix}data/{m.key}")
        assert zi.compress_type == zipfile.ZIP_STORED
        # local header: 30 bytes + name + extra
        lh = _range(self.url, zi.header_offset, zi.header_offset + 29)
        nlen, xlen = struct.unpack("<HH", lh[26:30])
        data_off = zi.header_offset + 30 + nlen + xlen
        esz = torch.empty(0, dtype=m.dtype).element_size()
        numel = 1
        for s in m.shape:
            numel *= s
        # contiguous row-major tensors only (true for embeddings / heads)
        exp = [1] * len(m.shape)
        for i in range(len(m.shape) - 2, -1, -1):
            exp[i] = exp[i + 1] * m.shape[i + 1]
        assert m.stride == exp, f"non-contiguous {name}: {m.stride}"
        r0, r1 = rows if rows else (0, m.shape[0])
        row = numel // m.shape[0]
        b0 = data_off + (m.offset + r0 * row) * esz
        b1 = data_off + (m.offset + r1 * row) * esz
        key = f"{self.repo}@{self.revision}/{self.filename}/{name}/{b0}-{b1}".replace("/", "__")
        cp = CACHE / key
        if cp.exists():
            data = cp.read_bytes()
        else:
            data = _range(self.url, b0, b1 - 1)
            CACHE.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(data)
        return torch.frombuffer(bytearray(data), dtype=m.dtype).reshape([r1 - r0] + m.shape[1:])
