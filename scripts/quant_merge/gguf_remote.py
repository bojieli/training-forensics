"""Read one tensor from a GGUF file on the Hugging Face Hub with HTTP Range requests.

Only the header and the requested tensor's bytes are fetched. Quantized tensors
are returned as raw blocks, one row per uint8 row (row = ne0 elements)."""
from __future__ import annotations

import os
import pathlib
import struct

import numpy as np
from huggingface_hub import hf_hub_url

from trainforensics.remote import _range

# ggml type -> (block size, bytes per block)
GGML = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
        7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
        12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 30: ("BF16", 1, 2),
        16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
        20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
        29: ("IQ1_M", 256, 56)}
CACHE = pathlib.Path(os.environ.get("CKPTF_CACHE", pathlib.Path.home() / ".cache" / "ckptforensics"))


class _Buf:
    def __init__(self, url, chunk=8 << 20):
        self.url, self.chunk, self.data, self.pos = url, chunk, b"", 0

    def need(self, n):
        while self.pos + n > len(self.data):
            a = len(self.data)
            self.data += _range(self.url, a, a + self.chunk - 1)

    def read(self, n):
        self.need(n)
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def u(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, self.read(n))[0]

    def s(self):
        return self.read(self.u("Q")).decode("utf-8", "replace")


_SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _val(b, t):
    if t == 8:
        return b.s()
    if t == 9:
        it, n = b.u("I"), b.u("Q")
        if it == 8:
            for _ in range(n):
                b.s()
            return None
        b.read(struct.calcsize(_SC[it]) * n)
        return None
    return b.u(_SC[t])


class RemoteGGUF:
    def __init__(self, repo, filename):
        self.repo, self.filename = repo, filename
        self.url = hf_hub_url(repo, filename)
        b = _Buf(self.url)
        assert b.read(4) == b"GGUF", "not gguf"
        self.version = b.u("I")
        nt, nkv = b.u("Q"), b.u("Q")
        self.kv = {}
        for _ in range(nkv):
            k = b.s()
            t = b.u("I")
            self.kv[k] = _val(b, t)
        self.tensors = {}
        for _ in range(nt):
            name = b.s()
            nd = b.u("I")
            dims = [b.u("Q") for _ in range(nd)]
            t = b.u("I")
            off = b.u("Q")
            self.tensors[name] = (dims, t, off)
        al = self.kv.get("general.alignment") or 32
        self.data_start = (b.pos + al - 1) // al * al

    def info(self, name):
        dims, t, _ = self.tensors[name]
        return GGML[t][0], dims

    def raw_rows(self, name):
        """uint8 array [rows, row_bytes] of the tensor's stored blocks."""
        dims, t, off = self.tensors[name]
        _, bs, bb = GGML[t]
        ne0 = dims[0]
        rows = int(np.prod(dims[1:]))
        rb = ne0 // bs * bb
        n = rb * rows
        key = f"{self.repo}__{self.filename}__{name}".replace("/", "__")
        cp = CACHE / key
        if cp.exists() and cp.stat().st_size == n:
            data = cp.read_bytes()
        else:
            parts, a, step = [], self.data_start + off, 256 << 20
            for s in range(0, n, step):
                parts.append(_range(self.url, a + s, a + min(n, s + step) - 1))
            data = b"".join(parts)
            CACHE.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(data)
        return np.frombuffer(data, dtype=np.uint8).reshape(rows, rb)
