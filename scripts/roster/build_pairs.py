#!/usr/bin/env python3
"""Build roster/pairs.json: for every roster model, the public checkpoint it was
most directly trained FROM (parent), popular derived children for bases, and the
embedding / output-head tensor metadata of every side, so embeddings can be diffed.

All Hub traffic (model_info, config.json, safetensors index + headers, child
listings) is cached on disk under $TF_HUB_CACHE (default ~/.cache/trainforensics/hub),
so reruns are cheap. Delete the cache directory (or pass --refresh) to refetch.

Usage:  python scripts/roster/build_pairs.py [--refresh] [--workers 8] [--only NAME ...]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import re
import struct
import sys
import threading
import time

import requests
from huggingface_hub import HfApi, hf_hub_url, get_token

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from trainforensics import remote  # noqa: E402

CKPTF = pathlib.Path.home() / "checkpoint-forensics"
ROSTER = ROOT / "roster" / "roster.json"
SUMMARY = CKPTF / "results" / "summary.json"
CARDS = CKPTF / "ckptforensics" / "cards"
OUT = ROOT / "roster" / "pairs.json"
CACHE = pathlib.Path(os.environ.get("TF_HUB_CACHE", pathlib.Path.home() / ".cache" / "trainforensics" / "hub"))
REFRESH = False
API = HfApi()
_locks: dict[str, threading.Lock] = {}
_glock = threading.Lock()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lineage import OVERRIDES, EXTRA_PAIRS, CHILD_OVERRIDES  # noqa: E402


# --------------------------------------------------------------------------- cache
class Transient(Exception):
    """Rate limit / server error: never cached."""


def _cached(kind: str, key: str, fn):
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", key)[:80]
    p = CACHE / kind / f"{safe}.{h}.json"
    with _glock:
        lk = _locks.setdefault(str(p), threading.Lock())
    with lk:
        if p.exists() and not REFRESH:
            return json.loads(p.read_text())
        for i in range(8):
            try:
                val = fn()
                break
            except Transient:
                time.sleep(min(300, 15 * 2 ** i))
        else:
            raise IOError(f"persistent transient errors for {kind}:{key}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(val))
        return val


def _get(url, headers=None, timeout=60):
    for i in range(4):
        try:
            return requests.get(url, headers=remote._headers(headers), timeout=timeout, allow_redirects=True)
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    raise IOError(url)


def model_info(repo: str) -> dict:
    def f():
        try:
            i = API.model_info(repo, expand=["cardData", "gated", "siblings", "downloads", "likes",
                                             "tags", "createdAt", "safetensors", "sha"])
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (429, 500, 502, 503, 504) or code is None and name not in (
                    "RepositoryNotFoundError", "GatedRepoError"):
                raise Transient(name)
            return {"ok": False, "error": name, "status": code}
        cd = i.card_data.to_dict() if i.card_data else {}
        return {"ok": True, "id": i.id, "sha": i.sha, "gated": i.gated, "downloads": i.downloads,
                "likes": i.likes, "tags": i.tags or [],
                "created": str(i.created_at) if i.created_at else None,
                "files": [s.rfilename for s in (i.siblings or [])],
                "params": dict(i.safetensors.parameters) if i.safetensors else None,
                "card": {k: cd.get(k) for k in ("base_model", "base_model_relation", "datasets",
                                                "license", "license_name", "pipeline_tag", "library_name")}}
    return _cached("info", repo, f)


def fetch_json(repo: str, filename: str, revision=None):
    def f():
        r = _get(hf_hub_url(repo, filename, revision=revision))
        if r.status_code in (429, 500, 502, 503, 504):
            raise Transient(r.status_code)
        if r.status_code != 200:
            return {"__status__": r.status_code}
        try:
            return {"__status__": 200, "data": r.json()}
        except ValueError:
            return {"__status__": -1}
    return _cached("file", f"{repo}@{revision}/{filename}", f)


def readme(repo: str) -> str:
    def f():
        r = _get(hf_hub_url(repo, "README.md"))
        if r.status_code in (429, 500, 502, 503, 504):
            raise Transient(r.status_code)
        return {"status": r.status_code, "text": r.text[:60000] if r.status_code == 200 else ""}
    return _cached("readme", repo, f)["text"]


def st_header(repo: str, filename: str, revision=None) -> dict:
    """{tensor: [dtype, shape]} from a safetensors file header (2 range requests)."""
    def f():
        url = hf_hub_url(repo, filename, revision=revision)
        try:
            n = struct.unpack("<Q", remote._range(url, 0, 7))[0]
            h = json.loads(remote._range(url, 8, 8 + n - 1))
        except Exception as e:  # noqa: BLE001
            if "(429)" in str(e) or "(50" in str(e):
                raise Transient(str(e))
            return {"__error__": f"{type(e).__name__}: {e}"[:300]}
        h.pop("__metadata__", None)
        return {k: [v["dtype"], v["shape"]] for k, v in h.items()}
    return _cached("sthdr", f"{repo}@{revision}/{filename}", f)


def bin_info(repo: str, filename: str, names: tuple, revision=None) -> dict:
    def f():
        try:
            rb = remote.open_file(repo, filename, revision)
            out = {}
            for n in rb.keys():
                if n in names or _is_emb_candidate(n):
                    d, s = rb.info(n)
                    out[n] = [str(d), list(s)]
            return {"__keys__": rb.keys(), **out}
        except Exception as e:  # noqa: BLE001
            return {"__error__": f"{type(e).__name__}: {e}"[:300]}
    return _cached("binhdr", f"{repo}@{revision}/{filename}", f)


def list_children(repo: str, author: str) -> list:
    def f():
        out = []
        try:
            for m in API.list_models(filter=f"base_model:finetune:{repo}", author=author,
                                     sort="downloads", direction=-1, limit=15):
                out.append({"id": m.id, "downloads": m.downloads})
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "50" in str(getattr(getattr(e, "response", None), "status_code", "")):
                raise Transient(str(e))
            out.append({"error": type(e).__name__})
        return out
    return _cached("children", f"{repo}|{author}", f)


def org_search(author: str, stem: str) -> dict:
    """lower-cased repo id -> exact repo id, for all repos of `author` whose name contains `stem`."""
    def f():
        try:
            return {m.id.lower(): m.id for m in API.list_models(author=author, search=stem, limit=1000)}
        except Exception as e:  # noqa: BLE001
            raise Transient(str(e))
    return _cached("orgsearch", f"{author}|{stem}", f)


def exists(repo: str, stem: str | None = None) -> str | None:
    org, nm = repo.split("/")
    return org_search(org, stem or nm).get(repo.lower())


# --------------------------------------------------------------------------- tensors
_EXCL = re.compile(r"(vision|visual|audio|image|vit|mm_|projector|position|pos_emb|per_layer|"
                   r"altup|laurel|speech|mtp|nextn|draft|token_type|patch|rotary|layers\.\d|"
                   r"\.h\.\d|blocks\.\d|encoder\.)", re.I)
_IN = [r"embed_tokens\.weight$", r"\bwte\.weight$", r"embed_in\.weight$", r"word_embeddings\.weight$",
       r"tok_embeddings\.weight$", r"embedding\.weight$", r"embeddings\.weight$", r"\bembed\.weight$",
       r"tok_embeddings$", r"wte$", r"embedder\.weight$", r"emb\.weight$", r"in_embed.*\.weight$"]
_OUT = [r"lm_head\.weight$", r"embed_out\.weight$", r"(^|\.)output\.weight$", r"output_layer\.weight$",
        r"lm_head\.linear\.weight$", r"(^|\.)head\.weight$", r"lm_heads?\.\d*\.?weight$", r"unembed.*weight$"]


def _is_emb_candidate(k):
    return any(re.search(p, k) for p in _IN + _OUT) and not _EXCL.search(k)


def _pick(keys, pats):
    cands = [k for k in keys if any(re.search(p, k) for p in pats) and not _EXCL.search(k)]
    if not cands:
        return None
    def score(k):
        lm = 0 if re.search(r"language_model|text_model|^model\.|^transformer\.|^gpt_neox\.|^backbone\.", k) else 1
        return (min(i for i, p in enumerate(pats) if re.search(p, k)), lm, len(k))
    return sorted(cands, key=score)[0]


_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
          "torch.float32": 4, "torch.float16": 2, "torch.bfloat16": 2}


def weights_layout(repo: str, info: dict, revision=None) -> dict:
    files = info.get("files", [])
    fmt = ("safetensors" if any(f.endswith(".safetensors") for f in files) else
           "bin" if any(f.endswith(".bin") for f in files) else
           "gguf" if any(f.endswith(".gguf") for f in files) else "other")
    root = [f for f in files if "/" not in f]
    wm, src = None, None
    for idx in ("model.safetensors.index.json", "consolidated.safetensors.index.json", "pytorch_model.bin.index.json"):
        if idx in root:
            j = fetch_json(repo, idx, revision)
            if j["__status__"] == 200 and set(j["data"]["weight_map"].values()) <= set(files):
                wm, src = j["data"]["weight_map"], idx
                break
    if wm is None:
        for single in ("model.safetensors", "consolidated.safetensors", "pytorch_model.bin"):
            if single in root:
                if single.endswith(".bin"):
                    b = bin_info(repo, single, (), revision)
                    if "__error__" in b:
                        return {"format": fmt, "error": b["__error__"]}
                    wm = {k: single for k in b["__keys__"]}
                else:
                    h = st_header(repo, single, revision)
                    if "__error__" in h:
                        return {"format": fmt, "error": h["__error__"]}
                    wm = {k: single for k in h}
                src = single
                break
    if wm is None:
        return {"format": fmt, "error": "no root-level HF weights (index/model.safetensors/pytorch_model.bin)"}
    emb, head = _pick(wm, _IN), _pick(wm, _OUT)
    out = {"format": "safetensors" if src.endswith((".safetensors", ".safetensors.index.json")) else "bin",
           "weights_index": src, "n_tensors": len(wm), "embed": None, "head": None}
    for role, name in (("embed", emb), ("head", head)):
        if not name:
            continue
        fn = wm[name]
        if fn.endswith(".bin"):
            d = bin_info(repo, fn, (name,), revision).get(name)
        else:
            d = st_header(repo, fn, revision).get(name)
        if d:
            dt, shp = d
            n = 1
            for s in shp:
                n *= s
            out[role] = {"tensor": name, "file": fn, "dtype": dt, "shape": shp,
                         "bytes": n * _BYTES.get(dt, 2)}
        else:
            out[role] = {"tensor": name, "file": fn, "error": "header lookup failed"}
    return out


# --------------------------------------------------------------------------- config
_INITKEY = re.compile(r"(init|_std$|^std$|multiplier|scale_emb|embed.*scale|scale.*emb|mup|"
                      r"logit.*scale|output_mult|input_mult|width_mult)", re.I)


def config_facts(repo: str, revision=None) -> dict:
    j = fetch_json(repo, "config.json", revision)
    if j["__status__"] != 200:
        p = fetch_json(repo, "params.json", revision)
        if p["__status__"] == 200:
            c = p["data"]
            return {"status": 200, "source": "params.json", "vocab_size": c.get("vocab_size"),
                    "hidden_size": c.get("dim"), "tie_word_embeddings": c.get("tied_embeddings")}
        return {"status": j["__status__"]}
    c = j["data"]
    tc = c.get("text_config") or c.get("llm_config") or c.get("language_config") or {}
    def g(k):
        v = c.get(k)
        return tc.get(k) if v is None and isinstance(tc, dict) else v
    init = {}
    def walk(d, pre=""):
        if not isinstance(d, dict):
            return
        for k, v in d.items():
            if isinstance(v, dict) and k in ("text_config", "llm_config", "language_config", "mup", "mup_config"):
                walk(v, pre + k + ".")
            elif _INITKEY.search(k) and isinstance(v, (int, float, str, bool, list)) and k not in ("initializer_cutoff_factor_unused",):
                if "rope" in k or "rotary" in k or "attn_logit" in k:
                    continue
                init[pre + k] = v
    walk(c)
    return {"status": 200, "model_type": c.get("model_type") or tc.get("model_type"),
            "architectures": c.get("architectures"), "vocab_size": g("vocab_size") or g("padded_vocab_size"),
            "hidden_size": g("hidden_size") or g("d_model") or g("n_embd") or g("dim"),
            "tie_word_embeddings": g("tie_word_embeddings"), "torch_dtype": g("torch_dtype") or g("dtype"),
            "init_and_scaling": init}


# --------------------------------------------------------------------------- lineage
_SUFFIX = re.compile(r"[-_](instruct|chat|it|thinking|thinker|think|rl|sft|dpo|exp|preview|hf|"
                     r"\d{4}|\d{6}|fp8|bf16|instruct-\d{4}|instruct-\d{6})$", re.I)


def _norm_base(b):
    if isinstance(b, list):
        b = b[0] if b else None
    if isinstance(b, str):
        b = b.strip()
        b = re.sub(r"^https?://huggingface\.co/", "", b)
    return b or None


def side(repo: str, revision=None) -> dict:
    info = model_info(repo)
    s = {"repo": info.get("id", repo) if info.get("ok") else repo, "revision": revision}
    if not info.get("ok"):
        s.update({"exists": False, "error": info.get("error"), "status": info.get("status"),
                  "gated": info.get("error") in ("GatedRepoError",) or info.get("status") in (401, 403)})
        return s
    cfg = config_facts(repo, revision)
    s.update({"exists": True, "gated": bool(info.get("gated")), "gated_mode": info.get("gated") or None,
              "accessible": cfg.get("status") == 200, "downloads": info.get("downloads"),
              "license": info["card"].get("license"), "card_base_model": info["card"].get("base_model"),
              "datasets": info["card"].get("datasets")})
    if cfg.get("status") in (401, 403):
        s["gated"] = True
    lay = weights_layout(repo, info, revision) if s["accessible"] or True else {}
    s["format"] = lay.get("format")
    s["weights_index"] = lay.get("weights_index")
    if lay.get("error"):
        s["weights_error"] = lay["error"]
    s["embed"], s["head"] = lay.get("embed"), lay.get("head")
    s["config"] = {k: cfg.get(k) for k in ("model_type", "vocab_size", "hidden_size", "tie_word_embeddings", "torch_dtype")}
    s["init_and_scaling"] = cfg.get("init_and_scaling")
    has_head = bool(s["head"]) and s["head"].get("tensor") != (s["embed"] or {}).get("tensor")
    if s["embed"] is None and s["head"] is None:
        s["tied"] = cfg.get("tie_word_embeddings")
    else:
        s["tied"] = not has_head
    if has_head and cfg.get("tie_word_embeddings") is True:
        s["tie_note"] = "config says tied but a separate head tensor exists"
    return s


def compare(parent: dict, child: dict) -> dict:
    out = {}
    for role in ("embed", "head"):
        a, b = (parent or {}).get(role), (child or {}).get(role)
        if not a or not b or "shape" not in a or "shape" not in b:
            out[role] = {"compatible": None, "why": "missing tensor info"}
            continue
        (va, da), (vb, db) = a["shape"][:2] if len(a["shape"]) >= 2 else (a["shape"][0], 1), \
                             b["shape"][:2] if len(b["shape"]) >= 2 else (b["shape"][0], 1)
        r = {"parent_shape": a["shape"], "child_shape": b["shape"], "same_shape": a["shape"] == b["shape"],
             "dtype_parent": a["dtype"], "dtype_child": b["dtype"]}
        if da != db:
            r.update(compatible=False, why="hidden size differs")
        else:
            r.update(compatible=True, common_rows=[0, min(va, vb)])
            if va != vb:
                r["vocab_change"] = "extended in child" if vb > va else "truncated in child"
        out[role] = r
    return out


def resolve_parent(name: str, rec: dict, info: dict, inherits: dict, roster: dict) -> dict:
    ov = OVERRIDES.get(name)
    if ov is not None:
        d = dict(ov)
        d.setdefault("how", "known lineage")
        return d
    repo = rec["repo"]
    if info.get("ok"):
        b = _norm_base(info["card"].get("base_model"))
        norm = lambda x: re.sub(r"^meta-", "", x.split("/")[-1].lower())  # noqa: E731
        if b and b.lower() != repo.lower() and norm(b) == norm(repo):
            return {"role": "unknown", "parent_repo": None, "mirror_of": b, "how": "card base_model is the original "
                    "of this mirror", "confidence": "low"}
        if b and b.lower() != repo.lower():
            bi = model_info(b)
            if bi.get("ok"):
                return {"role": "derived", "parent_repo": bi["id"], "how": "card base_model",
                        "confidence": "high", "relation": info["card"].get("base_model_relation")}
    if name in inherits:
        pr = roster.get(inherits[name], {}).get("repo")
        if pr:
            return {"role": "derived", "parent_repo": pr, "how": "checkpoint-forensics inherits (shared init draw)",
                    "confidence": "medium"}
    # same-family naming: X-Instruct -> X-Base / X
    org, nm = repo.split("/")
    stem, cand = nm, []
    for _ in range(3):
        st2 = _SUFFIX.sub("", stem)
        if st2 == stem:
            break
        stem = st2
        cand += [f"{org}/{stem}-Base", f"{org}/{stem}-base", f"{org}/{stem}"]
    for c in cand:
        if c.lower() == repo.lower():
            continue
        cid = exists(c, stem)
        if cid:
            return {"role": "derived", "parent_repo": cid, "how": "same-family naming", "confidence": "medium"}
    return {"role": "base" if re.search(r"base|pretrain|-pt$", nm, re.I) else "unknown", "parent_repo": None,
            "how": "no card base_model, no family parent found", "confidence": "low",
            "readme_hint": readme_hint(repo)}


_HINT = re.compile(r"[^.\n]{0,160}(fine-?tuned (?:from|on top of)|based on|built (?:on|upon)|continued?[- ]pre-?train|"
                   r"initiali[sz]ed from|post-?trained (?:from|on)|derived from|upcycl|base model)[^.\n]{0,160}", re.I)


def readme_hint(repo: str) -> list:
    txt = readme(repo)
    out = []
    for m in _HINT.finditer(txt):
        t = " ".join(m.group(0).split())
        if t not in out:
            out.append(t)
        if len(out) >= 4:
            break
    return out


def find_children(repo: str) -> list:
    if repo in CHILD_OVERRIDES:
        return CHILD_OVERRIDES[repo]
    org, nm = repo.split("/")
    stem = re.sub(r"[-_](base|pretrain|pt|hf)$", "", nm, flags=re.I)
    stem = re.sub(r"-Base-", "-", stem)
    names = [f"{stem}-Instruct", f"{stem}-instruct", f"{stem}-Chat", f"{stem}-chat", f"{stem}-it",
             f"{stem}-Thinking", f"{stem}-Instruct-2507", stem]
    out = []
    for n in names:
        c = f"{org}/{n}"
        if c.lower() == repo.lower():
            continue
        cid = exists(c, stem)
        if cid and cid not in out:
            out.append(cid)
    for m in list_children(repo, org):
        if "id" in m and m["id"] not in out and not re.search(r"(gguf|awq|gptq|fp8|int4|int8|mlx|bnb|4bit|8bit|nvfp4|quant)", m["id"], re.I):
            out.append(m["id"])
    return out[:3]


# --------------------------------------------------------------------------- main
def build_one(name, rec, summ, inherits, roster):
    repo = rec["repo"]
    info = model_info(repo)
    child = side(repo)
    lin = resolve_parent(name, rec, info, inherits, roster)
    r = {"name": name, "repo": repo, "role": lin["role"], "stage": lin.get("stage"),
         "parent_repo": lin.get("parent_repo"), "parent_revision": lin.get("parent_revision"),
         "parent_alt": lin.get("parent_alt"), "how": lin.get("how"), "confidence": lin.get("confidence"),
         "lineage_note": lin.get("note"), "self": child}
    for k in ("mirror_of", "parent_canonical", "candidate_parent", "row_map", "init_revision", "readme_hint",
              "relation"):
        if lin.get(k):
            r[k] = lin[k]
    if lin.get("parent_repo"):
        p = side(lin["parent_repo"], lin.get("parent_revision"))
        r["parent"] = p
        r["compat"] = compare(p, child)
        r["usable"] = bool(p.get("exists") and p.get("accessible") and child.get("accessible")
                           and r["compat"]["embed"].get("compatible") and p.get("format") == "safetensors"
                           and child.get("format") == "safetensors")
    kids = lin.get("children") or (find_children(repo) if lin["role"] in ("base", "unknown") else [])
    if kids:
        r["children"] = []
        for k in kids[:3]:
            ks = side(k)
            r["children"].append({"repo": ks["repo"], "side": ks, "compat": compare(child, ks)})
    s = summ.get(name, {})
    r["forensics"] = {"status": s.get("status"), "init_group": s.get("init_group"), "inherits": s.get("inherits"),
                      "tensor": s.get("tensor"), "shape": s.get("shape"), "seed": s.get("seed"),
                      "threads": s.get("threads"), "evidence": s.get("evidence")}
    if s.get("status") == "fingerprint":
        cp = CARDS / f"{name}.json"
        r["forensics"]["card"] = str(cp) if cp.exists() else None
        if cp.exists():
            c = json.loads(cp.read_text())
            r["forensics"]["card_init"] = {k: c.get(k) for k in ("dist", "std", "init_dtype", "tensor", "offset", "tp")}
        r["forensics"]["config_init"] = child.get("init_and_scaling")
        mt = (child.get("config") or {}).get("model_type") or ""
        if mt.startswith("gemma") or mt.startswith("t5"):
            r["forensics"]["embedding_scale_note"] = "forward multiplies embeddings by sqrt(hidden_size)"
    return r


def build_extra(ep):
    p = side(ep["parent"], ep.get("parent_revision"))
    c = side(ep["child"])
    comp = compare(p, c)
    out = dict(ep)
    out.update({"parent_side": p, "child_side": c, "compat": comp,
                "tied": bool(p.get("tied") or c.get("tied")),
                "embed_bytes": (c.get("embed") or {}).get("bytes"),
                "child_card_datasets": c.get("datasets"),
                "usable": bool(p.get("accessible") and c.get("accessible") and comp["embed"].get("compatible")
                               and p.get("format") == "safetensors" and c.get("format") == "safetensors")})
    return out


def _pair_class(p: dict, c: dict, comp: dict) -> dict:
    """Classify a parent->child pair."""
    gated = bool(p.get("gated") and not p.get("accessible")) or bool(c.get("gated") and not c.get("accessible"))
    access = bool(p.get("exists") and p.get("accessible") and c.get("exists") and c.get("accessible"))
    st = p.get("format") == "safetensors" and c.get("format") == "safetensors"
    e = comp.get("embed", {})
    tied = bool(p.get("tied") or c.get("tied"))
    return {"accessible": access, "gated": gated, "safetensors": st, "tied": tied,
            "compatible": bool(e.get("compatible")), "same_shape": bool(e.get("same_shape")),
            "vocab_change": e.get("vocab_change"), "hidden_differs": e.get("compatible") is False,
            "usable": access and st and not tied and bool(e.get("compatible"))}


def _gb(x):
    return f"{x / 1e9:.2f}" if x else "?"


def summarize(out: dict) -> str:
    from collections import Counter
    M = out["models"]
    L = [f"# Roster parent/child pairs ({out['generated']})", "",
         "Generated by `scripts/roster/build_pairs.py` from `roster/roster.json`; details in `roster/pairs.json`.",
         "A pair is **usable** when both sides are public and readable with our token, both are safetensors, "
         "neither side ties input and output embeddings, and the input embeddings have the same hidden size "
         "(vocab may differ: diff the common rows).", ""]
    role = Counter(m.get("role") for m in M)
    conf = Counter((m.get("role"), m.get("confidence")) for m in M)
    stage = Counter(m.get("stage") for m in M if m.get("role") == "base")
    L += ["## Roster lineage", "",
          f"- models: {len(M)}; roles: " + ", ".join(f"{k} {v}" for k, v in role.most_common()),
          "- derived by confidence: " + ", ".join(f"{k[1]} {v}" for k, v in conf.items() if k[0] == "derived"),
          "- base by stage: " + ", ".join(f"{k or 'pretrained'}: {v}" for k, v in stage.items()),
          "- how derived parents were found: " + ", ".join(
              f"{k} {v}" for k, v in Counter(m.get("how") for m in M if m.get("role") == "derived").most_common()), ""]
    # parent pairs
    rows, cls = [], []
    for m in M:
        if m.get("parent"):
            c = _pair_class(m["parent"], m["self"], m["compat"])
            cls.append(c)
            rows.append((m, m["parent"], m["self"], m["compat"], c, "parent"))
    kid = []
    for m in M:
        for ch in m.get("children", []) or []:
            c = _pair_class(m["self"], ch["side"], ch["compat"])
            kid.append(c)
            rows.append((m, m["self"], ch["side"], ch["compat"], c, "child"))
    def counts(cs, title):
        n = len(cs)
        return [f"### {title}", "",
                f"- pairs: {n}", f"- **usable (both public, safetensors, untied, compatible): {sum(c['usable'] for c in cs)}**",
                f"- both sides accessible: {sum(c['accessible'] for c in cs)}",
                f"- gated / inaccessible side: {sum(c['gated'] for c in cs)}; "
                f"missing/unreadable side: {sum(not c['accessible'] and not c['gated'] for c in cs)}",
                f"- tied embeddings (either side): {sum(c['tied'] for c in cs)}",
                f"- non-safetensors side: {sum(not c['safetensors'] for c in cs)}",
                f"- identical embedding shape: {sum(c['same_shape'] for c in cs)}; vocab changed "
                f"(diff common rows): {sum(bool(c['vocab_change']) for c in cs)}; hidden size differs: "
                f"{sum(c['hidden_differs'] for c in cs)}", ""]
    L += ["## Pair counts", ""] + counts(cls, "Roster model vs its parent (derived models)") + \
         counts(kid, "Roster base vs its public children (base -> Instruct/Chat/...)")
    fp = [m for m in M if (m.get("forensics") or {}).get("status") == "fingerprint"]
    L += ["## Fingerprinted models (weight-decay test)", "",
          f"- fingerprinted in summary.json: {len(fp)}",
          f"- with a card in ckptforensics/cards: {sum(bool(m['forensics'].get('card')) for m in fp)}",
          f"- config.json states an init std (initializer_range / init_std / ...): "
          f"{sum(any('init' in k or 'std' in k for k in (m['forensics'].get('config_init') or {})) for m in fp)}",
          f"- config has an embedding multiplier / scale key: "
          f"{sum(any(re.search('mult|scale', k) for k in (m['forensics'].get('config_init') or {})) for m in fp)}",
          f"- Gemma-style sqrt(d) embedding scale: {sum(bool(m['forensics'].get('embedding_scale_note')) for m in fp)}", ""]
    # table of usable pairs
    L += ["## Usable roster pairs", "", "| model | kind | parent -> child | embed shape (parent -> child) | emb GB | note |",
          "|---|---|---|---|---|---|"]
    for m, p, c, comp, k, kind in rows:
        if not k["usable"]:
            continue
        e = comp["embed"]
        note = (e.get("vocab_change") or "") + ("" if m.get("confidence") in (None, "high") or kind == "child"
                                                else f" conf={m.get('confidence')}")
        L.append(f"| {m['name']} | {kind} | {p['repo']} -> {c['repo']} | {e['parent_shape']} -> {e['child_shape']} | "
                 f"{_gb((c.get('embed') or {}).get('bytes'))} | {note.strip()} |")
    L += ["", "## Not usable (roster parent pairs)", "", "| model | parent | reason |", "|---|---|---|"]
    for m, p, c, comp, k, kind in rows:
        if k["usable"] or kind != "parent":
            continue
        why = [w for w, f in (("gated", k["gated"]), ("tied", k["tied"]), ("not safetensors", not k["safetensors"]),
                              ("hidden size differs", k["hidden_differs"]),
                              ("side missing/unreadable", not k["accessible"] and not k["gated"])) if f]
        L.append(f"| {m['name']} | {p['repo']} | {', '.join(why)} |")
    nb = [m for m in M if m.get("role") == "base" and not m.get("children")]
    un = [m for m in M if m.get("role") == "unknown"]
    L += ["", f"## Bases without a public child ({len(nb)})", "", ", ".join(m["name"] for m in nb), "",
          f"## Unknown lineage ({len(un)})", ""]
    for m in un:
        L.append(f"- {m['name']} ({m['repo']}): {m.get('lineage_note') or m.get('how')}"
                 + (f"; candidate {m['candidate_parent']}" if m.get("candidate_parent") else ""))
    X = out.get("extra_pairs", [])
    xc = [_pair_class(x["parent_side"], x["child_side"], x["compat"]) for x in X]
    L += ["", "## Extra (non-roster) pairs", ""] + counts(xc, "Extra pairs")
    L += ["| family | stage | parent -> child | usable | tied | embed (child) | emb GB | datasets |", "|---|---|---|---|---|---|---|---|"]
    for x, k in zip(X, xc):
        e = x["child_side"].get("embed") or {}
        pr = x["parent"] + (f"@{x['parent_revision']}" if x.get("parent_revision") else "")
        L.append(f"| {x['family']} | {x['stage']} | {pr} -> {x['child']} | {'yes' if k['usable'] else 'no'} | "
                 f"{'yes' if k['tied'] else 'no'} | {e.get('shape')} | {_gb(e.get('bytes'))} | {', '.join(x['datasets'])} |")
    return "\n".join(L) + "\n"


def main():
    global REFRESH
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    REFRESH = a.refresh
    roster = json.loads(ROSTER.read_text())
    summ = {x["name"]: x for x in json.loads(SUMMARY.read_text())}
    inherits = {}
    for n, x in summ.items():
        h = x.get("inherits")
        if h and h in roster:
            inherits[n] = h
    names = a.only or list(roster)
    res = {}
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(build_one, n, roster[n], summ, inherits, roster): n for n in names}
        for f in cf.as_completed(futs):
            n = futs[f]
            try:
                res[n] = f.result()
            except Exception as e:  # noqa: BLE001
                res[n] = {"name": n, "repo": roster[n]["repo"], "error": f"{type(e).__name__}: {e}"}
                print("ERR", n, e, file=sys.stderr)
        extras = list(ex.map(build_extra, EXTRA_PAIRS)) if not a.only else []
    out = {"generated": time.strftime("%Y-%m-%d"), "n": len(res),
           "models": [res[n] for n in names], "extra_pairs": extras}
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    if not a.only:
        md = pathlib.Path(a.out).with_name("pairs_summary.md")
        md.write_text(summarize(out))
        print("wrote", md)
    print("wrote", a.out, len(res), "models", len(extras), "extra pairs")


if __name__ == "__main__":
    main()
