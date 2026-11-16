#!/usr/bin/env python3
"""Fetch every input of the LIMO check at pinned revisions (pins.py) into $TF_CACHE/limo_check.

- embedding matrices (ranged reads of one safetensors tensor per model)
- the embedding slice of exp_avg, exp_avg_sq and the fp32 master weights from each of the 32 ZeRO-3 optimizer
  shards in GAIR/LIMO@1833babb94 (deleted from main in 910230fdaa)
- run metadata of the release (trainer_state.json, training_args.bin, config.json, tokenizer.json)
- the published datasets (HF) and the GitHub repository (train/data/limo.json, eval sets, train yaml history)
Writes results/limo_check/out/fetch.json (what was fetched, from where, LFS hashes); no weights or data go into git.
"""
import json, os, struct, subprocess, requests
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_url, hf_hub_download, HfApi
from common import D, save, V, H
import pins
from optim_probe import layout

api = HfApi()


def rng(url, a, b, timeout=300):
    err = None
    for _ in range(8):
        try:
            r = requests.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=timeout); r.raise_for_status()
            assert len(r.content) == b - a; return r.content
        except Exception as e:
            err = e
    raise err


def fetch_emb(tag, repo, rev):
    out = f"{D}{tag}.bin"
    idx = requests.get(hf_hub_url(repo, "model.safetensors.index.json", revision=rev), timeout=120).json()["weight_map"]
    name = [k for k in idx if "embed_tokens" in k]; assert name == ["model.embed_tokens.weight"], name
    url = hf_hub_url(repo, idx[name[0]], revision=rev)
    n = struct.unpack("<Q", rng(url, 0, 8))[0]; hdr = json.loads(rng(url, 8, 8 + n)); m = hdr[name[0]]
    assert m["shape"] == [V, H], m["shape"]
    rec = dict(repo=repo, rev=rev, shard=idx[name[0]], dtype=m["dtype"], shape=m["shape"],
               tied="lm_head.weight" not in idx)
    if not os.path.exists(out):
        a, b = m["data_offsets"]; a += 8 + n; b += 8 + n
        parts = [(s, min(s + (32 << 20), b)) for s in range(a, b, 32 << 20)]
        with ThreadPoolExecutor(12) as ex: bufs = list(ex.map(lambda p: rng(url, *p), parts))
        open(out + ".tmp", "wb").write(b"".join(bufs)); os.replace(out + ".tmp", out)
    print(tag, rec, flush=True)
    return rec


N = V * H // pins.N_RANKS * 4  # bytes of the embedding slice per rank (first parameter in the flat partition)


def fetch_rank(rank):
    done = f"{D}optim/r{rank}.done"
    if os.path.exists(done): return json.load(open(done))
    url, ents, _ = layout(rank)
    g = {k.split("/", 1)[1]: v for k, v in ents.items()}
    step = struct.unpack("<f", rng(url, g["data/0"][0], g["data/0"][0] + 4))[0]
    for name, key in (("m", "data/1"), ("v", "data/2"), ("w", "data/6")):
        off = g[key][0]; assert g[key][1] == 4095427200, (rank, key, g[key])
        parts = [(s, min(s + (32 << 20), off + N)) for s in range(off, off + N, 32 << 20)]
        open(f"{D}optim/r{rank}_{name}.bin", "wb").write(b"".join(rng(url, *p) for p in parts))
    rec = {"rank": rank, "adam_step": step}
    json.dump(rec, open(done, "w")); print("rank", rank, "step", step, flush=True)
    return rec


def main():
    os.makedirs(D + "optim", exist_ok=True)
    out = {"pins": {k: v for k, v in vars(pins).items() if k.isupper()}}
    out["embeddings"] = {t: fetch_emb(t, r, v) for t, (r, v) in pins.MODELS.items()}
    # same weights at the optimizer revision and at main: compare LFS sha256 of every safetensors shard
    def lfs(rev):
        info = api.model_info(pins.LIMO, revision=rev, files_metadata=True)
        return {s.rfilename: (s.lfs.sha256 if s.lfs else None) for s in info.siblings if s.rfilename.endswith(".safetensors")}
    a, b = lfs(pins.LIMO_OPTIM_REV), lfs(pins.LIMO_REV)
    out["weights_same_optim_rev_vs_main"] = dict(shards=len(b), identical=sum(a.get(k) == v for k, v in b.items()))
    print("weights optim rev vs main:", out["weights_same_optim_rev_vs_main"], flush=True)
    with ThreadPoolExecutor(8) as ex: ranks = list(ex.map(fetch_rank, range(pins.N_RANKS)))
    out["optimizer"] = dict(rev=pins.LIMO_OPTIM_REV, deleted_in=pins.LIMO_DELETE_REV, file=pins.OPTIM_FILE,
                            adam_step_per_rank=sorted({r["adam_step"] for r in ranks}), ranks=len(ranks))
    out["run_files"] = {}
    for f in pins.RUN_FILES:
        p = hf_hub_download(pins.LIMO, f, revision=pins.LIMO_REV, local_dir=D + "limo_files")
        out["run_files"][f] = dict(repo=pins.LIMO, rev=pins.LIMO_REV, bytes=os.path.getsize(p))
    hf_hub_download("Qwen/Qwen2.5-32B-Instruct", "tokenizer.json", revision=pins.MODELS["q32i"][1], local_dir=D + "q32i_files")
    out["datasets"] = {}
    for tag, (repo, rev, f) in pins.DATASETS.items():
        if f:
            p = hf_hub_download(repo, f, repo_type="dataset", revision=rev, local_dir=D + "ds_" + tag)
            out["datasets"][tag] = dict(repo=repo, rev=rev, file=f, bytes=os.path.getsize(p))
        else:
            from datasets import load_dataset
            ds = load_dataset(repo, split="train", revision=rev, cache_dir=D + "hf_datasets")
            out["datasets"][tag] = dict(repo=repo, rev=rev, rows=len(ds))
    g = D + "gh_limo"
    if not os.path.exists(g):
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", pins.GITHUB, g], check=True)
    subprocess.run(["git", "-C", g, "checkout", "-q", pins.GITHUB_REV], check=True)
    blob = lambda rev: subprocess.run(["git", "-C", g, "rev-parse", f"{rev}:train/data/limo.json"], capture_output=True, text=True, check=True).stdout.strip()
    yaml = lambda rev, path: [l.strip() for l in subprocess.run(["git", "-C", g, "show", f"{rev}:{path}"], capture_output=True, text=True).stdout.splitlines() if l.startswith("cutoff_len")]
    out["github"] = dict(repo=pins.GITHUB, rev=pins.GITHUB_REV, limo_json_blob=blob(pins.GITHUB_REV),
                         limo_json_unchanged_since_first_commit=blob(pins.GITHUB_REV) == blob(pins.GITHUB_FIRST),
                         yaml_cutoff_at_first_commit=yaml(pins.GITHUB_FIRST, "train/train_limo.yaml") or yaml(pins.GITHUB_FIRST, "train/examples/train_limo.yaml"),
                         yaml_cutoff_after_change=yaml(pins.GITHUB_YAML_CHANGE, "train/examples/train_limo.yaml"),
                         yaml_change_commit=pins.GITHUB_YAML_CHANGE)
    print(out["github"], flush=True)
    save("fetch", out)


if __name__ == "__main__":
    main()
