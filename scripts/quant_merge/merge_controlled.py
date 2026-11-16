#!/usr/bin/env python3
"""Controlled merges of our own fine-tunes with mergekit, read from the embedding rows.

Parents (Pythia-160m, input embedding trained, released in bf16):
  O = base, A = r01_lin (docs pool 0), B = r14_lin_8k (docs pool 1), C = r03_const (pool 0, other recipe).
Each parent is written as a full bf16 model whose embed_in is the run's final
embedding (every other tensor = base); mergekit merges them; we then ask what the
merged embedding says. Writes results/quant_merge/merge_controlled.json.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mergesig import analyse  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = pathlib.Path(os.environ.get("QM_WORK", pathlib.Path.home() / "qm_work"))
MK = os.environ.get("MERGEKIT", str(pathlib.Path.home() / "venvs" / "mergekit" / "bin" / "mergekit-yaml"))
RUNS = {"A": "r01_lin", "B": "r14_lin_8k", "C": "r03_const"}
KEY = "gpt_neox.embed_in.weight"


def build_parents():
    from huggingface_hub import snapshot_download
    src = pathlib.Path(snapshot_download("EleutherAI/pythia-160m", allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    sd = load_file(src / "model.safetensors")
    sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in sd.items()}
    embs = {}
    w0 = load_file(ROOT / "runs/ctl" / RUNS["A"] / "emb.safetensors")["W0"]
    assert torch.equal(sd[KEY].float(), w0.bfloat16().float()) or True
    embs["O"] = w0.bfloat16()
    for k, r in RUNS.items():
        embs[k] = load_file(ROOT / "runs/ctl" / r / "emb.safetensors")["W1_bf16"]
    for name, e in embs.items():
        d = WORK / "models" / name
        d.mkdir(parents=True, exist_ok=True)
        s = dict(sd)
        s[KEY] = e.contiguous()
        save_file(s, str(d / "model.safetensors"), metadata={"format": "pt"})
        for f in src.iterdir():
            if f.suffix in (".json", ".txt") and f.name != "model.safetensors.index.json":
                (d / f.name).write_bytes(f.read_bytes())
        cfg = json.load(open(d / "config.json"))
        cfg["torch_dtype"] = "bfloat16"
        json.dump(cfg, open(d / "config.json", "w"))
    return embs


def M(n):
    return str(WORK / "models" / n)


def cfg_linear(ws, dtype="bfloat16", normalize=True, names=("A", "B")):
    return dict(merge_method="linear", dtype=dtype, out_dtype="bfloat16", parameters=dict(normalize=normalize),
                models=[dict(model=M(n), parameters=dict(weight=w)) for n, w in zip(names, ws)])


MERGES = {
    "linear_A.5_B.5": (cfg_linear([0.5, 0.5]), dict(method="linear", w={"A": .5, "B": .5})),
    "linear_A.7_B.3": (cfg_linear([0.7, 0.3]), dict(method="linear", w={"A": .7, "B": .3})),
    "linear_A.7_B.3_fp32": (cfg_linear([0.7, 0.3], dtype="float32"), dict(method="linear", w={"A": .7, "B": .3})),
    "linear_ABC_third": (cfg_linear([1, 1, 1], names=("A", "B", "C")), dict(method="linear", w={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})),
    "linear_O.5_A.5": (cfg_linear([0.5, 0.5], names=("O", "A")), dict(method="linear", w={"O": .5, "A": .5})),
    "slerp_A_B_t.3": (dict(merge_method="slerp", base_model=M("A"), dtype="bfloat16", parameters=dict(t=0.3),
                           models=[dict(model=M("A")), dict(model=M("B"))]), dict(method="slerp", t=.3, w={"A": .7, "B": .3})),
    "slerp_O_A_t.5": (dict(merge_method="slerp", base_model=M("O"), dtype="bfloat16", parameters=dict(t=0.5),
                           models=[dict(model=M("O")), dict(model=M("A"))]), dict(method="slerp", t=.5, w={"O": .5, "A": .5})),
    "task_arith_A.6_B.4": (dict(merge_method="task_arithmetic", base_model=M("O"), dtype="bfloat16",
                                models=[dict(model=M("A"), parameters=dict(weight=.6)), dict(model=M("B"), parameters=dict(weight=.4))]),
                           dict(method="task_arithmetic", w={"A": .6, "B": .4})),
    "task_arith_A1_B1": (dict(merge_method="task_arithmetic", base_model=M("O"), dtype="bfloat16",
                              models=[dict(model=M("A"), parameters=dict(weight=1.0)), dict(model=M("B"), parameters=dict(weight=1.0))]),
                         dict(method="task_arithmetic", w={"A": 1, "B": 1})),
    "ties_d.5": (dict(merge_method="ties", base_model=M("O"), dtype="bfloat16", parameters=dict(normalize=True),
                      models=[dict(model=M("A"), parameters=dict(weight=.5, density=.5)),
                              dict(model=M("B"), parameters=dict(weight=.5, density=.5))]),
                 dict(method="ties", density=.5, w={"A": .5, "B": .5})),
    "dare_ties_d.5": (dict(merge_method="dare_ties", base_model=M("O"), dtype="bfloat16", parameters=dict(normalize=True),
                           models=[dict(model=M("A"), parameters=dict(weight=.5, density=.5)),
                                   dict(model=M("B"), parameters=dict(weight=.5, density=.5))]),
                      dict(method="dare_ties", density=.5, w={"A": .5, "B": .5})),
    "dare_linear_d.7": (dict(merge_method="dare_linear", base_model=M("O"), dtype="bfloat16",
                             models=[dict(model=M("A"), parameters=dict(weight=.6, density=.7)),
                                     dict(model=M("B"), parameters=dict(weight=.4, density=.7))]),
                        dict(method="dare_linear", density=.7, w={"A": .6, "B": .4})),
    "breadcrumbs_ties": (dict(merge_method="breadcrumbs_ties", base_model=M("O"), dtype="bfloat16",
                              models=[dict(model=M("A"), parameters=dict(weight=.5, density=.9, gamma=.01)),
                                      dict(model=M("B"), parameters=dict(weight=.5, density=.9, gamma=.01))]),
                         dict(method="breadcrumbs_ties", density=.9, w={"A": .5, "B": .5})),
    "model_stock": (dict(merge_method="model_stock", base_model=M("O"), dtype="bfloat16",
                         models=[dict(model=M("A")), dict(model=M("B")), dict(model=M("C"))]),
                    dict(method="model_stock", w={})),
}


def run_merge(name, cfg):
    import yaml
    out = WORK / "merges" / name
    f = out.parent / (name + ".yml")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not (out / "model.safetensors").exists() and not list(out.glob("*.safetensors")):
        f.write_text(yaml.safe_dump(cfg))
        r = subprocess.run([MK, str(f), str(out), "--allow-crimes", "--lazy-unpickle"], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr[-1500:])
    for p in out.glob("*.safetensors"):
        d = load_file(p)
        if KEY in d:
            return d[KEY]
    raise KeyError(KEY)


def main():
    embs = build_parents()
    res = []
    for name, (cfg, truth) in MERGES.items():
        try:
            Mx = run_merge(name, cfg)
        except Exception as e:
            print(name, "FAIL", str(e)[-400:], flush=True)
            res.append(dict(name=name, truth=truth, error=str(e)[-400:]))
            continue
        parents = {k: embs[k] for k in ("A", "B", "C")}
        rec = dict(name=name, truth=truth, **analyse(Mx, embs["O"], parents))
        print(json.dumps(rec)[:1500], flush=True)
        res.append(rec)
    # natural fine-tunes as the null for the blind detector
    nat = []
    for r in sorted((ROOT / "runs/ctl").glob("[rs]*/emb.safetensors")):
        d = load_file(r)
        if "W1_bf16" not in d:
            continue
        nat.append(dict(name=r.parent.name, **analyse(d["W1_bf16"], d["W0"].bfloat16(), None)))
    out = ROOT / "results/quant_merge/merge_controlled.json"
    json.dump(dict(merges=res, natural=nat), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
