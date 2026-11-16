"""Dataset loaders for LIMO / LIMO-v2.

LIMO v1 is read from the GitHub copy (train/data/limo.json, which carries the per-example system prompt) at a
pinned commit, cached under $TF_CACHE/limo_check/gh_limo; it is not stored in this repository."""
import json, os, pathlib, urllib.request

SYSP2 = "Please reason step by step, and put your final answer within \\boxed{}."
GITHUB_RAW = "https://raw.githubusercontent.com/GAIR-NLP/LIMO/2284c6a0e6653aa8894bd12fdecc1212ba706c3a/train/data/limo.json"
LIMO2 = ("GAIR/LIMO-v2", "ef8bf143f8cc56ca8634795835b55083a3ae3061")


def _limo_rows():
    p = pathlib.Path(os.environ.get("TF_CACHE", pathlib.Path.home() / ".cache" / "trainforensics")) / "limo_check" / "gh_limo" / "train" / "data" / "limo.json"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(GITHUB_RAW, p)
    return json.load(open(p))


def _limo2_rows():
    from datasets import load_dataset
    return load_dataset(LIMO2[0], split="train", revision=LIMO2[1])


def _lf_limo(OUT=None):
    return [(x["system"], [{"role": "user", "content": x["instruction"]},
                           {"role": "assistant", "content": x["output"]}]) for x in _limo_rows()]


def _lf_limo2(OUT=None):
    return [(SYSP2, [{"role": "user", "content": x["question"]}, {"role": "assistant", "content": x["solution"]}]) for x in _limo2_rows()]


def _tid_limo(ROOT=None):
    return [dict(system=x["system"], conversations=[{"from": "user", "value": x["instruction"]},
                                                    {"from": "assistant", "value": x["output"]}]) for x in _limo_rows()]


def _tid_limo2(ROOT=None):
    return [dict(system=SYSP2, conversations=[{"from": "user", "value": x["question"]},
                                              {"from": "assistant", "value": x["solution"]}]) for x in _limo2_rows()]


LF_ROWS = {"limo": _lf_limo, "limo2": _lf_limo2}
TID_ROWS = {"limo": _tid_limo, "limo2": _tid_limo2}
