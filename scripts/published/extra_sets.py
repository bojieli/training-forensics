"""Dataset loaders for LIMO / LIMO-v2."""
import json

SYSP2 = "Please reason step by step, and put your final answer within \\boxed{}."


def _lf_limo(OUT):
    d = json.load(open(OUT / "limo.json"))
    return [(x["system"], [{"role": "user", "content": x["instruction"]},
                           {"role": "assistant", "content": x["output"]}]) for x in d]


def _lf_limo2(OUT):
    from datasets import load_dataset
    ds = load_dataset("GAIR/LIMO-v2", split="train")
    return [(SYSP2, [{"role": "user", "content": x["question"]}, {"role": "assistant", "content": x["solution"]}]) for x in ds]


def _tid_limo(ROOT):
    d = json.load(open(ROOT / "results" / "published" / "limo.json"))
    return [dict(system=x["system"], conversations=[{"from": "user", "value": x["instruction"]},
                                                    {"from": "assistant", "value": x["output"]}]) for x in d]


def _tid_limo2(ROOT):
    from datasets import load_dataset
    ds = load_dataset("GAIR/LIMO-v2", split="train")
    return [dict(system=SYSP2, conversations=[{"from": "user", "value": x["question"]},
                                              {"from": "assistant", "value": x["solution"]}]) for x in ds]


LF_ROWS = {"limo": _lf_limo, "limo2": _lf_limo2}
TID_ROWS = {"limo": _tid_limo, "limo2": _tid_limo2}
