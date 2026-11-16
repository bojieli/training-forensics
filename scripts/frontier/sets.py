#!/usr/bin/env python3
"""Set comparisons behind two report claims (review minors 7 and 8), on cleaned sets
(`left_ids`: bit-identical after removing unreachable, reserved, at-init, zero, duplicate rows).

Hermes-4: "70B and 405B leave the same tokens bit-identical". Compared with the sets other
Llama-3.1 post-trainings leave bit-identical, to see how much of the overlap any tune shares.
MemT-1.0-Flash: is its set against DeepSeek-V4-Flash the *same set* as V4-Flash-0731's
against V4-Flash (the evidence for 0731 as its starting point), not just the same count?
Writes results/frontier/sets.json.
"""
import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RC = ROOT / "results" / "frontier" / "recount"


def find(parent, child):
    for f in glob.glob(str(RC / "*.json")):
        d = json.loads(open(f).read())
        if d["parent"] == parent and d["child"] == child and d.get("left_ids") is not None:
            return set(d["left_ids"]), d
    return None, None


def cmp(a, b):
    if a is None or b is None:
        return None
    return dict(a=len(a), b=len(b), both=len(a & b), only_a=len(a - b), only_b=len(b - a),
                jaccard=round(len(a & b) / max(1, len(a | b)), 4), identical=a == b)


def main():
    out = {}
    L70, L405 = "unsloth/Meta-Llama-3.1-70B", "NousResearch/Meta-Llama-3.1-405B-FP8"
    h70, _ = find(L70, "NousResearch/Hermes-4-70B")
    h405, _ = find(L405, "NousResearch/Hermes-4-405B")
    out["hermes4_70B_vs_405B"] = cmp(h70, h405)
    base = {}
    for par, ch in ((L405, "NousResearch/Hermes-3-Llama-3.1-405B"), (L405, "allenai/Llama-3.1-Tulu-3-405B-SFT"),
                    ("meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
                    ("unsloth/Meta-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct"),
                    ("meta-llama/Meta-Llama-3.1-70B", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
                    ("unsloth/Meta-Llama-3.1-70B", "unsloth/Meta-Llama-3.1-70B-Instruct"),
                    ("meta-llama/Meta-Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-SFT")):
        s, _ = find(par, ch)
        if s is not None:
            base[ch] = s
    both = (h70 & h405) if h70 is not None and h405 is not None else None
    out["hermes4_shared_vs_other_llama31_tunes"] = {k: dict(cmp(both, v) or {}, shared_inside_other=(
        round(len(both & v) / max(1, len(both)), 4) if both is not None else None)) for k, v in base.items()}
    if both is not None and base:
        inter = set.intersection(*base.values())
        out["hermes4_shared_also_left_by_every_other_tune"] = dict(n=len(both & inter), of=len(both))
    D = "deepseek-ai/DeepSeek-"
    m_flash, _ = find(D + "V4-Flash", "MemTensor/MemT-1.0-Flash")
    m_0731, _ = find(D + "V4-Flash-0731", "MemTensor/MemT-1.0-Flash")
    f_0731, _ = find(D + "V4-Flash", D + "V4-Flash-0731")
    out["memt_vs_v4flash__and__0731_vs_v4flash"] = cmp(m_flash, f_0731)
    out["memt_vs_0731"] = None if m_0731 is None else dict(n=len(m_0731))
    json.dump(out, open(ROOT / "results" / "frontier" / "sets.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
