#!/usr/bin/env python3
"""Lineage statements re-derived with a margin rule (review B1).

Bit-identical token vectors show shared ancestry, not a direct parent: a sibling or an
ancestor shares the vectors of tokens neither later training touched, and separately
trained runs from one random start share their untrainable vectors. So for every lineage
statement in the reports we compare the release against each plausible candidate (named
parent, its siblings, its ancestors) on the *cleaned* count (`left`: bit-identical after
removing unreachable, reserved, at-init, zero and duplicate rows; scripts/frontier/recount.py).

Rule: a candidate is named as the closest published starting point only if its `left` count
is at least MARGIN_RATIO times, and MARGIN_ABS vectors more than, every other candidate's.
Otherwise the wording is "shares never-trained vectors with X (and Y)".
Writes results/frontier/lineage.json and lineage.md.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "frontier"))
from recount import OUT, run  # noqa: E402
from extras import meta  # noqa: E402

MARGIN_RATIO, MARGIN_ABS = 1.5, 200

DS = "deepseek-ai/DeepSeek-"
L405I, L405B = "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8", "NousResearch/Meta-Llama-3.1-405B-FP8"
CLAIMS = [
    ("moonshotai/Kimi-K2.6", ["moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2-Thinking", "moonshotai/Kimi-K2-Instruct-0905",
                              "moonshotai/Kimi-K2-Base"], "moonshotai/Kimi-K2.5", "FRONTIER_X finding 2"),
    (DS + "V3-0324", [DS + "V3", DS + "V3-Base", DS + "R1"], DS + "V3", "FRONTIER_X finding 3"),
    (DS + "V3.1", [DS + "V3.1-Base", DS + "V3-Base", DS + "V3-0324"], DS + "V3.1-Base", "FRONTIER_X finding 3"),
    (DS + "V3.1-Terminus", [DS + "V3.1", DS + "V3.1-Base"], DS + "V3.1", "FRONTIER_X finding 3"),
    (DS + "V3.2-Exp", [DS + "V3.2-Exp-Base", DS + "V3.1-Terminus", DS + "V3.1"], DS + "V3.2-Exp-Base", "FRONTIER_X finding 3"),
    (DS + "V3.2", [DS + "V3.2-Exp", DS + "V3.2-Exp-Base", DS + "V3.1-Terminus"], DS + "V3.2-Exp", "FRONTIER_X finding 3"),
    (DS + "V3.2-Speciale", [DS + "V3.2", DS + "V3.2-Exp"], DS + "V3.2", "FRONTIER_X finding 3"),
    (DS + "R1-0528", [DS + "R1", DS + "V3-Base", DS + "V3"], DS + "R1", "review B1 example"),
    ("Qwen/QwQ-32B", ["Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct", "Qwen/QwQ-32B-Preview"], "Qwen/Qwen2.5-32B",
     "LARGE finding 6"),
    ("a-m-team/AM-Thinking-v1", ["Qwen/Qwen2.5-32B", "Qwen/Qwen2.5-32B-Instruct"], "Qwen/Qwen2.5-32B", "LARGE finding 6"),
    ("nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", [L405I, L405B], L405I, "LARGE finding 6"),
    ("NousResearch/Hermes-4-405B", [L405B, L405I, "NousResearch/Hermes-3-Llama-3.1-405B"], L405B, "LARGE finding 6"),
    ("NousResearch/Hermes-4-70B", ["unsloth/Meta-Llama-3.1-70B", "unsloth/Meta-Llama-3.1-70B-Instruct"],
     "unsloth/Meta-Llama-3.1-70B", "LARGE finding 2"),
    ("Qwen/WebWorld-14B", ["Qwen/Qwen3-14B", "Qwen/Qwen3-14B-Base"], "Qwen/Qwen3-14B", "LABS2026 finding 1"),
    # siblings (other published fine-tunes of Qwen3.6-35B-A3B) are candidates too (re-review blocker 2)
    ("Qwen/Qwen-AgentWorld-35B-A3B", ["Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B-Base",
                                      "Kwaipilot/KAT-Coder-V2.5-Dev", "Diffuspace/Clover-35B-A3B-Preview"],
     None, "LABS2026 finding 2 (reworded: no starting point named)"),
    ("nvidia/Nemotron-3-Labs-Ultra-Math-SFT", ["nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
                                               "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16"],
     "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16", "LABS2026 finding 3"),
    ("nvidia/NVIDIA-Nemotron-Labs-Teacher-STEM", ["nvidia/NVIDIA-Nemotron-Labs-Teacher-Competition-Coding",
                                                  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
                                                  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16"],
     None, "LABS2026 finding 3 (teachers share vectors with each other)"),
    ("internlm/Sage-S2.1", ["zai-org/GLM-5.2", "internlm/Atria-Dawn-Preview", "zai-org/GLM-5.3-BF16", "zai-org/GLM-5.1"],
     "zai-org/GLM-5.2", "LABS2026 finding 4"),
    ("internlm/Atria-Dawn-Preview", ["zai-org/GLM-5.2", "zai-org/GLM-5.1"], "zai-org/GLM-5.2", "LABS2026 finding 4"),
    ("Kwaipilot/KAT-Coder-V2.5-Dev", ["Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen-AgentWorld-35B-A3B",
                                      "Diffuspace/Clover-35B-A3B-Preview"], "Qwen/Qwen3.6-35B-A3B",
     "LABS2026 finding 4"),
    ("microsoft/MagenticBrain", ["Qwen/Qwen3-14B", "Qwen/Qwen3-14B-Base"], "Qwen/Qwen3-14B", "LABS2026 finding 4"),
    ("MemTensor/MemT-1.0-Flash", [DS + "V4-Flash-0731", DS + "V4-Flash", DS + "V4-Flash-Base"], DS + "V4-Flash-0731",
     "FT2026 finding 3"),
    ("deep-principle/Hydrogen-1.0", ["zai-org/GLM-5.3-BF16", "zai-org/GLM-5.2"], "zai-org/GLM-5.3-BF16", "GATED"),
    ("microsoft/UserLM-2.0", ["unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B"],
     "unsloth/Meta-Llama-3.1-8B-Instruct", "GATED"),
    ("PrimeIntellect/INTELLECT-3.1", ["PrimeIntellect/INTELLECT-3", "zai-org/GLM-4.5-Air-Base", "zai-org/GLM-4.5-Air"],
     "PrimeIntellect/INTELLECT-3", "FT2026 finding 3"),
]


def rec_for(cand, child):
    dp, sp = meta(cand)
    dc, sc = meta(child)
    if sp[1] != sc[1]:
        return None
    n = min(sp[0], sc[0])
    return dict(key=f"LIN {cand} -> {child}", parent=cand, child=child, parent_revision=None, tied=False,
                dtype_parent=dp, dtype_child=dc, shape_parent=[n, sp[1]], shape_child=[n, sc[1]],
                _real_shapes=(sp, sc), embed={}, _fit=True, _src_file="lineage")


def main(workers=3):
    jobs = {}
    for child, cands, named, src in CLAIMS:
        for c in cands:
            try:
                r = rec_for(c, child)
            except Exception as e:  # noqa: BLE001
                print("meta fail", c, child, type(e).__name__, e, flush=True)
                continue
            if r is None:
                print("width differs", c, child, flush=True)
                continue
            jobs[r["key"]] = r
    print(len(jobs), "lineage comparisons", flush=True)
    with cf.ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(run, k, r, None): k for k, r in jobs.items()}
        for fu in cf.as_completed(futs):
            try:
                print(*fu.result(), flush=True)
            except Exception as e:  # noqa: BLE001
                print("ERR", futs[fu], type(e).__name__, str(e)[:300], flush=True)
    res = []
    lines = ["# Lineage statements re-derived with a margin rule", "",
             f"Generated by `scripts/frontier/lineage.py`. For each release, every plausible starting point (named parent, "
             f"siblings, ancestors) is compared on the cleaned count of bit-identical token vectors (`left`: unreachable, "
             f"reserved, at-init, zero and duplicate rows removed). A candidate is named only if its count is at least "
             f"{MARGIN_RATIO}x and {MARGIN_ABS} vectors more than every other candidate's; otherwise the release "
             f"\"shares never-trained vectors\" with the leading candidates.", "",
             "| release | candidate: cleaned bit-identical (raw) | verdict | source |", "|---|---|---|---|"]
    for child, cands, named, src in CLAIMS:
        rows = []
        for c in cands:
            safe = f"LIN {c} -> {child}".replace("/", "__").replace(" -> ", "--").replace("@", "_")
            f = OUT / (safe + ".json")
            if f.exists():
                d = json.loads(f.read_text())
                rows.append(dict(candidate=c, left=d["left"], band=d["band"], strict=d["strict"], P=d["P"],
                                 fit=d["P_exact_frac"], zeroed=d["zeroed_edits"]))
            else:
                rows.append(dict(candidate=c, left=None))
        ok = [r for r in rows if r["left"] is not None]
        ok.sort(key=lambda r: -r["left"])
        if not ok:
            verdict, top = "not computed", None
        elif len(ok) == 1:
            top = ok[0]["candidate"]
            verdict = (f"shares {ok[0]['left']:,} never-trained vectors with {top} (only candidate computed)"
                       if ok[0]["left"] else f"no cleaned vectors shared with {top}")
        else:
            a, b = ok[0], ok[1]
            if a["left"] >= MARGIN_RATIO * b["left"] and a["left"] - b["left"] >= MARGIN_ABS:
                top = a["candidate"]
                verdict = f"closest published starting point: {top}"
            else:
                top = None
                sh = [r["candidate"] for r in ok if r["left"] and r["left"] >= a["left"] / MARGIN_RATIO]
                verdict = ("shares never-trained vectors with " + " and ".join(sh)) if sh else "no cleaned vectors shared"
        agrees = None if named is None else (top == named)
        res.append(dict(release=child, candidates=rows, named_in_report=named, verdict=verdict, top=top,
                        report_statement_holds=agrees, source=src))
        cell = "; ".join(f"{r['candidate'].split('/')[-1]}: {r['left']:,} ({r['strict']:,})" if r["left"] is not None
                         else f"{r['candidate'].split('/')[-1]}: n/a" for r in rows)
        lines.append(f"| {child} | {cell} | {verdict}{'' if agrees is None else (' (report holds)' if agrees else ' (report wording must change)')} | {src} |")
    json.dump(res, open(ROOT / "results" / "frontier" / "lineage.json", "w"), indent=1)
    (ROOT / "results" / "frontier" / "lineage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
