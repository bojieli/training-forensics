#!/usr/bin/env python3
"""Large SOTA fine-tunes vs their bases: resolve candidate pairs into sweep-ready records
(results/large/pairs_large.json). Reuses scripts/roster/build_pairs.py side()/compare()."""
import json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "roster")); sys.path.insert(0, str(ROOT))
import build_pairs as bp  # noqa

R1 = "deepseek-ai/DeepSeek-R1"
R1776 = "mlx-community/perplexity-ai-r1-1776-bf16"   # perplexity-ai/r1-1776 is no longer on HF; bf16 mirror
L405 = "NousResearch/Meta-Llama-3.1-405B-FP8"         # meta-llama/Llama-3.1-405B is gated; FP8 release keeps a bf16 embedding
L405I = "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8"
CAND = [  # (parent, child, relation)
    (R1, R1776, "fine-tune (third party)"),
    (R1, "microsoft/MAI-DS-R1", "fine-tune (third party)"),
    ("microsoft/MAI-DS-R1", R1776, "sibling (both from R1)"),
    (R1, "deepseek-ai/DeepSeek-R1-0528", "successive"),
    ("deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-R1-0528", "base"),
    ("deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3.1", "successive"),
    ("Qwen/Qwen2.5-32B", "Qwen/QwQ-32B", "base"),
    ("Qwen/Qwen2.5-32B-Instruct", "Qwen/QwQ-32B", "instruct"),
    ("Qwen/Qwen2.5-32B-Instruct", "Qwen/QwQ-32B-Preview", "instruct"),
    ("Qwen/QwQ-32B-Preview", "Qwen/QwQ-32B", "successive"),
    ("Qwen/Qwen2.5-72B", "moonshotai/Kimi-Dev-72B", "base"),
    ("Qwen/Qwen2.5-72B-Instruct", "moonshotai/Kimi-Dev-72B", "instruct"),
    ("Qwen/Qwen2.5-32B", "a-m-team/AM-Thinking-v1", "base"),
    ("Qwen/Qwen2.5-32B-Instruct", "a-m-team/AM-Thinking-v1", "instruct"),
    ("Qwen/Qwen2.5-72B-Instruct", "Nexusflow/Athene-V2-Chat", "instruct"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", "nvidia/AceReason-Nemotron-14B", "distill"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "nvidia/AceReason-Nemotron-7B", "distill"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "Skywork/Skywork-OR1-32B", "distill"),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "Skywork/Skywork-OR1-7B", "distill"),
    (L405, "allenai/Llama-3.1-Tulu-3-405B-SFT", "base"),
    ("allenai/Llama-3.1-Tulu-3-405B-SFT", "allenai/Llama-3.1-Tulu-3-405B-DPO", "stage"),
    ("allenai/Llama-3.1-Tulu-3-405B-DPO", "allenai/Llama-3.1-Tulu-3-405B", "stage"),
    (L405, "NousResearch/Hermes-3-Llama-3.1-405B", "base"),
    (L405, "NousResearch/Hermes-4-405B", "base"),
    ("NousResearch/Hermes-3-Llama-3.1-405B", "NousResearch/Hermes-4-405B", "successive"),
    ("unsloth/Meta-Llama-3.1-70B", "NousResearch/Hermes-4-70B", "base"),
    (L405, L405I, "base (reference: Meta's own instruct)"),
    (L405I, "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", "instruct"),
    (L405, "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1", "base"),
]


def main():
    out, skipped, sides = [], [], {}
    def S(r):
        if r not in sides:
            sides[r] = bp.side(r)
        return sides[r]
    for par, ch, kind in CAND:
        a, b = S(par), S(ch)
        bad = [x["repo"] + ": " + ("missing" if not x.get("exists") else "gated" if x.get("gated") else
               "no embedding tensor: " + str((x.get("embed") or {}).get("error") or x.get("weights_error")))
               for x in (a, b) if not x.get("exists") or x.get("gated") or not (x.get("embed") or {}).get("shape")]
        if bad:
            skipped.append(dict(parent=par, child=ch, why="; ".join(bad))); print("skip", par, "->", ch, bad, flush=True); continue
        cmp_ = bp.compare(a, b)
        ce = cmp_.get("embed") or {}
        if not ce.get("compatible"):
            skipped.append(dict(parent=par, child=ch, why="embedding not comparable: " + str(ce.get("why")))); continue
        out.append(dict(key=f"{par}@ -> {ch}", parent=a, child=b, src=["large:" + kind], compat=cmp_,
                        tied=bool(a.get("tied") or b.get("tied")), relation=kind, rank=0,
                        bytes=a["embed"]["bytes"] + b["embed"]["bytes"]))
        print("ok ", par, "->", ch, a["embed"]["shape"], b["embed"]["shape"], flush=True)
    od = ROOT / "results" / "large"; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "pairs_large.json", "w"), indent=1, default=str)
    json.dump(skipped, open(od / "skipped_large.json", "w"), indent=1, default=str)
    print(len(out), "pairs;", len(skipped), "skipped; GB", round(sum(p["bytes"] for p in out) / 1e9, 1))


if __name__ == "__main__":
    main()
