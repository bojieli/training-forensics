#!/usr/bin/env python3
"""Special-token grid for non-DeepSeek frontier releases (DeepSeek's grid is built separately).

For each release vs its base, named special tokens (reserved placeholder slots excluded) are put in
categories and counted by state after removing the weight-decay shrink:
  updated (>1% of numbers changed) / barely changed (<=1%) / bit-identical.
A release is 'testable' only if its decay fit reproduces >= 99% (comparability.json class
'token record'); otherwise every cell is 'not testable'. Qwen3-30B-A3B's special-token vectors
are unrelated to its base's (qwen3_vision.json replacement test) and are marked 'replaced'.
Writes results/frontier/special_grid_nondeepseek.json."""
from __future__ import annotations

import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import OUT, is_placeholder, records  # noqa: E402

RELEASES = [
    ("Qwen/Qwen3-8B-Base", "Qwen/Qwen3-8B"), ("Qwen/Qwen3-14B-Base", "Qwen/Qwen3-14B"),
    ("Qwen/Qwen3-30B-A3B-Base", "Qwen/Qwen3-30B-A3B"), ("Qwen/Qwen3.5-35B-A3B-Base", "Qwen/Qwen3.5-35B-A3B"),
    ("unsloth/Meta-Llama-3.1-8B", "unsloth/Meta-Llama-3.1-8B-Instruct"),
    ("unsloth/Meta-Llama-3.1-70B", "unsloth/Meta-Llama-3.1-70B-Instruct"),
    ("unsloth/Llama-4-Scout-17B-16E", "unsloth/Llama-4-Scout-17B-16E-Instruct"),
    ("unsloth/Llama-4-Maverick-17B-128E", "unsloth/Llama-4-Maverick-17B-128E-Instruct"),
    ("mistralai/Mistral-Nemo-Base-2407", "mistralai/Mistral-Nemo-Instruct-2407"),
    ("mistralai/Mistral-Small-24B-Base-2501", "mistralai/Mistral-Small-24B-Instruct-2501"),
    ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
    ("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-Base-BF16", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"),
    ("openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b"), ("openai/gpt-oss-120b", "openai/gpt-oss-safeguard-120b"),
    ("microsoft/phi-4", "microsoft/Phi-4-reasoning"), ("microsoft/phi-4", "microsoft/Phi-4-reasoning-plus"),
    ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-SFT"), ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Think-SFT"),
    ("moonshotai/Kimi-K2-Base", "moonshotai/Kimi-K2-Instruct"), ("moonshotai/Kimi-K2-Base", "moonshotai/Kimi-K2-Thinking"),
    ("zai-org/GLM-4.5-Base", "zai-org/GLM-4.5"), ("zai-org/GLM-4.5-Air-Base", "zai-org/GLM-4.5-Air"),
]
CATS = [("chat markers", ("im_start", "im_end", "header", "eot", "eom", "user", "assistant", "system", "start|>", "end|>", "<|start", "<|end", "<|message", "<|channel", "<|return", "<|call", "<|constrain", "turn", "bos", "eos", "[inst]", "[/inst]", "[system_prompt]", "[/system_prompt]", "begin_of", "end_of", "endoftext", "<|im_", "sep|>")),
        ("tool calls", ("tool", "function", "python", "parameter", "ipython", "<|call")),
        ("thinking", ("think",)),
        ("vision / audio", ("vision", "image", "video", "box", "quad", "object_ref", "audio", "tts", "img", "patch", "tile")),
        ("code fill-in", ("fim", "repo_name", "file_sep")),
        ]


def category(s):
    s = (s or "").lower()
    order = ["thinking", "tool calls", "vision / audio", "code fill-in", "chat markers"]
    for name in order:
        keys = dict(CATS)[name]
        if any(k in s for k in keys):
            return name
    return "other"


def main():
    recs = records()
    comp = {(x["parent"], x["child"]): x["cls"] for x in json.load(open(OUT / "comparability.json"))}
    rows = []
    found = {(r["parent"], r["child"]): r for r in recs.values()}
    for par, ch in RELEASES:
        r = found.get((par, ch))
        if r is None:
            # parent repo spelled differently in the sweep: match on child only
            cand = [x for x in recs.values() if x["child"] == ch]
            r = cand[0] if cand else None
        if r is None:
            rows.append(dict(parent=par, release=ch, error="not in sweep")); continue
        cls = comp.get((r["parent"], r["child"]))
        testable = cls == "token record"
        cells = {}
        for t in r.get("special", []):
            if is_placeholder(t["content"]):
                continue
            c = category(t["content"])
            st = {"moved": "updated", "near": "barely changed", "unmoved": "bit-identical"}[t["state"]]
            if ch == "Qwen/Qwen3-30B-A3B":
                st = "replaced"
            cells.setdefault(c, {}).setdefault(st, []).append(t["content"])
        rows.append(dict(parent=r["parent"], release=ch, comparability=cls, testable=testable,
                         fit_reproduced=r["embed"].get("P_exact_frac"),
                         cells={c: {s: len(v) for s, v in d.items()} for c, d in cells.items()},
                         tokens={c: d for c, d in cells.items()}))
        print(ch, cls, {c: {s: len(v) for s, v in d.items()} for c, d in cells.items()})
    json.dump(rows, open(OUT / "special_grid_nondeepseek.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
