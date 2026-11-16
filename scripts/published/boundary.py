#!/usr/bin/env python3
"""How many 'updated but absent from the data' tokens can tokenization context explain?
Upper bound: tokens produced by any of
  - each message tokenized alone, with a leading space, with a leading newline, with surrounding
    template text, whole-conversation rendering (with and without each system prompt);
  - joins between samples (packing / concatenation without separators): last 64 chars of one
    sample + first 64 chars of another, for all ordered pairs (sampled when there are too many),
    and with "<|im_end|>\n", "\n", " " and "" as separators;
  - the default system prompts of LLaMA-Factory (two versions), TRL and Qwen.
Writes results/published/boundary_<child>.json."""
import ast, json, pathlib, random, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "published"))
from lf_exact import load_rows  # noqa
from transformers import AutoTokenizer
OUT = ROOT / "results" / "published"
SYSTEMS = ["You are a helpful assistant.", "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", ""]


def main():
    child, data, base = sys.argv[1], sys.argv[2], sys.argv[3]
    r = json.load(open(OUT / f"{child.replace('/', '__')}__{data}.json"))
    m = r["variants"]["main"]
    tok = AutoTokenizer.from_pretrained(base)
    rows = load_rows(data)
    counts = json.load(open(OUT / f"data_{data}.json"))["counts"]
    allin = set()
    for k, c in counts.items():
        allin.update(int(x) for x in c)
    # recompute the extra set from the stored comparison is not possible (examples are truncated), so
    # recompute changed from the embeddings
    import torch
    sys.path.insert(0, str(ROOT))
    from trainforensics import remote
    from compare import frac_changed, EMB, PUB
    remote.CACHE = EMB
    W0 = remote.tensor(base, "model.embed_tokens.weight")
    remote.CACHE = PUB
    W1 = remote.tensor(child, "model.embed_tokens.weight")
    n = min(len(W0), len(W1))
    f = frac_changed(W0[:n], W1[:n], r["P"])
    main_set = set(int(x) for x in counts["main"])
    extra = set(np.nonzero(f > 0)[0].tolist()) - main_set
    extra = {i for i in extra if i < len(tok)}
    expl = {}
    ctx = set()
    # 1 every counted variant of the data
    expl["other counted variants (system prompt, cutoff, final position)"] = len(extra & allin)
    # 2 per-message context variants
    texts = []
    for system, msgs in rows:
        for s in [system] + SYSTEMS:
            if s:
                texts.append(s); texts.append(" " + s)
        for mm in msgs:
            c = mm["content"]
            texts += [c, " " + c, "\n" + c, c + "\n", "\n" + c + "\n",
                      f"<|im_start|>{mm['role']}\n{c}<|im_end|>\n"]
        texts.append("".join(f"<|im_start|>{mm['role']}\n{mm['content']}<|im_end|>\n" for mm in msgs))
    for i in range(0, len(texts), 2000):
        for ids in tok(texts[i:i + 2000], add_special_tokens=True)["input_ids"]:
            ctx.update(ids)
    expl["+ message context variants"] = len(extra & (allin | ctx))
    # 3 joins between samples
    heads = [(msgs[0]["content"])[:64] for _, msgs in rows] + [(s or "")[:64] for s, _ in rows]
    tails = [(msgs[-1]["content"])[-64:] for _, msgs in rows]
    pairs = [(a, b) for a in range(len(tails)) for b in range(len(heads))]
    random.Random(0).shuffle(pairs)
    pairs = pairs[:60000]
    join = set()
    for sep in ("", "\n", " ", "<|im_end|>\n", "<|endoftext|>"):
        strs = [tails[a] + sep + heads[b] for a, b in pairs]
        for i in range(0, len(strs), 5000):
            for ids in tok(strs[i:i + 5000], add_special_tokens=False)["input_ids"]:
                join.update(ids)
    expl["+ joins between samples (packing)"] = len(extra & (allin | ctx | join))
    left = sorted(extra - allin - ctx - join)
    res = dict(child=child, data=data, extra=len(extra), explained_cumulative=expl, unexplained=len(left),
               pairs_tested=len(pairs),
               unexplained_examples=[(i, tok.convert_ids_to_tokens(i), tok.decode([i])) for i in left[:80]])

    def cat(i):
        s = tok.decode([i])
        if any("一" <= ch <= "鿿" or "぀" <= ch <= "ヿ" for ch in s):
            return "CJK"
        if "�" in s:
            return "partial UTF-8 bytes"
        if any(ord(ch) > 127 for ch in s):
            return "other non-ASCII"
        return "ASCII"
    from collections import Counter
    res["unexplained_categories"] = dict(Counter(cat(i) for i in left))
    json.dump(res, open(OUT / f"boundary_{child.replace('/', '__')}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "unexplained_examples"}, ensure_ascii=False))
    print([x[2] for x in res["unexplained_examples"][:40]])


if __name__ == "__main__":
    main()
