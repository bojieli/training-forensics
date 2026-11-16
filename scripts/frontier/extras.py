#!/usr/bin/env python3
"""Build the negative-control and null comparisons for the classifier (review B1, M4).

siblings: two children of the same parent in the sweep, neither the parent of the other
          (not parent -> child: a "token record" or lineage reading here is a false positive)
null:     same tokenizer, separately trained models (Pythia seeds / deduped, Qwen2 vs Qwen2.5):
          calibrates the "changed throughout, consistent with the base plus decay" threshold.
Writes results/frontier/negctl_pairs.json (records in sweep format with _fit = True)."""
import json, sys, pathlib, itertools
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "frontier"))
from recount import all_records, EMB_NAMES
from trainforensics import remote

def find_embed(repo, revision=None):
    """(file, tensor, dtype, shape) of the token embedding. Falls back to scanning every
    shard header when the index points at the wrong shard (AM-Thinking-v1)."""
    ix = remote.index(repo, revision)
    names = sorted([n for n in ix if any(n.endswith(s) for s in EMB_NAMES) and "vision" not in n and "visual" not in n
                    and "mtp" not in n and "position" not in n],
                   key=lambda n: ("language_model" not in n, len(n)))
    for n in names:
        try:
            dt, sh = remote.open_file(repo, ix[n], revision).info(n)
            return ix[n], n, dt, list(sh)
        except KeyError:
            for fn in sorted(set(ix.values())):
                f = remote.open_file(repo, fn, revision)
                if n in f.header:
                    dt, sh = f.info(n)
                    return fn, n, dt, list(sh)
    raise KeyError(repo)


def meta(repo):
    _, _, dt, sh = find_embed(repo)
    return dt, sh

def main():
    recs = all_records()
    par = {}
    edges = set()
    for r in recs.values():
        if r.get("tied"):
            continue
        edges.add((r["parent"], r["child"]))
        par.setdefault(r["parent"], []).append(r)
    out = []
    for p, ch in par.items():
        ch = sorted({c["child"]: c for c in ch}.values(), key=lambda c: c["child"])
        for a, b in itertools.combinations(ch, 2):
            if (a["child"], b["child"]) in edges or (b["child"], a["child"]) in edges:
                continue
            if a["shape_child"][1] != b["shape_child"][1] or a["dtype_child"] != b["dtype_child"]:
                continue
            out.append(dict(key=f"SIB {a['child']} ~ {b['child']}", kind="sibling", common_parent=p,
                            parent=a["child"], child=b["child"], parent_revision=None, tied=False,
                            dtype_parent=a["dtype_child"], dtype_child=b["dtype_child"],
                            shape_parent=[min(a["shape_child"][0], b["shape_child"][0]), a["shape_child"][1]],
                            shape_child=[min(a["shape_child"][0], b["shape_child"][0]), b["shape_child"][1]],
                            _real_shapes=(a["shape_child"], b["shape_child"]),
                            emb_gb=(a.get("emb_gb") or 0) + (b.get("emb_gb") or 0), embed={}, _fit=True))
    NULL = []
    for s in ("70m", "160m", "410m", "1b", "1.4b", "2.8b"):
        NULL.append((f"EleutherAI/pythia-{s}", f"EleutherAI/pythia-{s}-deduped"))
    for s in ("14m", "31m", "70m", "160m", "410m"):
        NULL += [(f"EleutherAI/pythia-{s}", f"EleutherAI/pythia-{s}-seed1"),
                 (f"EleutherAI/pythia-{s}-seed2", f"EleutherAI/pythia-{s}-seed3"),
                 (f"EleutherAI/pythia-{s}-seed4", f"EleutherAI/pythia-{s}-seed5")]
    for s in ("0.5B", "1.5B", "7B", "72B"):
        NULL.append((f"Qwen/Qwen2-{s}", f"Qwen/Qwen2.5-{s}"))
    for a, b in NULL:
        try:
            da, sa = meta(a); db, sb = meta(b)
        except Exception as e:
            print("skip", a, b, type(e).__name__, e); continue
        if sa[1] != sb[1]:
            print("skip shape", a, b); continue
        n = min(sa[0], sb[0])
        out.append(dict(key=f"NULL {a} ~ {b}", kind="null", parent=a, child=b, parent_revision=None, tied=False,
                        dtype_parent=da, dtype_child=db, shape_parent=sa, shape_child=sb, embed={}, _fit=True))
    json.dump(out, open(ROOT / "results" / "frontier" / "negctl_pairs.json", "w"), indent=1)
    from collections import Counter
    print(len(out), Counter(o["kind"] for o in out))


if __name__ == "__main__":
    main()
