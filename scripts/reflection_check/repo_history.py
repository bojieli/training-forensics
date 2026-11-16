#!/usr/bin/env python3
"""Which shards changed between the first upload and main; index.json key layout; tokenizer extras."""
import json, sys, requests
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import st
R = "mattshumer/Reflection-Llama-3.1-70B"
revs = {"dup": "276a4a0a0a", "first": "e7bfef9796", "main": None}
F = {k: st.files(R, v) for k, v in revs.items()}
out = {}
for k, fs in F.items():
    ss = {n: s["lfs"]["sha256"] for n, s in fs.items() if n.endswith(".safetensors")}
    out[k] = dict(n_files=len(fs), n_shards=len(ss), bytes=sum(fs[n]["size"] for n in ss), other=sorted(n for n in fs if n not in ss))
    print(k, out[k])
a = {n: s["lfs"]["sha256"] for n, s in F["first"].items() if n.endswith(".safetensors")}
b = {n: s["lfs"]["sha256"] for n, s in F["main"].items() if n.endswith(".safetensors")}
out["shards_same_first_vs_main"] = sorted(n for n in b if a.get(n) == b[n])
out["shards_changed_first_vs_main"] = sum(1 for n in b if n in a and a[n] != b[n])
out["sizes_first"] = sorted(set(F["first"][n]["size"] for n in a))[:5]; out["sizes_main"] = sorted(set(F["main"][n]["size"] for n in b))[:5]
print("same:", out["shards_same_first_vs_main"], "changed:", out["shards_changed_first_vs_main"])
for k, v in revs.items():
    for f in ("config.json", "tokenizer_config.json"):
        try:
            j = requests.get(st.url(R, v, f), headers=st.UA, timeout=60).json()
            if f == "config.json": print(k, f, {x: j.get(x) for x in ("vocab_size", "torch_dtype", "_name_or_path", "transformers_version", "eos_token_id")}); out[k + "_config"] = j
            else:
                ad = {i: t["content"] for i, t in j.get("added_tokens_decoder", {}).items() if int(i) >= 128256 or "reserved" not in t["content"]}
                print(k, f, {i: c for i, c in ad.items() if int(i) >= 128256 or int(i) in (128002, 128003, 128004, 128005, 128010, 128011, 128012, 128013, 128014)}); out[k + "_added"] = ad
        except Exception as e: print(k, f, "ERR", e)
try: print("origin repo:", st.api("models/sahil2801/reflect_v5_70").get("sha"))
except Exception as e: print("origin repo not accessible:", str(e)[:80])
# dtype/shape layout of both revisions
for k in ("first", "main"):
    ix = st.index(R, revs[k]); e = ix["model.embed_tokens.weight"]
    dts = {}
    for n, v in ix.items(): dts[v[1]] = dts.get(v[1], 0) + 1
    out[k + "_layout"] = dict(tensors=len(ix), dtypes=dts, embed=e[:3], lm_head=ix.get("lm_head.weight", [None] * 3)[:3])
    print(k, out[k + "_layout"])
json.dump(out, open(sys.argv[1], "w"), indent=1)
