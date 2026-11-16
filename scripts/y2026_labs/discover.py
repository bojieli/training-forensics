#!/usr/bin/env python3
"""List 2026 open-weight releases from major labs (HF API, createdAt >= 2026-01-01)."""
import json, pathlib, datetime as dt
from huggingface_hub import HfApi
ROOT = pathlib.Path(__file__).resolve().parents[2]
A = HfApi(library_name="tf-sweep")
AUTH = ["deepseek-ai","Qwen","moonshotai","zai-org","MiniMaxAI","XiaomiMiMo","stepfun-ai","meituan-longcat","inclusionAI",
 "baidu","tencent","ByteDance-Seed","meta-llama","facebook","mistralai","google","microsoft","nvidia","ibm-granite","openai",
 "allenai","arcee-ai","LGAI-EXAONE","CohereLabs","ai21labs","swiss-ai","HuggingFaceTB","tiiuae","rednote-hilab","internlm",
 "PrimeIntellect","NousResearch","upstage","sarvamai","kakaocorp","naver-hyperclovax","LiquidAI","servicenow-ai","Motif-Technologies",
 "poolside","reka-ai","xai-org","apple","amazon","Salesforce","zyphra","essentialai","openbmb","OpenGVLab","jdopensource","Kwaipilot","Kwai-Kolors","bytedance-research","ant-research"]
SKIP = ("gguf","awq","gptq","fp8","int4","int8","mlx","eagle","dflash","dspark","mtp","-nvfp4","w4a16","onnx","exl2","-4bit","-8bit","embedding","reranker","speculator","draft")
out = []
for a in AUTH:
    try:
        ms = list(A.list_models(author=a, sort="created_at", direction=-1, limit=200, expand=["createdAt","downloads","pipeline_tag","safetensors","tags"]))
    except Exception as e:
        print("ERR", a, e); continue
    for m in ms:
        if not m.created_at or m.created_at < dt.datetime(2026,1,1,tzinfo=dt.timezone.utc):
            continue
        low = m.id.lower()
        if any(s in low for s in SKIP):
            continue
        st = getattr(m, "safetensors", None)
        params = getattr(st, "total", None) if st else None
        out.append(dict(id=m.id, created=m.created_at.date().isoformat(), downloads=m.downloads, pipeline=m.pipeline_tag, params=params))
json.dump(out, open(ROOT/"results"/"y2026_labs"/"discovered.json","w"), indent=1)
print(len(out))
for r in sorted(out, key=lambda r: (r["id"].split("/")[0], r["created"])):
    if (r["params"] or 0) >= 1e9 or r["params"] is None:
        print(r["created"], r["id"], r["pipeline"], (round(r["params"]/1e9,1) if r["params"] else "?"), r["downloads"])
