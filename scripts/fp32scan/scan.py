import json, sys
from huggingface_hub import HfApi
api = HfApi()
out = []
seen = 0
for m in api.list_models(pipeline_tag="text-generation", sort=sys.argv[2], limit=int(sys.argv[3]),
                         expand=["safetensors", "cardData", "downloads", "likes", "tags", "library_name"]):
    seen += 1
    st = getattr(m, "safetensors", None)
    if not st or not st.parameters:
        continue
    p = st.parameters
    tot = sum(p.values())
    f32 = p.get("F32", 0)
    if tot == 0 or f32 / tot < 0.95:
        continue
    cd = m.card_data.to_dict() if m.card_data else {}
    base = cd.get("base_model")
    tags = m.tags or []
    if not base:
        base = [t.split(":", 1)[1] for t in tags if t.startswith("base_model:") and t.count(":") == 1]
    ds = cd.get("datasets") or [t.split(":", 1)[1] for t in tags if t.startswith("dataset:")]
    out.append(dict(repo=m.id, downloads=m.downloads, likes=m.likes, params=tot, base=base, datasets=ds,
                    adapter=("peft" in tags)))
print(seen, len(out), file=sys.stderr)
json.dump(out, open(sys.argv[1], "w"), indent=1)
