"""Build revisions.json: every Hugging Face repo the committed results depend on, pinned to a commit sha.

Repo ids are collected from structured fields of results/**/*.json(l). Each is resolved with the
Hub API (model first, then dataset). Where a result file recorded the revision it read
(repo@rev), that revision is kept as `recorded`.
"""
import concurrent.futures as cf, datetime, json, os, pathlib, re, sys
from huggingface_hub import HfApi
ROOT = pathlib.Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "."))
FIELDS = {"parent", "child", "repo", "base", "model", "base_model", "parent_repo", "child_repo", "candidate",
          "reference", "ref", "dataset", "data_repo", "base_repo", "release", "src_repo", "card_base", "best_parent"}
RID = re.compile(r"^[A-Za-z0-9][\w.\-]*/[A-Za-z0-9][\w.\-]*$")
found, recorded = {}, {}


def add(v, src):
    if not isinstance(v, str):
        return
    v = v.strip()
    rev = None
    if "@" in v:
        v, rev = v.split("@", 1)
        rev = rev.strip() or None
    if RID.match(v) and not v.endswith((".json", ".md", ".py", ".jsonl", ".safetensors", ".npz", ".png")) \
            and not v.startswith(("results/", "scripts/", "runs/")):
        found.setdefault(v, set()).add(src)
        if rev and re.fullmatch(r"[0-9a-f]{7,40}", rev):
            recorded.setdefault(v, set()).add(rev)


def walk(o, src):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in FIELDS:
                if isinstance(v, list):
                    for x in v:
                        add(x, src)
                else:
                    add(v, src)
            if k == "key" and isinstance(v, str) and "->" in v:
                for part in v.split("->"):
                    add(part.strip(), src)
            if k == "candidates" and isinstance(v, dict):
                for kk in v:
                    add(kk, src)
            walk(v, src)
    elif isinstance(o, list):
        for x in o:
            walk(x, src)


for p in sorted(ROOT.glob("results/**/*")):
    if p.suffix not in (".json", ".jsonl") or p.stat().st_size > 60e6:
        continue
    src = str(p.relative_to(ROOT).parent)
    try:
        if p.suffix == ".jsonl":
            for line in open(p):
                try:
                    walk(json.loads(line), src)
                except Exception:
                    pass
        else:
            walk(json.load(open(p)), src)
    except Exception:
        pass

# Training sets read by the published-data audits, controls and membership runs (named in code, not in result
# fields). Stored under "dataset:<id>" because several share an id with a model repo (GAIR/LIMO).
DATASETS = ["allenai/tulu-3-sft-mixture", "allenai/llama-3.1-tulu-3-405b-preference-mixture",
            "allenai/tulu-3-sft-olmo-2-mixture", "GAIR/LIMO", "GAIR/LIMO-v2", "simplescaling/s1K",
            "simplescaling/s1K-1.1", "simplescaling/s1K_tokenized", "simplescaling/s1K-1.1_tokenized",
            "NovaSky-AI/Sky-T1_data_17k", "bespokelabs/Bespoke-Stratos-17k", "open-thoughts/OpenThoughts-114k",
            "NousResearch/arete-data-pairs", "Qwen/WebWorldData", "tatsu-lab/alpaca"]
for dsid in DATASETS:
    found.setdefault("dataset:" + dsid, set()).add("scripts (training data)")

api = HfApi(token=os.environ.get("HF_TOKEN"))


def resolve(r):
    import time
    from huggingface_hub.utils import RepositoryNotFoundError
    err = None
    kinds = ("model", "dataset")
    if r.startswith("dataset:"):
        r0, kinds = r, ("dataset",)
        r = r[len("dataset:"):]
    else:
        r0 = r
    for rt in kinds:
        for attempt in range(6):
            try:
                i = api.repo_info(r, repo_type=rt)
                return r0, dict(type=rt, sha=i.sha, gated=bool(getattr(i, "gated", False)))
            except RepositoryNotFoundError as e:
                err = type(e).__name__
                break
            except Exception as e:  # rate limit / transient: back off and retry
                err = type(e).__name__
                time.sleep(5 * (attempt + 1))
    return r0, dict(type=None, sha=None, error=err)


out = {}
prev = {}
if "--retry" in sys.argv and (ROOT / "revisions.json").exists():
    prev = json.load(open(ROOT / "revisions.json"))["repos"]
todo = [r for r in sorted(found) if not prev.get(r, {}).get("sha")]
for r in found:
    if prev.get(r, {}).get("sha"):
        out[r] = {k: v for k, v in prev[r].items() if k not in ("used_by", "recorded")}
with cf.ThreadPoolExecutor(4) as ex:
    for r, info in ex.map(resolve, todo):
        out[r] = info
for r in sorted(found):
    info = out[r]
    info["used_by"] = sorted(found[r])
    if r in recorded:
        info["recorded"] = sorted(recorded[r])
ok = sum(1 for v in out.values() if v.get("sha"))
doc = {"pinned_at": datetime.date.today().isoformat(),
       "note": ("sha = the repo's main commit when pinned; results computed earlier read main at that time. "
                "`recorded` = revisions a result file names explicitly. Repos with sha null no longer resolve "
                "(deleted, renamed or private). Fetch exactly these revisions with TF_PINNED=1."),
       "repos": dict(sorted(out.items()))}
json.dump(doc, open(ROOT / "revisions.json", "w"), indent=1, sort_keys=False)
print(len(found), "repos;", ok, "resolved;", len(found) - ok, "unresolved")
for r, v in out.items():
    if not v.get("sha"):
        print("  unresolved:", r, v.get("error"), v["used_by"][:2])
