#!/usr/bin/env python3
"""Fetch the data file behind the Nous Research write-up on Arete (https://nousresearch.com/arete) into the cache.

Usage: python3 scripts/arete_check/fetch_nous.py CACHE_DIR
Writes CACHE_DIR/nous_data.js (read by c_format.py, e_concat.py, f_final.py) and, in the repository, only
results/arete_check/nous_lengths.json: the summary numbers our checks use (token-length summary, per-topic
counts, steps per epoch, warm-up) with the source URL and sha256. The file itself belongs to Nous Research
and is not redistributed here.
"""
import hashlib, json, pathlib, sys, urllib.request

URL = "https://nousresearch.com/interactive/arete/data.js"
SHA256 = "51d57afa1467893dae909cc4f70c09742f118793473b16b3ee3b0727230762ca"  # as fetched 2026-11-16
ROOT = pathlib.Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    D = pathlib.Path(sys.argv[1]); D.mkdir(parents=True, exist_ok=True)
    out = D / "nous_data.js"
    if not out.exists():
        req = urllib.request.Request(URL, headers={"User-Agent": "training-forensics"})
        out.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    raw = out.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != SHA256:
        print(f"warning: {URL} changed since 2026-11-16 (sha256 {sha})", file=sys.stderr)
    s = raw.decode()
    N = json.loads(s[s.index("{"):s.rindex("}") + 1])
    derived = {"source": URL, "sha256": sha, "sha256_2026_11_16": SHA256,
               "examples": len(N["sorted_lengths"]), "tokens_total": int(sum(N["sorted_lengths"])),
               "lengths": N["lengths"], "topics": N["topics"],
               "steps_per_epoch": N["steps_per_epoch"], "warmup": N["warmup"], "steps_logged": len(N["steps"])}
    json.dump(derived, open(ROOT / "results" / "arete_check" / "nous_lengths.json", "w"), indent=1)
    print(json.dumps(derived)[:700])
