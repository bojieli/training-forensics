# Fix-D: reproducibility (technical review M11)

Commits: paths (70a55e4), Makefile/deps/lock (b5ddf09), pins + REPRODUCE.md (this log's commit).

Done
- `trainforensics/paths.py`: `CKPTF_REPO` (sibling checkpoint-forensics checkout) and `TF_CACHE` (root of all
  download caches) replace every `Path.home()/...` in scripts/ except scripts/limo_check (Fix-B rewrites it) and
  three files that were mid-edit by other lanes (scripts/gated/run_gated2.py, scripts/published/ww_compare.py,
  ww_fetch.py). Defaults keep the existing ~/.cache layout, so no cache is invalidated.
- Makefile: one target per report; `make all-offline` regenerates every generated report. Verified in a clean
  worktree under `unshare -rn` with empty caches: byte-identical except results/roster/SWEEP.md, which was stale
  against sweep.jsonl (Fork R patched the V4-Flash and Trinity-Large fits without regenerating it) and is
  regenerated. `make score` loops over run directories only (the old glob matched .log files) and includes the
  gaps swap/membership runs, m01 and the Qwen runs. `make revisions` re-pins.
- scripts/stacks/report.py kept the status of runs whose logs live in runs/ ("stopped mid-run") only on the
  machine that ran them; it now falls back to the committed status.json.
- pyproject: scipy, tokenizers, pyyaml added; extras audits, figures, quant, stacks, llamafactory, test, all.
  requirements-lock.txt pins the versions used.
- revisions.json (scripts/revisions_manifest.py): 1,725 repos (1,710 from result fields + 15 training sets),
  1,717 pinned, 8 no longer resolve. trainforensics.remote reads pins when TF_PINNED=1 (opt-in, because the cache
  key contains the revision). `paths.revision(repo)` for scripts that call hf_hub_download directly; not yet
  threaded through those scripts.
- Literals made computed: DeepScaleR token count (scripts/fp32scan/report.py), the DeepMount00 Ita round-trip
  count (scripts/quant_merge/report.py). Reports regenerate identically.

Left as literals (sources exist but the scripts belong to other lanes, or the number comes from a log not JSON)
- Hermes 2,739, MemT 1,261, Reflection section, LIMO summary: Fix-A / Fix-B.
- scripts/large/report.py: Kimi-Dev 2,171; Tulu RL counts; Nemotron-Ultra 671 / 0.765-0.769%; r1-1776 20,876;
  405B 2,101,346,304; Skywork 7, V3->V3.1 429 (file shared with Fix-A's Hermes edit).
- scripts/y2026_ft/report.py: Arete counts (from results/arete_check/*.log, not JSON).
- scripts/series/report.py: Viking 8 of 9 / 3.7%, Amber 259, OLMo 818 (in results/series/verify.json).
- scripts/gaps/gaps_report.py: old v1 swap flags 26/50/87/367; two-epoch 94% / 99.6% / 95% (the Qwen lane owns it).
- scripts/tokens/report.py: Thai 1,578 of 2,549 (method example).
- scripts/stacks/report.py: prose describing t08 / t13.
