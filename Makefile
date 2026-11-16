# training-forensics: common tasks. See REPRODUCE.md for what each target needs.
#
# Environment (all optional):
#   CKPTF_REPO  checkout of github.com/bojieli/checkpoint-forensics (default ~/checkpoint-forensics);
#               needed by roster / coverage / tokens / frontier_ds (fingerprint cards)
#   TF_CACHE    root of every download cache (default: the historical ~/.cache/<name> dirs)
#   CKPTF_CACHE embedding-tensor cache (default $TF_CACHE/trainforensics/emb)
#   HF_TOKEN    only for gated repos (scripts/gated)
#   TF_PINNED=1 fetch the commits pinned in revisions.json instead of each repo's main branch
#
# Run order for a full rebuild from scratch:
#   1. make test
#   2. make public                    (network, CPU)
#   3. make grid score                (one GPU)          -> results/controlled
#   4. make roster                    (network, CPU; needs CKPTF_REPO)
#   5. lane pipelines (network, CPU; see REPRODUCE.md), each ending in its report target below
#   6. make all-offline               regenerate every report from the committed JSON
export PYTHONPATH := .
PY ?= python3
OFFLINE := HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

.PHONY: test grid score table public roster all-offline revisions \
        controlled gaps stacks published tokens quant_merge series frontier frontier_ds frontier_x \
        y2026_labs y2026_ft large gated fp32scan limo_check arete_check reflection_check \
        roster-report coverage figures

test:            ## offline unit tests (CPU, ~1 min)
	$(PY) -m pytest -q tests/

grid:            ## controlled fine-tunes with ground truth (one GPU, ~30 min)
	scripts/controlled/grid.sh

score:           ## blind reading of every controlled run on this machine, scored against truth (one GPU or CPU)
	for r in runs/ctl/r[0-9][0-9]_*/ runs/ctl/s0[2-6]_*/ runs/ctl/q7_*/; do \
	  [ -d "$$r" ] && $(PY) scripts/controlled/score.py $${r%/} --cache || true; done
	for r in runs/gaps/g0[1-5]_*/ runs/ctl/m01/; do \
	  [ -d "$$r" ] && $(PY) scripts/controlled/score.py $${r%/} --cache --outdir results/gaps/scored || true; done

table:
	$(PY) scripts/controlled/table.py results/controlled

public:          ## the four public claims (CPU + network; set HF_TOKEN)
	$(PY) scripts/public/claim1_pythia_lr.py && $(PY) scripts/public/claim1b_pythia_clean_fit.py
	$(PY) scripts/public/claim2_olmo2_sft.py
	$(PY) scripts/public/claim3_qwen25_control_tokens.py && $(PY) scripts/public/claim4_qwen25_adam_signature.py

revisions:       ## re-pin revisions.json (network; HF_TOKEN for gated repos). --retry keeps existing pins
	$(PY) scripts/revisions_manifest.py . --retry

roster:          ## parent->child embedding diffs over the checkpoint-forensics roster (network; needs CKPTF_REPO)
	$(PY) scripts/roster/build_pairs.py && $(PY) scripts/roster/sweep.py && $(MAKE) roster-report

# ---- report targets: regenerate one results/<lane>/*.md from committed JSON (no network, no GPU) ----

controlled:      ## results/controlled/CONTROLLED.md
	$(OFFLINE) $(PY) scripts/controlled/report.py
gaps:            ## results/gaps/GAPS.md
	$(OFFLINE) $(PY) scripts/gaps/gaps_report.py
stacks:          ## results/stacks/STACKS.md + status.json
	$(OFFLINE) $(PY) scripts/stacks/report.py
published:       ## results/published/PUBLISHED.md and PUBLISHED_LIMO.md
	$(OFFLINE) $(PY) scripts/published/report.py
	$(OFFLINE) $(PY) scripts/published/report_limo.py
tokens:          ## results/tokens/TOKENS.md
	$(OFFLINE) $(PY) scripts/tokens/report.py
quant_merge:     ## results/quant_merge/QUANT_MERGE.md
	$(OFFLINE) $(PY) scripts/quant_merge/report.py
series:          ## results/series/SERIES.md
	$(OFFLINE) $(PY) scripts/series/report.py
frontier:        ## results/frontier/{comparability,decay_ranges}.json and fig/*.json (FRONTIER.md is hand-written from these)
	$(OFFLINE) $(PY) scripts/frontier/comparability.py
	$(OFFLINE) $(PY) scripts/frontier/figures.py
frontier_ds:     ## results/frontier_ds/DEEPSEEK.md and figure (b)
	$(OFFLINE) $(PY) scripts/frontier_ds/ds_report.py
	$(OFFLINE) $(PY) scripts/frontier_ds/ds_fig.py
frontier_x:      ## results/frontier_x/FRONTIER_X.md
	$(OFFLINE) $(PY) scripts/frontier_x/report.py
y2026_labs:      ## results/y2026_labs/LABS2026.md
	$(OFFLINE) $(PY) scripts/y2026_labs/report.py
y2026_ft:        ## results/y2026_ft/FT2026.md
	$(OFFLINE) $(PY) scripts/y2026_ft/report.py
large:           ## results/large/LARGE.md
	$(OFFLINE) $(PY) scripts/large/report.py
gated:           ## results/gated/GATED.md
	$(OFFLINE) $(PY) scripts/gated/gated_report.py
fp32scan:        ## results/fp32scan/FP32SCAN.md and FP32SCAN_REFLECTION.md
	$(OFFLINE) $(PY) scripts/fp32scan/report.py
	$(OFFLINE) $(PY) scripts/fp32scan/report_reflection.py
reflection_check: fp32scan   ## its write-up is FP32SCAN_REFLECTION.md (raw checks: scripts/reflection_check, network)
limo_check: published        ## its write-up is PUBLISHED_LIMO.md (raw checks: scripts/limo_check, network)
arete_check: y2026_ft        ## its write-up is in FT2026.md (raw checks: scripts/arete_check, network)
roster-report:   ## results/roster/SWEEP.md (+ exact P-interval table), PRETRAIN_DECAY.md (needs CKPTF_REPO cards)
	$(OFFLINE) $(PY) scripts/roster/sweep_report.py
	$(OFFLINE) $(PY) scripts/roster/p_table.py
	$(OFFLINE) $(PY) scripts/roster/pretrain_decay.py
coverage:        ## results/roster/COVERAGE.md (needs CKPTF_REPO results/summary.json)
	$(OFFLINE) $(PY) scripts/roster/coverage.py
figures:         ## article figures in results/figures (matplotlib)
	@if [ -f results/figures/make_figures.py ]; then $(OFFLINE) $(PY) results/figures/make_figures.py; fi
	@if [ -f results/figures/make_fig0_cover.py ]; then $(OFFLINE) $(PY) results/figures/make_fig0_cover.py; fi

all-offline: roster-report coverage controlled gaps stacks published tokens quant_merge series \
             frontier frontier_ds frontier_x y2026_labs y2026_ft large gated fp32scan  ## every report, offline
	@git status --short -- results | grep -v '^??' || echo "all-offline: reports identical to the committed files"
