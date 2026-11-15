# training-forensics: common tasks
export PYTHONPATH := .

.PHONY: test grid score table public roster

test:            ## offline unit tests (CPU, ~1 min)
	python3 -m pytest -q tests/

grid:            ## controlled fine-tunes with ground truth (one GPU, ~30 min)
	scripts/controlled/grid.sh

score:           ## blind reading of every controlled run, scored against truth
	for r in runs/ctl/r* runs/ctl/s0[2-6]*; do python3 scripts/controlled/score.py $$r; done

table:
	python3 scripts/controlled/table.py results/controlled

public:          ## the four public claims (CPU + network; set HF_TOKEN)
	python3 scripts/public/claim1_pythia_lr.py && python3 scripts/public/claim1b_pythia_clean_fit.py
	python3 scripts/public/claim2_olmo2_sft.py
	python3 scripts/public/claim3_qwen25_control_tokens.py && python3 scripts/public/claim4_qwen25_adam_signature.py

roster:          ## parent->child embedding diffs over the checkpoint-forensics roster
	python3 scripts/roster/build_pairs.py && python3 scripts/roster/sweep.py && python3 scripts/roster/sweep_report.py
