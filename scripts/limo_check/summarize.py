#!/usr/bin/env python3
"""Step 8: results/limo_check/summary.json, assembled only from the step outputs in results/limo_check/out/."""
import json
from common import load, ROOT

f, cmp_, ra, op, tk, de, qc, ctl = (load(n) for n in ("fetch", "compare", "run_args", "optim", "tok", "deeper", "qcheck", "controls"))
p = f["pins"]
cut = de["by_cutoff"]
s = {
    "generated_by": "scripts/limo_check/run_all.sh (steps fetch, compare, run_args, optim_analyze, tok, deeper, qcheck, controls, summarize)",
    "release": f"{p['LIMO']}@{p['LIMO_REV']}",
    "optimizer_state_rev": f"{p['LIMO']}@{p['LIMO_OPTIM_REV']} ({p['OPTIM_FILE']}, deleted from main in {p['LIMO_DELETE_REV']})",
    "weights_same_optim_rev_vs_main": f["weights_same_optim_rev_vs_main"],
    "parents": cmp_["parents_of_limo"],
    "published_set": {
        "examples": tk["examples"], "lengths": tk["lengths"],
        # the three numbers people mix up: distinct tokens in the set, of which with/without change, and changed-but-absent
        "cut16384_bf16_changed": tk["variants"]["cut16384"],
        "full_bf16_changed": tk["variants"]["full"],
        "full_plus_all_hf_fields": tk["variants"]["full+all HF fields"],
    },
    "optimizer": op,
    "run": {**ra["training_args"], **{k: ra[k] for k in ("num_processes", "steps_per_epoch", "examples_per_optimizer_step", "implied_examples_per_epoch")},
            "trainer_state": ra["trainer_state"], "config": ra["config"], "source": ra["source"]},
    "total_flos_check": ra["total_flos_check"],
    "by_cutoff": cut,
    "yaml_cutoff": {"at_release_day": f["github"]["yaml_cutoff_at_first_commit"], "from_2025_02_14": f["github"]["yaml_cutoff_after_change"]},
    "no_gradient_published_tokens": de["no_gradient_published_tokens"],
    "per_example_unique_tokens": de["per_example_unique_tokens"],
    "questions": {k: qc[k] for k in ("question_tokens", "question_tokens_without_gradient", "question_only_tokens", "question_only_with_gradient",
                                     "question_only_in_one_question", "published_tokens_with_gradient_share")},
    "flagged_examples_solution_unique_with_gradient_share": qc["their_solution_unique_tokens_with_gradient_share"],
    "cjk": {"vocab_rows": qc["cjk_vocab_rows"], "rows_with_gradient": qc["cjk_rows_with_gradient"],
            "absent_cut16384_cjk": tk["absent_cut16384_cjk"], "cjk_chars_in_published_text": tk["cjk_chars_in_raw_published_text"]},
    "eval_questions_fully_covered": de["eval_questions_fully_covered"],
    "controls": ctl,
    "github": f["github"],
}
json.dump(s, open(ROOT / "results" / "limo_check" / "summary.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(s, indent=1, ensure_ascii=False)[:4000])
