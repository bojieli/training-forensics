#!/usr/bin/env python3
"""Pure-bf16 runs: the blind schedule fit needs single-step moves of measurable size,
which bf16 parameters quantise to 1-2 steps, so score_stack.py cannot read a recipe.
This reads what survives directly: which token vectors differ from the bf16 start,
scored against the tokens the run actually trained on (truth.json, scoring only),
and how far the changed vectors moved in 16-bit steps.
  python3 scripts/stacks/purebf16_tokens.py runs/stacks/t14_purebf16_hi [...]"""
import json, pathlib, sys
import torch
from safetensors.torch import load_file
ROOT = pathlib.Path(__file__).resolve().parents[2]
out = {}
for r in sys.argv[1:]:
    run = pathlib.Path(r)
    e = load_file(str(run / "emb.safetensors"))
    W0 = e["W0"].bfloat16(); W1 = e["W1_param"]
    changed = ~(W0 == W1).all(1)
    tr = json.load(open(run / "truth.json"))
    sys.path.insert(0, str(ROOT)); from trainforensics.truth import Run
    seen_mask = torch.from_numpy(Run(run).seen_mask()[: W0.shape[0]])
    seen = range(int(seen_mask.sum()))
    tp = int((changed & seen_mask).sum()); fp = int((changed & ~seen_mask).sum()); fn = int((~changed & seen_mask).sum())
    steps = (W0.view(torch.int16).int() - W1.view(torch.int16).int()).abs()
    ch = steps[seen_mask & changed]
    lr = tr["lr"]; wd = tr["args"]["wd"]
    # decoupled decay per step vs half a bf16 step (2^-9 relative): does decay survive at all?
    unseen_equal = int((~changed & ~seen_mask).sum()); unseen = int((~seen_mask).sum())
    out[run.name] = dict(peak_lr=max(lr), wd=wd, max_lr_x_wd=max(lr) * wd, half_bf16_step_rel=2 ** -9,
                         seen=len(seen), changed=int(changed.sum()), tp=tp, fp=fp, fn=fn,
                         unseen_bit_identical=unseen_equal, unseen=unseen,
                         changed_coords_frac_median=float((steps[seen_mask & changed] > 0).float().mean(1).median()) if tp else None,
                         rel_change_median=float(((W1.float() - W0.float()).norm(dim=1) / W0.float().norm(dim=1))[seen_mask & changed].median()) if tp else None,
                         recipe_readable=False,
                         note="schedule fit finds no measurable single-step moves: bf16 parameters quantise each update to whole 16-bit steps")
    print(run.name, json.dumps(out[run.name]))
p = ROOT / "results" / "stacks" / "purebf16.json"
old = json.load(open(p)) if p.exists() else {}
old.update(out); json.dump(old, open(p, "w"), indent=1)
