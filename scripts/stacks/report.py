#!/usr/bin/env python3
"""results/stacks/*.json -> results/stacks/STACKS.md (partial: stopped early)."""
from __future__ import annotations

import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results" / "stacks"
RUNS = ROOT / "runs" / "stacks"

PLAN = [
    ("t01_clip_off", "Pythia-160m, no gradient clipping (control)"),
    ("t02_clip_on", "Pythia-160m, gradient clipping at 1.0 (clips every step: norms 20-30)"),
    ("t03_clip_tight", "Pythia-160m, clipping at 0.01 (tiny gradients, Adam eps matters)"),
    ("t04_ga4", "Pythia-160m, gradient accumulation 4 x 2 documents"),
    ("t05_ddp2", "Pythia-160m, 2 data-parallel ranks (gloo, one GPU)"),
    ("t06_ddp4_ga2", "Pythia-160m, 4 ranks x accumulation 2"),
    ("t16_ddp2_ga4", "Pythia-160m, 2 data-parallel ranks x accumulation 4 (gloo, one GPU)"),
    ("t07_fsdp_410m", "Pythia-410m, FSDP full_shard, bf16 mixed precision"),
    ("t08_ds_z2_dsopt", "DeepSpeed ZeRO-2, bf16, DeepSpeed FusedAdam"),
    ("t09_ds_z2_torchopt", "DeepSpeed ZeRO-2, bf16, torch AdamW"),
    ("t10_ds_z3_offload", "DeepSpeed ZeRO-3, optimizer on CPU (CPUAdam)"),
    ("t11_muon_hidden", "Muon on hidden matrices, AdamW on embeddings"),
    ("t12_muon_all", "Muon on every matrix including the embedding"),
    ("t13_pack", "packed 256-token sequences"),
    ("t14_purebf16_hi", "16-bit weights and optimizer, no 32-bit master copy, lr 2e-4"),
    ("t15_purebf16_lo", "same, lr 2e-5"),
    ("q7_ds_z3_nvme", "Qwen2.5-7B full fine-tune, ZeRO-3, optimizer state on NVMe"),
]


def status(name):
    log = RUNS / f"{name}.log"
    js = R / f"{name}.fp32.json"
    if js.exists():
        return "scored"
    pb = R / "purebf16.json"
    if pb.exists() and name in json.load(open(pb)):
        return "scored (tokens only)"
    if log.exists() and "\ndone " in "\n" + log.read_text(errors="ignore"):
        return "trained, not scored"
    if log.exists():
        return "stopped mid-run"
    return "not run"


def main():
    lines = ["# Training stacks: does the record survive real fine-tuning setups?", "",
             "**Partial.** Several variants were not run; only the rows marked *scored* carry",
             "results. The DeepSpeed ZeRO-2 run (t08) was trained with DeepSpeed itself, not replayed with its formulas. Everything ran on one RTX PRO 6000 (96 GB): no multi-GPU box was",
             "available, so several ranks would have shared one GPU over gloo, and the 7B run would have kept its",
             "optimizer state on NVMe.", "",
             "Method: each run fine-tunes with full ground truth (`scripts/stacks/finetune_stack.py`); the unchanged",
             "blind reader (`scripts/controlled/score.py`, wrapped by `scripts/stacks/score_stack.py`) recovers which",
             "vocabulary tokens were trained, the learning-rate schedule, the shuffle seed, and replays the token",
             "vectors that were never updated. Counts below are counts of distinct vocabulary tokens.", "",
             "The wrapper adds two stack-aware steps: the starting point may be the base model cast to 16 bits",
             "(as DeepSpeed does), and weight decay is replayed with three kernels (torch `p*(1-lr*wd)`, DeepSpeed",
             "FusedAdam `fma(-lr, wd*p, p)`, DeepSpeed CPUAdam `fma(p, -lr*wd, p)`), keeping the one that is bit-exact.", "",
             "| run | setup | status | tokens trained (recovered / true, false positives) | schedule recovered exactly | seed | never-updated vectors reproduced bit-exactly |",
             "|---|---|---|---|---|---|---|"]
    out = {}
    for name, desc in PLAN:
        st = status(name)
        row = [name, desc, st, "-", "-", "-", "-"]
        js = R / f"{name}.fp32.json"
        if js.exists():
            d = json.load(open(js))
            out[name] = d
            t = d["tokens"]
            tr = d.get("tokens_replay")
            tok = f"{t['tp']:,} / {t['seen']:,}" + (f", {t['fp']} fp" if t["fp"] else "")
            if tr:
                tok += f"; via replay {tr['tp']:,} / {t['seen']:,}" + (f", {tr['fp']} fp" if tr["fp"] else "")
            ok = d["recipe_ok"]
            rr = d["recipe_read"]
            exact = bool(ok.get("schedule_exact"))
            row[3] = tok
            row[4] = "yes" if exact else f"no ({rr['kind']}, {rr['N']} steps, warm-up {rr['warmup']}, peak {rr['peak_snap']:.3g})"
            sd = d.get("seed")
            row[5] = "-" if not sd else ("recovered" if sd["found"] == sd["true"] else "missed")
            rp = d.get("replay")
            if rp:
                form = d.get("decay_forms", {}).get("best", "mul")
                row[6] = f"{rp['bit_exact']:,} of {rp['weights']:,} weights ({form} kernel)"
        pbj = R / "purebf16.json"
        if not js.exists() and pbj.exists() and name in json.load(open(pbj)):
            p = json.load(open(pbj))[name]
            row[3] = f"{p['tp']:,} / {p['seen']:,}" + (f", {p['fp']} fp" if p["fp"] else "") + " (bit-identity test)"
            row[4] = "not readable"
            row[5] = "not readable"
            row[6] = f"weight decay rounds away: all {p['unseen_bit_identical']:,} never-trained vectors bit-identical"
        lines.append("| " + " | ".join(row) + " |")
    gn = {k: v.get("grad_norm_logged") for k, v in out.items()}
    lines += ["", "Gradient norms logged during training (clipping threshold in brackets): " +
              "; ".join(f"{k}: {g['min']:.3g}-{g['max']:.3g} [{g['clip']}]" for k, g in gn.items() if g and g.get("n")), "",
              "Note: the 24 earlier controlled runs (`results/controlled/`) already used the Hugging Face default",
              "clipping at 1.0 with gradient norms of 13-19, so clipping was active on every step there too.", "",
              "t03 (clipping at 0.01) fails, as expected: clipped gradients are so small that Adam's epsilon dominates",
              "the update, so a token seen once no longer moves by about 31.8 x the learning rate and the schedule cannot",
              "be estimated. Real fine-tunes clip at about 1.0, as t02 does.", "",
              "t08 is trained with DeepSpeed ZeRO-2 and its FusedAdam: the base is cast to 16 bits before the 32-bit",
              "master copy is built, and weight decay rounds as fma(-lr, wd*p, p). With that starting point and kernel",
              "the never-updated vectors replay bit-exactly; with the torch formula they do not. Its warm-up was first",
              "estimated one step long (21 vs 20); trying warm-up +-3 and steps +-1 against the bit-exact replay",
              "(`refine_by_replay` in score_stack.py) fixed it, and the seed and document steps then followed."]
    pk = R / "pack_lasttoken.json"
    if pk.exists():
        q = json.load(open(pk))
        lines += ["", "## Packed sequences (t13)", "",
                  "Documents joined with an end-of-text token into 256-token chunks. The reader recovers the schedule, the",
                  "seed (as the chunk order) and the bit-exact replay unchanged. The last-token rule changes: without packing a",
                  "document's final token is the last input of its sequence and gets no gradient, so the published-data audits",
                  f"drop final positions. Packed, it is followed by the end-of-text token and is trained: {q['of_which_updated']} of the",
                  f"{q['tokens_only_in_final_position']} tokens that occur in the training documents only in final position were updated, and the",
                  "end-of-text token itself was " + ("updated." if q["eos_token_updated"] else "not updated.") + " Only each chunk's last token gets no gradient.",
                  "For an audit: if tokens that occur only at the ends of published documents were updated, the run was packed",
                  "(or appended an end token after each document); if they are all bit-identical, it was not. Either way the",
                  "token-set comparison stays exact once the matching rule is applied (`scripts/stacks/pack_lasttoken.py`).",
                  "The reader's direct reading misses exactly these 68 tokens; its replay reading, which does not assume the",
                  "rule, finds all of them."]
    pb = R / "purebf16.json"
    if pb.exists():
        P = json.load(open(pb))
        lines += ["", "## Pure 16-bit training (t14, t15)", "",
                  "16-bit weights and optimizer state with no 32-bit master copy. Each update is rounded to whole 16-bit steps,",
                  "so a token seen once no longer moves by a measurable 31.8 x the learning rate: the schedule fit finds nothing",
                  "to fit, and the recipe, seed and document steps are not recoverable. Weight decay (lr x wd at most "
                  + ", ".join(f"{v['max_lr_x_wd']:.0e}" for v in P.values()) + " per step) is below half a 16-bit step",
                  "(about 0.2%), so it rounds away entirely and every never-trained vector stays bit-identical. The token list is",
                  "therefore read by a plain bit-identity test, with no replay (`scripts/stacks/purebf16_tokens.py`):", "",
                  "| run | peak lr | tokens trained (found / true) | false positives | missed | share of numbers changed per updated vector (median) | relative change (median) |",
                  "|---|---|---|---|---|---|---|"]
        for k, v in P.items():
            lines.append(f"| {k} | {v['peak_lr']:g} | {v['tp']:,} / {v['seen']:,} | {v['fp']} | {v['fn']} | "
                         f"{100 * v['changed_coords_frac_median']:.0f}% | {v['rel_change_median']:.2g} |")
        lines += ["", "At the low learning rate, trained tokens whose updates never reached one 16-bit step stay bit-identical:",
                  "that is the detection floor."]
    (R / "STACKS.md").write_text("\n".join(lines) + "\n")
    json.dump({k: dict(status=status(k)) for k, _ in PLAN}, open(R / "status.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
