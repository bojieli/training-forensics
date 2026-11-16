#!/usr/bin/env python3
"""Reflection-Llama-3.1-70B lineage -> results/fp32scan/FP32SCAN_REFLECTION.md
(independent check: results/reflection_check/, scripts/reflection_check/)"""
import json, pathlib
R = pathlib.Path(__file__).resolve().parents[2] / "results" / "fp32scan"
J = lambda n: json.load(open(R / n)) if (R / n).exists() else None
L = []
# ---- Reflection
lin = J("reflection70b_lineage.json")
ten = J("reflection70b_tensors.json")
if lin:
    cur = lin["children"]["mattshumer/Reflection-Llama-3.1-70B"]
    old = [v for k, v in lin["children"].items() if "@" in k][0]
    oldk = [k for k in lin["children"] if "@" in k][0]
    L += ["# Reflection-Llama-3.1-70B (mattshumer)", "",
          "The card names Meta-Llama-3.1-70B-Instruct as the base. The repo was created on 2024-09-05 and re-uploaded on",
          "2024-09-06 (162 fp32 shards); both uploads are compared. Reference models are the mirrors",
          "`unsloth/Meta-Llama-3.1-70B[-Instruct]`, `unsloth/Llama-3.3-70B-Instruct` and `NousResearch/Meta-Llama-3-70B[-Instruct]`",
          "(the official repos are gated); every safetensors shard of each mirror has the same LFS sha256 as the official "
          "meta-llama repo (results/reflection_check/reference_hashes.json), so they are byte-identical to Meta's releases.", "",
          "| | current upload (2024-09-06) | first upload (2024-09-05, " + oldk.split("@")[1][:10] + ") |", "|---|---|---|",
          f"| vocabulary entries | {cur['rows']:,} (6 added) | {old['rows']:,} |",
          f"| numbers on the 16-bit grid (embedding) | {100 * cur['share_values_on_bf16_grid']:.1f}% | {100 * old['share_values_on_bf16_grid']:.1f}% |"]
    for k in ("llama-3.1-70b", "llama-3.1-70b-instruct", "llama-3.3-70b-instruct", "llama-3-70b", "llama-3-70b-instruct"):
        L.append(f"| token vectors bit-identical to {k} | {cur['per_candidate'][k]['bit_identical']:,} | {old['per_candidate'][k]['bit_identical']:,} |")
    for k in ("llama-3.1-70b", "llama-3.1-70b-instruct", "llama-3-70b", "llama-3-70b-instruct"):
        L.append(f"| tokens whose nearest reference is {k} | {100 * cur['nearest_candidate_share'][k]:.1f}% | {100 * old['nearest_candidate_share'][k]:.1f}% |")
    L += ["", "- **First upload:** the embedding matrix equals Llama-3.1-70B-Instruct's in every vocabulary token (bit-identical).",
          f"- **Current upload, embedding:** {cur['per_candidate']['llama-3-70b-instruct']['bit_identical']:,} of 128,256 token "
          "vectors are bit-identical to **Llama-3-70B-Instruct** (Llama 3, not 3.1) and none to any Llama 3.1 or 3.3 model; "
          f"{100 * cur['nearest_candidate_share']['llama-3-70b-instruct']:.1f}% of token vectors are nearest to Llama-3-70B-Instruct. "
          "Measured from the Llama-3.1 base, the change points the way Llama-3-70B-Instruct does (median cosine "
          f"{cur['direction_cos_median']['llama-3-70b-instruct - llama-3.1-70b']:.2f}) and not the way Llama-3.1-70B-Instruct does "
          f"({cur['direction_cos_median']['llama-3.1-70b-instruct - llama-3.1-70b']:.2f}).",
          "- The 6 added tokens (`<thinking>`, `</thinking>`, `<output>`, `</output>`, `<reflection>`, `</reflection>`) have vectors of "
          f"size {min(cur['added_vectors_norms']):.2f}-{max(cur['added_vectors_norms']):.2f}, about twice a typical token's."]
    if ten:
        tt = ten["tensors"]
        norms = [n for n in tt if "norm" in n and "error" not in tt[n]]
        ident = [n for n in norms if tt[n]["llama-3-70b-instruct"]["values_identical"] == 1.0]
        L.append(f"- **Other tensors (current upload):** {len(ident)} of {len(norms)} sampled normalisation weight vectors are "
                 "bit-identical to Llama-3-70B-Instruct (and differ from every Llama 3.1 model). The sampled attention "
                 "projections are fp32 values (off the 16-bit grid) closest to Llama-3-70B-Instruct "
                 "(relative distance " + ", ".join(f"{tt[n]['llama-3-70b-instruct']['rel_dist']:.4f}" for n in tt if "proj" in n and "error" not in tt[n]) +
                 " vs " + ", ".join(f"{tt[n]['llama-3.1-70b-instruct']['rel_dist']:.3f}" for n in tt if "proj" in n and "error" not in tt[n]) +
                 " for Llama-3.1-70B-Instruct).")
        if "lm_head.weight" in tt and "error" not in tt["lm_head.weight"]:
            h = tt["lm_head.weight"]
            L.append(f"- **Output matrix:** {100 * h['llama-3-70b-instruct']['values_identical']:.1f}% of its numbers are bit-identical to "
                     f"Llama-3-70B-Instruct's, {100 * h['llama-3.1-70b-instruct']['values_identical']:.4f}% to Llama-3.1-70B-Instruct's.")
    L += ["", "## Independent check (results/reflection_check/)", "",
          "Recomputed with separate scripts over every shard and tensor of both uploads:", "",
          "- First upload: all 723 tensors bit-identical to Llama-3.1-70B-Instruct in the sampled elements; embedding and "
          "output matrix identical in full.",
          "- Current upload: all 723 tensors across all 162 shards are nearest Llama-3-70B-Instruct; all 161 normalisation "
          "vectors are bit-identical to it; the 560 projection matrices sit at relative distance 0.0001-0.045 from it versus "
          "0.0097-0.15 from Llama-3.1-70B-Instruct (median ratio about 12x).",
          "- In 35 sampled matrices the difference to Llama-3-70B-Instruct has rank exactly 256 (the difference to "
          "Llama-3.1-70B-Instruct is full rank).",
          "- A merge or average with Llama 3.1 is ruled out: it would leave no bit-identical normalisation weights or token vectors.",
          "- The repo's own config.json names `meta-llama/Meta-Llama-3-70B-Instruct` (the first upload's named Llama 3.1).",
          "- The 6 added tokens (`<thinking>` ... `</reflection>`) look like default N(0, 0.02) initialisation; whether they "
          "were trained cannot be determined.", "",
          "## Summary", "",
          "The card names Llama-3.1-70B-Instruct as base. The first upload (5 Sep 2024) is bit-identical to Llama-3.1-70B-Instruct "
          "in every tensor we sampled. The current weights (6 Sep 2024) match Llama-3-70B-Instruct instead: all normalisation "
          "weights and 51,241 token vectors are bit-identical to it, none to any Llama 3.1/3.3 model, the remaining matrices "
          "differ from it by a rank-256 update, and the repo's config.json names Meta-Llama-3-70B-Instruct. References were "
          "verified byte-identical to Meta's releases by sha256.", "",
          "This agrees with public reports from September 2024. A rank-256 difference is consistent with a low-rank adapter "
          "(e.g. LoRA) merged into Llama-3-70B-Instruct, but the training method is not proven: an intermediate model that is "
          "itself Llama-3-70B-Instruct plus a low-rank update would look the same. The weights say nothing about intent. "
          "Because the embedding values sit on the 16-bit grid, no step-level timing can be recovered.", ""]

(R / "FP32SCAN_REFLECTION.md").write_text("\n".join(L) + "\n")
