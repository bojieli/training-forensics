#!/usr/bin/env python3
"""Reflection-Llama-3.1-70B lineage -> results/fp32scan/FP32SCAN_REFLECTION.md
(independent check: results/reflection_check/, scripts/reflection_check/)"""
import json, pathlib
R = pathlib.Path(__file__).resolve().parents[2] / "results" / "fp32scan"
J = lambda n: json.load(open(R / n)) if (R / n).exists() else None
def independent_check():
    """Generated from results/reflection_check/*.json (scripts/reflection_check/, revisions pinned in its pins.py)."""
    import collections, statistics, sys
    RC = R.parent / "reflection_check"
    J2 = lambda n: json.load(open(RC / n))
    sys.path.insert(0, str(R.parents[1] / "scripts" / "reflection_check"))
    from pins import REFL_DUP, REFL_FIRST, REFL_MAIN
    s, e, h, lr, rh, rf = (J2(n) for n in ("all_tensors_summary.json", "embedding.json", "lm_head.json", "lowrank.json",
                                             "repo_history.json", "reference_hashes.json"))
    cur, fst = s["current"], s["first"]
    ec, ef = e["children"]["current"], e["children"]["first"]
    norm_types = [t for t in cur["by_type"] if "norm" in t]
    n_norm = sum(cur["by_type"][t]["n"] for t in norm_types)
    n_norm_eq = sum(cur["by_type"][t]["n"] for t in norm_types if cur["by_type"][t]["eq_L3I"][0] == 1.0)
    proj = [t for t in cur["by_type"] if t.endswith("proj.weight")]
    n_proj = sum(cur["by_type"][t]["n"] for t in proj)
    rel3 = (min(cur["by_type"][t]["rel_L3I"][0] for t in proj), max(cur["by_type"][t]["rel_L3I"][2] for t in proj))
    rel31 = (min(cur["by_type"][t]["rel_L31I"][0] for t in proj), max(cur["by_type"][t]["rel_L31I"][2] for t in proj))
    med_ratio = [cur["by_type"][t]["rel_L31I"][1] / cur["by_type"][t]["rel_L3I"][1] for t in proj]
    first_all_eq = all(b["eq_L31I"][0] == 1.0 for b in fst["by_type"].values())
    ranks = collections.Counter(v["L3-I"]["rank_gt_1e3"] for v in lr.values())
    drops = [v["L3-I"]["s_ratio"]["255"] / v["L3-I"]["s_ratio"]["256"] for v in lr.values()]
    full31 = sum(v["L3.1-I"]["rank_gt_1e3"] > 256 for v in lr.values())
    pe3, pe31 = ec["per_ref"]["L3-I"], ec["per_ref"]["L3.1-I"]
    others = {k: v["rows_bit_identical"] for k, v in ec["per_ref"].items() if k != "L3-I"}
    hc = h["children"]["current"]["per_ref"]
    xr = ec["extra_rows"]
    ref_ok = {k: (v["sha256_equal"], v["n_official"]) for k, v in rf.items()}
    rank_txt = ", ".join(f"{n} at {r}" for r, n in sorted(ranks.items(), reverse=True))
    L = ["", "## Independent check (results/reflection_check/)", "",
         f"Recomputed with separate scripts (`scripts/reflection_check/`) over every shard and tensor; revisions pinned in"
         f" `scripts/reflection_check/pins.py` (current upload = main @ `{REFL_MAIN[:10]}`, first upload `{REFL_FIRST}`).", "",
         f"- **Three uploads.** `{REFL_DUP}` (2024-09-05 18:29, \"Duplicate from sahil2801/reflect_v5_70\", {rh['dup']['n_shards']} shards,"
         f" config naming `{rh['dup_config']['_name_or_path']}`), `{REFL_FIRST}` (2024-09-05 23:53, {rh['first']['n_shards']} shards, config"
         f" naming `{rh['first_config']['_name_or_path']}`), and the re-upload of 2024-09-06 on main ({rh['main']['n_shards']} shards,"
         f" {rh['shards_changed_first_vs_main']} of them different from the first upload, config naming"
         f" `{rh['main_config']['_name_or_path']}`). The `{REFL_DUP}` upload was not compared tensor by tensor.",
         "- **References.** " + "; ".join(f"`{k}` {a}/{b} shards sha256-equal to `{rf[k]['official']}`" for k, (a, b) in ref_ok.items()) + ".",
         f"- **First upload:** {'all' if first_all_eq else 'not all'} {fst['tensors']} tensors bit-identical to Llama-3.1-70B-Instruct in the"
         f" sampled elements (first 2^20 numbers of each tensor); embedding and output matrix identical in full ({ef['per_ref']['L3.1-I']['rows_bit_identical']:,} of {e['rows']:,} token vectors).",
         f"- **Current upload, structure:** all {cur['tensors']} tensors across {cur['shards_covered']} shards are nearest Llama-3-70B-Instruct"
         f" (shards with mixed nearest reference: {len(cur['shards_not_pure'])}). {n_norm_eq} of {n_norm} normalisation weight vectors"
         f" are bit-identical to it. The {n_proj} projection matrices sit at relative distance {rel3[0]:.4f}-{rel3[1]:.3f} from it versus"
         f" {rel31[0]:.4f}-{rel31[1]:.3f} from Llama-3.1-70B-Instruct (per-type median ratio {min(med_ratio):.0f}-{max(med_ratio):.0f}x).",
         f"- **Current upload, embedding:** {pe3['rows_bit_identical']:,} token vectors bit-identical to Llama-3-70B-Instruct"
         f" (bit-identical to the others: " + ", ".join(f"{k} {v:,}" for k, v in others.items()) + f"); Llama-3 and Llama-3.1 Instruct share"
         f" {e['ref_pairs']['L3-I vs L3.1-I']['rows_bit_identical']} token vectors, so a bit-identical vector cannot fit both."
         f" {100 * pe3['nearest_share']:.1f}% of token vectors are nearest Llama-3-70B-Instruct, but for the changed vectors the margin"
         f" over the second-nearest reference is small (median ratio {ec['nearest_margin_ratio_2nd_over_1st'][3]:.2f}), so the"
         " identical vectors, not the nearest-reference share, carry the lineage.",
         f"- **Output matrix:** {100 * hc['L3-I']['values_identical_share']:.1f}% of its numbers equal Llama-3-70B-Instruct's,"
         f" {100 * hc['L3.1-I']['values_identical_share']:.1f}% Llama-3.1-70B-Instruct's (unrelated Llama references share"
         f" {100 * h['ref_pairs']['L3-I vs L3.1-I']['values_identical_share']:.1f}%); no output vector is fully identical.",
         f"- **Rank of the change** (first 1,024 rows of {len(lr)} sampled projection matrices in 5 layers): the difference to"
         f" Llama-3-70B-Instruct has rank at most 256 in all {len(lr)}: the 256th singular value is followed by a drop of"
         f" {min(drops):.0f}x or more (median {statistics.median(drops):,.0f}x). Counting singular values above 1e-3 of the largest"
         f" gives {rank_txt} (the one below 256 has small values before the drop). The difference to Llama-3.1-70B-Instruct is"
         f" above rank 256 in {full31} of {len(lr)}. The embedding and output matrix were not rank-tested and changed differently"
         f" (values on the 16-bit grid, {e['rows'] - pe3['rows_bit_identical']:,} token vectors not identical to Llama-3-70B-Instruct).",
         f"- **Merge or average with Llama 3.1:** ruled out. It would leave no bit-identical normalisation weights or token vectors;"
         f" projecting each changed embedding row's difference onto (Llama-3.1-Instruct - Llama-3-Instruct) gives a median"
         f" coefficient of {ec['merge_alpha_live_changed_rows'][3]:.3f} (0 = pure Llama 3, 1 = pure Llama 3.1).",
         f"- **The {xr['count']} added tokens** (`<thinking>` ... `</reflection>`): norms {min(xr['norms']):.2f}-{max(xr['norms']):.2f},"
         f" per-number std {min(xr['std']):.4f}-{max(xr['std']):.4f}, consistent with default N(0, 0.02) initialisation; whether they"
         " were trained cannot be determined.", "",
         "## Summary", "",
         "The card names Llama-3.1-70B-Instruct as base. The first upload (5 Sep 2024) is bit-identical to Llama-3.1-70B-Instruct"
         " in every tensor we sampled. The current weights (6 Sep 2024) derive from Llama-3-70B-Instruct, not 3.1: all"
         f" normalisation weights and {pe3['rows_bit_identical']:,} token vectors are bit-identical to it and none to any Llama"
         f" 3.1/3.3 model, in {len(lr)} sampled projection matrices the difference from it has rank at most 256, and the"
         " repository's own config.json names Meta-Llama-3-70B-Instruct. The embedding and output matrix changed differently."
         " References were verified byte-identical to Meta's releases by sha256.", "",
         "This agrees with public reports from September 2024. A rank-256 difference is consistent with a low-rank adapter"
         " (e.g. LoRA) merged into Llama-3-70B-Instruct, but the training method is not proven: an intermediate model that is"
         " itself Llama-3-70B-Instruct plus a low-rank update would look the same. The weights say nothing about intent."
         " Because the embedding values sit on the 16-bit grid, no step-level timing can be recovered.", ""]
    return L


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
    L += independent_check()

(R / "FP32SCAN_REFLECTION.md").write_text("\n".join(L) + "\n")
