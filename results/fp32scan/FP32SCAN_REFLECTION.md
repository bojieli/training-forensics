# Reflection-Llama-3.1-70B (mattshumer)

The card names Meta-Llama-3.1-70B-Instruct as the base. The repo was created on 2024-09-05 and re-uploaded on
2024-09-06 (162 fp32 shards); both uploads are compared. Reference models are the mirrors
`unsloth/Meta-Llama-3.1-70B[-Instruct]`, `unsloth/Llama-3.3-70B-Instruct` and `NousResearch/Meta-Llama-3-70B[-Instruct]`
(the official repos are gated); every safetensors shard of each mirror has the same LFS sha256 as the official meta-llama repo (results/reflection_check/reference_hashes.json), so they are byte-identical to Meta's releases.

| | current upload (2024-09-06) | first upload (2024-09-05, e7bfef9796) |
|---|---|---|
| vocabulary entries | 128,262 (6 added) | 128,256 |
| numbers on the 16-bit grid (embedding) | 100.0% | 100.0% |
| token vectors bit-identical to llama-3.1-70b | 0 | 0 |
| token vectors bit-identical to llama-3.1-70b-instruct | 0 | 128,256 |
| token vectors bit-identical to llama-3.3-70b-instruct | 0 | 279 |
| token vectors bit-identical to llama-3-70b | 0 | 0 |
| token vectors bit-identical to llama-3-70b-instruct | 51,241 | 0 |
| tokens whose nearest reference is llama-3.1-70b | 0.0% | 0.0% |
| tokens whose nearest reference is llama-3.1-70b-instruct | 0.0% | 100.0% |
| tokens whose nearest reference is llama-3-70b | 0.6% | 0.0% |
| tokens whose nearest reference is llama-3-70b-instruct | 99.4% | 0.0% |

- **First upload:** the embedding matrix equals Llama-3.1-70B-Instruct's in every vocabulary token (bit-identical).
- **Current upload, embedding:** 51,241 of 128,256 token vectors are bit-identical to **Llama-3-70B-Instruct** (Llama 3, not 3.1) and none to any Llama 3.1 or 3.3 model; 99.4% of token vectors are nearest to Llama-3-70B-Instruct. Measured from the Llama-3.1 base, the change points the way Llama-3-70B-Instruct does (median cosine 0.82) and not the way Llama-3.1-70B-Instruct does (0.03).
- The 6 added tokens (`<thinking>`, `</thinking>`, `<output>`, `</output>`, `<reflection>`, `</reflection>`) have vectors of size 1.76-1.82, about twice a typical token's.
- **Other tensors (current upload):** 7 of 7 sampled normalisation weight vectors are bit-identical to Llama-3-70B-Instruct (and differ from every Llama 3.1 model). The sampled attention projections are fp32 values (off the 16-bit grid) closest to Llama-3-70B-Instruct (relative distance 0.0003, 0.0001, 0.0031, 0.0076, 0.0041, 0.0056 vs 0.011, 0.010, 0.053, 0.063, 0.066, 0.043 for Llama-3.1-70B-Instruct).
- **Output matrix:** 26.2% of its numbers are bit-identical to Llama-3-70B-Instruct's, 2.3832% to Llama-3.1-70B-Instruct's.

## Independent check (results/reflection_check/)

Recomputed with separate scripts (`scripts/reflection_check/`) over every shard and tensor; revisions pinned in `scripts/reflection_check/pins.py` (current upload = main @ `a376762159`, first upload `e7bfef9796`).

- **Three uploads.** `276a4a0a0a` (2024-09-05 18:29, "Duplicate from sahil2801/reflect_v5_70", 123 shards, config naming `meta-llama/Meta-Llama-3.1-70B-Instruct`), `e7bfef9796` (2024-09-05 23:53, 162 shards, config naming `meta-llama/Meta-Llama-3.1-70B-Instruct`), and the re-upload of 2024-09-06 on main (162 shards, 162 of them different from the first upload, config naming `meta-llama/Meta-Llama-3-70B-Instruct`). The `276a4a0a0a` upload was not compared tensor by tensor.
- **References.** `NousResearch/Meta-Llama-3-70B-Instruct` 30/30 shards sha256-equal to `meta-llama/Meta-Llama-3-70B-Instruct`; `NousResearch/Meta-Llama-3-70B` 30/30 shards sha256-equal to `meta-llama/Meta-Llama-3-70B`; `unsloth/Meta-Llama-3.1-70B-Instruct` 30/30 shards sha256-equal to `meta-llama/Meta-Llama-3.1-70B-Instruct`; `unsloth/Meta-Llama-3.1-70B` 30/30 shards sha256-equal to `meta-llama/Meta-Llama-3.1-70B`; `unsloth/Llama-3.3-70B-Instruct` 30/30 shards sha256-equal to `meta-llama/Llama-3.3-70B-Instruct`.
- **First upload:** all 723 tensors bit-identical to Llama-3.1-70B-Instruct in the sampled elements (first 2^20 numbers of each tensor); embedding and output matrix identical in full (128,256 of 128,256 token vectors).
- **Current upload, structure:** all 723 tensors across 162 shards are nearest Llama-3-70B-Instruct (shards with mixed nearest reference: 0). 161 of 161 normalisation weight vectors are bit-identical to it. The 560 projection matrices sit at relative distance 0.0001-0.045 from it versus 0.0097-0.150 from Llama-3.1-70B-Instruct (per-type median ratio 8-21x).
- **Current upload, embedding:** 51,241 token vectors bit-identical to Llama-3-70B-Instruct (bit-identical to the others: L3 0, L3.1-I 0, L3.1 0, L3.3-I 0); Llama-3 and Llama-3.1 Instruct share 0 token vectors, so a bit-identical vector cannot fit both. 99.4% of token vectors are nearest Llama-3-70B-Instruct, but for the changed vectors the margin over the second-nearest reference is small (median ratio 1.27), so the identical vectors, not the nearest-reference share, carry the lineage.
- **Output matrix:** 26.2% of its numbers equal Llama-3-70B-Instruct's, 2.4% Llama-3.1-70B-Instruct's (unrelated Llama references share 2.9%); no output vector is fully identical.
- **Rank of the change** (first 1,024 rows of 35 sampled projection matrices in 5 layers): the difference to Llama-3-70B-Instruct has rank at most 256 in all 35: the 256th singular value is followed by a drop of 12x or more (median 30,077x). Counting singular values above 1e-3 of the largest gives 34 at 256, 1 at 243 (the one below 256 has small values before the drop). The difference to Llama-3.1-70B-Instruct is above rank 256 in 35 of 35. The embedding and output matrix were not rank-tested and changed differently (values on the 16-bit grid, 77,015 token vectors not identical to Llama-3-70B-Instruct).
- **Merge or average with Llama 3.1:** ruled out. It would leave no bit-identical normalisation weights or token vectors; projecting each changed embedding row's difference onto (Llama-3.1-Instruct - Llama-3-Instruct) gives a median coefficient of 0.004 (0 = pure Llama 3, 1 = pure Llama 3.1).
- **The 6 added tokens** (`<thinking>` ... `</reflection>`): norms 1.76-1.82, per-number std 0.0194-0.0201, consistent with default N(0, 0.02) initialisation; whether they were trained cannot be determined.

## Summary

The card names Llama-3.1-70B-Instruct as base. The first upload (5 Sep 2024) is bit-identical to Llama-3.1-70B-Instruct in every tensor we sampled. The current weights (6 Sep 2024) derive from Llama-3-70B-Instruct, not 3.1: all normalisation weights and 51,241 token vectors are bit-identical to it and none to any Llama 3.1/3.3 model, in 35 sampled projection matrices the difference from it has rank at most 256, and the repository's own config.json names Meta-Llama-3-70B-Instruct. The embedding and output matrix changed differently. References were verified byte-identical to Meta's releases by sha256.

This agrees with public reports from September 2024. A rank-256 difference is consistent with a low-rank adapter (e.g. LoRA) merged into Llama-3-70B-Instruct, but the training method is not proven: an intermediate model that is itself Llama-3-70B-Instruct plus a low-rank update would look the same. The weights say nothing about intent. Because the embedding values sit on the 16-bit grid, no step-level timing can be recovered.

