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

Recomputed with separate scripts over every shard and tensor of both uploads:

- First upload: all 723 tensors bit-identical to Llama-3.1-70B-Instruct in the sampled elements; embedding and output matrix identical in full.
- Current upload: all 723 tensors across all 162 shards are nearest Llama-3-70B-Instruct; all 161 normalisation vectors are bit-identical to it; the 560 projection matrices sit at relative distance 0.0001-0.045 from it versus 0.0097-0.15 from Llama-3.1-70B-Instruct (median ratio about 12x).
- In 35 sampled matrices the difference to Llama-3-70B-Instruct has rank exactly 256 (the difference to Llama-3.1-70B-Instruct is full rank).
- A merge or average with Llama 3.1 is ruled out: it would leave no bit-identical normalisation weights or token vectors.
- The repo's own config.json names `meta-llama/Meta-Llama-3-70B-Instruct` (the first upload's named Llama 3.1).
- The 6 added tokens (`<thinking>` ... `</reflection>`) look like default N(0, 0.02) initialisation; whether they were trained cannot be determined.

## Summary

The card names Llama-3.1-70B-Instruct as base. The first upload (5 Sep 2024) is bit-identical to Llama-3.1-70B-Instruct in every tensor we sampled. The current weights (6 Sep 2024) match Llama-3-70B-Instruct instead: all normalisation weights and 51,241 token vectors are bit-identical to it, none to any Llama 3.1/3.3 model, the remaining matrices differ from it by a rank-256 update, and the repo's config.json names Meta-Llama-3-70B-Instruct. References were verified byte-identical to Meta's releases by sha256.

This agrees with public reports from September 2024. A rank-256 difference is consistent with a low-rank adapter (e.g. LoRA) merged into Llama-3-70B-Instruct, but the training method is not proven: an intermediate model that is itself Llama-3-70B-Instruct plus a low-rank update would look the same. The weights say nothing about intent. Because the embedding values sit on the 16-bit grid, no step-level timing can be recovered.

