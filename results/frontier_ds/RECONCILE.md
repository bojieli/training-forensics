# Reconciling the DeepSeek and Qwen3 statements across lanes

Written by Fork V. Sources: `results/frontier_ds/deepseek.json` (relabelled with the STYLE.md 1% threshold by
`scripts/frontier_ds/ds_reclass.py`), `results/roster/sweep*.jsonl`, `scripts/roster/sweep.py::fit_decay`,
a recomputation of `fit_decay` on the cached embeddings, and a direct V3 vs V3-0324 comparison.

## 1. DeepSeek V4 vs its bases: a valid reference

**Verdict: DEEPSEEK.md is right. FRONTIER.md's "V4-Flash not usable (fit 29%) / V4-Pro borderline (92%)" and
TOKENS.md's exclusion of V4-Flash come from a fit score that is not built to judge this case.**

- **What the score is.** `fit_decay` ranks every vector by how close it is to a pure shrink of its base vector, takes
  the 512 closest, and reports the share of their numbers that round(shrink × base) reproduces.
- **Why it falls below 99% here.** No weight decay applies (shrink exactly 1). Training touched every vector except 447
  (V4-Flash) and 446 (V4-Pro). The 512-vector sample therefore holds all the untouched vectors (100% reproduced) plus
  65–66 trained vectors (about a third of their numbers reproduced). The score is capped near 446/512, whatever the
  ancestry. Recomputed on the cached files: V4-Pro 0.9235, matching the sweep record; V4-Flash **0.912**.
- **The stored 0.293 for V4-Flash.** The sweep record has 0.293 with the same 447 untouched vectors. I could not
  reproduce it from the cached files: the same code on the same tensors gives 0.912. Treat 0.293 as a stale or bad
  record and re-run that one sweep entry.
- **Why the base is a valid reference.**
  - 446/447 vectors are bit-identical in every one of their 4,096/7,168 numbers. That is impossible unless the release
    continues that base.
  - The bit-identical vectors include 14–15 named special tokens and the 417 newer placeholder slots.
  - With shrink = 1, any changed number was changed by the optimizer, not by decay.
- **What this means for the named tokens.**
  - `<｜political｜>`/`<｜safety｜>`/`<｜authority｜>` are at their random starting size in the base.
  - In the release they changed in 97–99% of their numbers, stay 0.98–1.00 similar to the base vector, and change in
    correlated directions. Their siblings at the same starting size (table/td/tr/polygon/point tokens) stayed
    bit-identical.
  - So the change is a training update relative to the true parent, not a decay or ancestry artefact.
- **Replacement sentences.**
  - FRONTIER.md table row: "DeepSeek-V4-Pro / V4-Flash (vs their bases) | updated | updated | yes: 446/447 vectors
    bit-identical, no decay (shrink 1); the 512-vector fit score (92%/91%) is capped by how few vectors are
    untouched, not a sign of a different base".
  - FRONTIER.md class table: move "DeepSeek-V4 vs its bases" out of "decay fit below 99%". Add a rule: when shrink = 1
    and there are at least 100 bit-identical vectors, the comparison is a token record whatever the 512-vector score.
  - TOKENS.md: drop "DeepSeek-V4-Flash vs its base" from the excluded list, or say "excluded by an outdated fit score
    (0.293 not reproducible; recomputed 0.912)".

## 2. V3 placeholders: two different comparisons, one wording

**Verdict: both files are right about different comparisons. The counts differ only because the threshold
differs (now unified at 1%).**

- **V3-Base → V3.**
  - 97 placeholder vectors differ, all by tiny amounts (0.2–7.9% of numbers, change about 1/25,000 of the vector's size).
  - 87 of the 97 are above 1% of numbers and 10 are below.
  - 76 of the 97 are in the block 128374–128449.
  - DEEPSEEK.md now uses the 1% threshold: 703 bit-identical / 10 barely changed / 87 updated.
- **V3 → V3-0324 (FRONTIER_X.md).** The same 97 vectors differ. V3-0324 has 96 of them exactly at the V3-Base values,
  and it changed the 97th (128421) itself. FRONTIER_X's "87 updated / 10 barely / 703 same" is V3's change seen from
  the other side.
- **The pattern fits V3-0324 starting from V3-Base rather than from the released V3.** Treat this as an observation,
  not an article claim.
- **One wording for the article:** "DeepSeek-V3 changed 97 of its 800 placeholder vectors relative to V3-Base, each by
  a tiny amount (1–8% of its numbers in 87 cases, under 1% in 10), 76 of them in one contiguous block; V3-0324 has 96
  of those 97 exactly at the base values."
- **Never** write "V3 updated 87 placeholders" without the size qualifier.

## 3. Qwen3 in quantized releases

**Verdict: QUANT_MERGE.md's "updated tokens detected 151,936 / 151,936" is correct as a count of vectors that differ
from the base. As a statement about tokens "in the data" it is wrong, because it does not remove the post-training
weight-decay shrink.**

- In Qwen3-8B, 14B and 30B-A3B, 2,160 / 2,126 / 833 vectors were never updated. They only shrank by 1.30–1.31%
  (FRONTIER.md §5), so every vector differs from the base before any quantization.
- **Replacement sentence (QUANT_MERGE.md, Qwen3-8B / 14B / 30B-A3B rows and the list):** "quantized (gguf Q8_0 / mlx
  4-bit); every token vector differs from the base quantized by us, as in the 16-bit release, where all vectors
  differ because the never-updated ones shrank by 1.30–1.31% (weight decay). This test compares against the
  unshrunk base, so it cannot separate those 2,160 (8B) / 2,126 (14B) / 833 (30B-A3B) decay-only vectors from
  trained ones; token record not assessed."
- **Optional fix.** Quantize round(0.98694 × base) with the same tool and compare to that. Bit-identical vectors would
  then show whether the record survives quantization.
- The BF16/F32 "identical" rows (AWQ/FP8/NVFP4 files whose embedding is not quantized) are unaffected: those files
  carry the 16-bit record unchanged.
