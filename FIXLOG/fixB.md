# Fix-B: LIMO and Reflection-70B reproducibility (TECH_REVIEW B2, B4 Reflection row, M11 parts, minor 9)

## LIMO (B2)
- `scripts/limo_check/run_all.sh` reruns the whole check: `fetch.py` pulls every input at revisions pinned in
  `pins.py` (GAIR/LIMO main 7b59ab38, optimizer revision 1833babb94 and its deletion 910230fdaa, Qwen bases, LIMO-v2,
  s1-32B, both HF datasets, GitHub GAIR-NLP/LIMO 2284c6a with the limo.json blob checked unchanged since 70074cd)
  into `$TF_CACHE/limo_check` (~23 GB, not in git). Each step writes `results/limo_check/out/<step>.json` and a log
  `out/<step>.txt` (home path masked). `summarize.py` builds `summary.json` from those files only; nothing is typed in.
  Rerun on 2026-11-16 reproduced every earlier number (122,322 / 29,742; 29,742 nonzero second moments; 203 / 16,559).
- `run_args.py` reads `training_args.bin` with pickletools (no unpickling): per-device batch 1, gradient accumulation 1,
  weight decay 0.0, lr 5e-6, 15 epochs, seed 42, resume_from_checkpoint None, output_dir basename `aime817_deepseek-r1`,
  DeepSpeed `ds_z3_config.json`. Process count 32 = number of optimizer partitions. So 801-832 examples per epoch.
- total_flos check: not usable. Under ZeRO-3 the Trainer's parameter count depends on which tensors stay unpartitioned;
  candidates give 6.1M-14.9M tokens/epoch vs 5.67M published (16,384). Only the q/k/v-bias count divides exactly,
  which given total_flos/6 = 2^19 * 7^2 * 233 * 10273 is weak evidence. Reported as inconclusive.
- Cutoff dependence: examples with a no-gradient token: 1,024: 19; 2,048: 33; 4,096: 46; 8,192: 57; 16,384: 61;
  32,768: 76. The run's cutoff is not in the released files; GitHub yaml said 32768 on the release day (70074cd) and
  16384 from 2025-02-14 (04efb71); the paper says 16,384.
- 13,386 vs 13,183: 13,386 distinct tokens in the published set (16,384 window), 13,183 of them changed, 203 not;
  16,559 changed tokens absent. The article's "only 13,386 of those tokens occur" should read
  "only 13,183 of those occur in the published set (which has 13,386 distinct tokens)".
- Tension (551/624 in the review; our recount 596/654 with a 16,384 window): published examples add 7.1 unique tokens on
  average (median 4, max 308), so 16,559 absent tokens would need ~2,336 differing examples (>817) at the mean rate, or
  >=54 at the max rate. PUBLISHED_LIMO.md now says the differing trained text must be very unlike the published text
  (5,391 absent tokens are CJK) and that "at least 61" is a floor, not an estimate.
- Evaluation-set coverage: base rates computed (out/deeper.json). They do not neutralise the counts: LIMO's changed set
  covers 43/79 gaokao, 99/210 Chinese grade-school, 64/199 kaoyan questions while LIMO-v2's and s1-32B's changed sets
  cover 0-1, because only LIMO's trained text contained thousands of CJK tokens. Since coverage cannot show training on a
  question and the table is easy to misread, the counts are dropped from PUBLISHED_LIMO.md (kept in out/deeper.json)
  with an explanation. Do not put them in README or the article.
- "The questions match" is reworded to "consistent with the same questions" (98.3% of all published tokens got gradients).

Corrected sentences for README / article (I did not edit them):
- "LIMO changed 29,742 token vectors; the published 817-example set has 13,386 distinct tokens, of which 13,183 changed and
  203 never received a gradient, and 16,559 changed tokens (5,391 CJK) do not occur in it."
- "The solution text used in training differs from the published one for at least 61 examples at LIMO's 16,384-token cutoff
  (19-76 depending on the cutoff); the weights cannot say how many more."
- Drop "a set cleaned after training would produce this pattern" (speculation about the authors' process).
- Drop any evaluation-set coverage numbers.

## Republishing (B2)
- `git rm`: `results/limo_check/evidence/` (GAIR's trainer_state.json and config.json copies, the latter with GAIR's
  cluster path in `_name_or_path`, plus intermediate masks), `results/published/limo.json` (16 MB copy of the dataset),
  and the old hand-made `results/limo_check/{compare_parents,tok_limo,sets}.json` (superseded by out/).
  `scripts/published/extra_sets.py` now reads limo.json from the pinned GitHub commit into the cache.
- Licences of the datasets behind committed derived files (token-id counts only, no text): GAIR/LIMO apache-2.0,
  GAIR/LIMO-v2 apache-2.0, simplescaling/s1K apache-2.0, s1K-1.1 MIT, NovaSky-AI/Sky-T1_data_17k apache-2.0,
  bespokelabs/Bespoke-Stratos-17k apache-2.0, open-thoughts/OpenThoughts-114k apache-2.0. No other file in
  results/published contains raw text (checked: only limo.json had long text strings).
- History rewrite: not advisable. The removed files stay in commits 33ddcb4..; limo.json is apache-2.0 and redistributable
  with attribution, GAIR's config.json/trainer_state.json are public on GAIR/LIMO main today, and the cluster path is in
  GAIR's own public config. A force-push would break existing clones and links for little gain.
- Not mine, flagged: `results/arete_check/nous_blog_data.js` is a verbatim copy of Nous's blog data; check its terms.

## Reflection-70B (B4 row, minor 9)
- `scripts/reflection_check/pins.py` pins main (a376762159), the first upload (e7bfef9796), the dup upload (276a4a0a0a)
  and all five Llama reference repos; `st.full()` resolves any unpinned call to the pin, and `fp32scan/lineage70b*.py`
  use the same pins. Rerunning `repo_history.py` at the pinned main reproduces repo_history.json exactly.
- `report_reflection.py` now generates the Independent check section from results/reflection_check/*.json. Rank is
  reported as: difference to Llama-3-70B-Instruct has rank <= 256 in all 35 sampled projection matrices (drop after the
  256th singular value >= 12x, median ~30,000x; a 1e-3 count gives 34 at 256, 1 at 243); the embedding and output
  matrix were not rank-tested and changed differently (77,015 token vectors not identical). Embedding nearest-reference
  margin (median 1.27) is stated, so the identical vectors carry the lineage. The dup upload is listed.

Corrected sentence for README / article: "The current weights derive from Llama-3-70B-Instruct, not 3.1: all
normalisation weights and 51,241 token vectors are bit-identical to it and none to any Llama 3.1/3.3 model; in 35 sampled
projection matrices the difference has rank at most 256; the embedding and output matrix changed differently. The
repository's config.json names Meta-Llama-3-70B-Instruct." Replace "settle it bit for bit" and "remaining matrices
differ by a rank-256 update".
