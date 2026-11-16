## Notable findings (verdicts per STYLE.md)

Numbers come from `frontier_x.json` (generated from `results/roster/sweep_frontier_x.jsonl`). "Change histogram" =
share of token vectors whose numbers differ from the decayed base in 0 / <=1% / 1-10% / 10-50% / >50% of places.

1. **Reinforcement-learning stages leave a small-update footprint. Verdict: holds.**
   - **Lab-documented RL-only runs:**
     - Olmo-3-7B-RL-Zero-Math, RL straight from the base: 37.8% of token vectors bit-identical, 8.2% barely changed, 45.4% with 1-10% of numbers changed.
     - RL-Zero-Mix: 5.5 / 18.7 / 73.2%.
   - **Other RL or patch stages show the same shape:**
     - MiMo-V2.6-Pro MOPD -> RL: 7.2 / 12.0 / 44.1%.
     - MiMo-V2.6-Flash MOPD -> RL: 11.2 / 8.7 / 37.0%.
     - Nemotron-3 Labs Ultra-Math SFT -> RL: 57.4 / 4.7 / 7.4%.
     - EXAONE-4.0 -> 4.0.1: 42.8 / 4.1 / 38.3%.
   - **Contrast, full post-training:** DeepSeek-V3 -> V3-0324 changes 46% of vectors in over half their numbers.
   - Article sentence: "Models trained only with reinforcement learning, such as AI2's Olmo-3 RL-Zero, whose recipe is published, leave most token vectors bit-identical or changed in under 10% of their numbers. DeepSeek-R1-Zero's footprint has the same shape, so it does not by itself point to narrow data."

2. **Several "successive" releases are not decayed copies of the previous release's embedding. Verdict: holds-with-wording.**
   - **No bit-identical vectors and a poor shrink fit (0.004-0.42):**
     - MiniMax-M2 -> M2.1 and M2.5 -> M2.7.
     - GLM-5.1 -> 5.2 -> 5.3.
     - LongCat-Flash-Chat -> Thinking -> Thinking-2601.
     - Qwen3-235B Instruct-2507 -> Thinking-2507.
     - Kimi-K2-Thinking -> K2.5: vectors about 41% smaller on average, fit 0.004.
   - **By contrast, Kimi-K2.5 -> K2.6 is comparable:** 639 bit-identical, fit 1.000. After removing untrainable rows: 628 against K2.5, at most 13 against K2-Thinking, K2-Instruct-0905 or K2-Base, so it passes the review's margin rule (results/frontier/lineage.md).
   - Article sentence: "Kimi K2.6's token embedding descends directly from K2.5's, while K2.5's does not descend from K2-Thinking's. Its vectors were rescaled or retrained. Newer MiniMax, GLM-5.x and LongCat releases were likewise not fine-tuned from the previous public checkpoint's embedding."
   - Wording: "not a fine-tune of that exact public checkpoint". Never "retrained from scratch".

3. **DeepSeek release chain. Verdict: holds-with-wording after the review's margin rule** (lineage numbers; special-token details are owned by `results/frontier_ds`).
   - **Bit-identical vectors, raw / after removing unreachable, reserved, at-init, zero and duplicate rows** (results/frontier/lineage.md):
     - V3 -> V3-0324: 1,935 / 754. But V3-0324 shares *more* cleaned vectors with V3-Base (1,160) and R1 (1,015) than with V3: consistent with V3-0324 starting from V3-Base or a shared intermediate, not necessarily from V3.
     - V3.1-Base -> V3.1: 1,020 / 292 (V3-Base and V3-0324: 0). Holds.
     - V3.1 -> V3.1-Terminus: 399 / 1. The shared vectors are untrainable tokens only: no lineage evidence beyond a shared start.
     - V3.2-Exp-Base -> V3.2-Exp: 1,498 / 313 (V3.1-Terminus, V3.1: 0). Holds.
     - V3.2-Exp -> V3.2: 1,210 / 157, but V3.2-Exp-Base 203: shares never-trained vectors with both; the weights do not decide.
     - V3.2 -> V3.2-Speciale: 976 / 178, but V3.2-Exp 264: shares with both. Speciale is the smallest update in the chain: 70% of vectors changed in only 10-50% of numbers.
   - (Before the review this item said every step was a direct descendant; bit-identical counts alone cannot show a direct parent, because siblings and ancestors share the untouched vectors too.)
   - **V3-Base -> V3.1-Base** is only partly reproduced (fit 0.973, 470 bit-identical). This is consistent with the further pretraining DeepSeek describes for V3.1-Base.
   - **Updated reserved placeholder slots:**
     - V3 -> 0324: 87 of 800.
     - V3.1-Base -> V3.1: 488 of 796.
     - V3.2-Exp: 20.
     - V3.2: 48, plus 127 barely changed.
     - Speciale: 77, plus 371 barely changed.

4. **New comparable frontier base -> release comparisons. Verdict: holds.**
   - Mistral-Large-3-675B Base -> Instruct: 1,057 bit-identical, mostly the low-id control slots of its tokenizer.
   - Ling-2.6-1T base -> release: 1,774 bit-identical; 229 of 246 reserved slots bit-identical, 17 updated.
   - Ling-3.0-flash base -> release: 1,130.
   - Trinity-Large Base -> Preview: 1,555, including its fill-in-the-middle and pad tokens.
   - Each of these can carry a token record.

5. **Sibling comparisons need care. Verdict: holds-with-wording.**
   - Ling-2.6-1T -> Ring-2.6-1T: 1,650 bit-identical, fit 1.000.
   - Qwen3-Next Instruct -> Thinking: 3,700, no shrink.
   - **Why it needs care:** both are consistent with one model being trained from the other, *or* with both being trained from a shared, unreleased starting point. In that case the identical vectors are tokens neither post-training updated.
   - Article sentence: "consistent with a shared starting point".

6. **No usable comparison, by cause.**
   - **Shared input/output matrix:** Qwen3.5-2B/4B, ERNIE-4.5-21B-A3B, Granite-4.x.
   - **Base not comparable to the released model** (fit 0.21-0.47): Qwen3-Coder-Next, Ling-flash-2.0, Ling-mini-2.0, Ling-3.0-tiny, Ling-1T -> Ring-1T.
   - **Changed throughout:** Seed-OSS woSyn base -> Instruct, ERNIE-4.5-300B / VL-424B base -> release, Ling-2.6-flash, Kimi-K2-Instruct -> 0905, K2.6 -> K2.7-Code, GLM-5 -> 5.1, MiniMax-M2.1 -> M2.5.

7. **GLM-4.5 -> 4.6 -> 4.7: the 128 bit-identical vectors are the unused slots 151,424-151,551 at the end of the vocabulary. Verdict: drop as a data finding.** Every real token changed.

**Not run:**
- Apertus-v1.5 and Llama-3.1-405B: gated.
- dots.llm1 and GLM-4.7 -> GLM-5: hidden size differs.
- Hunyuan-A13B, Falcon-H1-34B, Apertus-70B and the Trinity-Mini / Large-Thinking comparisons were already in the roster sweep.
