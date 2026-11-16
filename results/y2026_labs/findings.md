## Summary

- 435 text / vision-language releases of 3B+ parameters created in 2026 were found across 42 lab accounts (list below).
  78 comparisons were run that the roster sweep, `frontier_x` and `results/large` did not already cover; of 15 other
  candidates, 4 were already covered and 11 were gated, had no fetchable embedding, or had a different vector width
  (last table).
- Outcome of the 78 (reviewed classifier, `scripts/frontier/comparability.py`): token record 31, bit-identical vectors that are all untrainable (unreachable, reserved or at init) 16,
  some bit-identical but poor shrink fit 8, every vector changed but consistent with the parent plus decay 5, not comparable to that parent 13,
  shared input/output matrix 5. A token record shows shared ancestry, not by itself a direct parent (`results/frontier/lineage.md`).
- Most successive flagship releases of 2026 are **not** comparable to their predecessor: their token vectors were
  replaced or re-trained wholesale (Kimi K2.7-Code -> K3 -> K3.1-Pro, MiniMax M3 -> M3.1-Flash, Step-3.5-Flash -> 3.7 ->
  Step-5-Preview, Qwen3.5 -> Qwen3.6 at 27B and 35B-A3B, Qwen3.6-27B -> Qwen3.8-27B, MiMo-V2.5 -> V2.6-Flash). The
  readable 2026 records are mostly base -> instruct pairs from the same release and third-party or lab-side
  derivatives of a released model.

## Findings (verdict, evidence, article sentence)

1. **Qwen WebWorld-14B starts from Qwen3-14B, not the Qwen3-8B its card names. Holds.**
   Qwen3-8B's token vectors are 4,096 numbers wide and WebWorld-14B's are 5,120, so the card's base cannot be the
   starting point. Against Qwen3-14B, 10,389 of 151,936 token vectors are bit-identical (0.2% barely changed); against
   Qwen3-14B-Base, 2,124 are bit-identical and the shrink of those vectors lies in 1.30-1.31% (P in [0.98691,
   0.98698]), the same band as Qwen3-8B/14B/30B-A3B themselves. The 11 vision tokens are bit-identical, as in Qwen3-14B.
   *Sentence:* "WebWorld-14B's card names Qwen3-8B as its base, but its token vectors are a different width; 10,389 of
   them are bit-identical to Qwen3-14B's, which is consistent with Qwen3-14B as the starting point."

2. **Qwen-AgentWorld-35B-A3B shares exact token vectors with Qwen3.6-35B-A3B, not with the Qwen3.5 base its card
   names. Holds with wording.** 940 token vectors are bit-identical to Qwen3.6-35B-A3B; against Qwen3.5-35B-A3B-Base
   (the card's base) and Qwen3.5-35B-A3B, only 1 is. Qwen3.5-35B-A3B-Base -> Qwen3.6-35B-A3B itself changed every vector
   but one, so the 940 shared values are Qwen3.6's. The rest of AgentWorld's vectors fit a uniform 0.46% *growth*
   rather than a shrink, which the standard classifier rejects; the bit-identity is the evidence used here.
   *Sentence:* "Qwen-AgentWorld-35B-A3B's card names the Qwen3.5 base, but 940 of its token vectors are bit-identical
   to Qwen3.6-35B-A3B and only one to the Qwen3.5 base, consistent with a Qwen3.6 (or pre-release Qwen3.6) starting point."

3. **NVIDIA's Nemotron Labs teacher models and Ultra-Math-SFT descend from the released Nemotron-3-Ultra; two of the
   teachers share an unreleased intermediate. Holds.** Against Nemotron-3-Ultra-550B-A55B-Base, Teacher-Chat and
   Math-SFT both keep 1,644-1,661 vectors bit-identical, with the same shrink 0.39-0.59% as the released instruct
   model. Against the released instruct model, Math-SFT keeps 81,667 of 131,072 vectors bit-identical (its card names
   that model), Teacher-Chat 9,555, Teacher-General-Reasoning 8,900, Teacher-Instruction-Following 4,763.
   Teacher-Competition-Coding and Teacher-STEM keep none bit-identical to the instruct model, but 81,698 bit-identical
   to *each other*.
   *Sentence:* "Two of NVIDIA's Nemotron Labs teachers (Competition-Coding and STEM) share 81,698 bit-identical token
   vectors with each other and none with the released Nemotron-3-Ultra instruct model, consistent with a common
   unreleased intermediate checkpoint."

4. **Model cards that name no base: the weights do. Holds.** InternLM's Atria-Dawn-Preview and Sage-S2.1 keep 3,179
   and 2,296 token vectors bit-identical to GLM-5.2 (Sage-S2.1 is not a decayed copy of GLM-5.3); Sage-S2.1-Flash keeps
   2,259 bit-identical to GLM-5.3-Flash. Kwaipilot KAT-Coder-V2.5-Dev keeps 22,598 bit-identical to Qwen3.6-35B-A3B
   (its card's base). Microsoft MagenticBrain (an fp32 release) keeps 79,967 bit-identical to Qwen3-14B, Fara1.5-27B
   141,346 to Qwen3.5-27B, AesCode-32B 40,339 to Qwen3-VL-32B-Instruct; AllenAI tmax-9b 167,381 to Qwen3.5-9B.
   *Sentence:* "InternLM's Sage-S2.1 card names no base model; 2,296 of its token vectors are bit-identical to GLM-5.2."
   *Margin rule (results/frontier/lineage.md):* after removing untrainable rows, Sage-S2.1 shares 2,279 vectors with GLM-5.2 and
   2,157 with InternLM's own Atria-Dawn-Preview, so the weights say "GLM-5.2 lineage", not which of the two it started from;
   Atria-Dawn-Preview (3,160 vs 0 for GLM-5.1), KAT-Coder-V2.5-Dev (22,449 vs 16) and MagenticBrain (79,184 vs 4) pass.

5. **2026 flagship successors mostly replace the vocabulary vectors. Holds.** Kimi-K2.7-Code -> Kimi-K3 (same
   163,840 x 7,168 shape): no vector bit-identical and a uniform-shrink fit reproducing 0.1% of numbers, so K3's token
   vectors are not a trained copy of K2.7's. The same "not comparable" verdict holds for K3 -> K3.1-Pro, MiniMax-M3 ->
   M3.1-Flash, Step-3.5-Flash -> Step-3.7-Flash -> Step-5-Preview, Qwen3.5 -> Qwen3.6 (27B and 35B-A3B), Qwen3.6-27B ->
   Qwen3.8-27B, MiMo-V2.5 -> V2.6-Flash-MOPD, Laguna-M.1-base -> Laguna-M.1, and Ling-3.0-flash
   -> its Sante and VL variants (vectors fit a uniform factor of about 0.084, i.e. rescaled).
   *Sentence:* "Between 2026 flagship versions (Kimi K3, MiniMax M3.1, Step-5, Qwen3.6, Qwen3.8) the token vectors do not
   carry over: none is a trained copy of its predecessor's, so release-to-release comparisons say nothing about the
   newer model's data."

6. **Exact total weight decay, 2026 base -> instruct. Holds.** Shrink of never-updated vectors (16-bit exact
   interval): Ling-3.1-flash 1.84-1.84%, Nemotron-3.5-Super-VL 1.21-1.21%, MiniCPM5-2B Midtrain -> SFT 2.62-2.62%,
   Nemotron-3-Ultra Base -> Teacher-Chat / Math-SFT 0.39-0.59%, WebWorld-14B (vs Qwen3-14B-Base) 1.30-1.31%. Several
   2026 base -> instruct pairs show no shrink at all (P = 1: JoyAI-LLM-Flash, Hy3-preview, Step3-VL-10B, Olmo-Hybrid).
   *Sentence:* "Ling-3.1-flash's never-updated token vectors shrank by 1.84% between base and release; JoyAI-LLM-Flash's,
   Hy3-preview's and Step3-VL-10B's did not shrink at all."

7. **Special tokens, 2026 base -> instruct (named tokens only; reserved slots excluded). Holds with wording.**
   Fill-in-the-middle tokens are bit-identical after post-training in Ling-3.1-flash (`<|fim_start|>`, `<|fim_hole|>`,
   `<|fim_end|>`) and Ling-2.6-1T; Hunyuan Hy3-preview leaves its FIM tokens and `<｜hy_EOT｜>` bit-identical; Mistral
   Leanstral-1.5 leaves `[IMG]`, `[AUDIO]`, `[MIDDLE]`/`[SUFFIX]` and `[TOOL_CONTENT]` bit-identical relative to
   Mistral-Small-4; Olmo-Hybrid Instruct-SFT updates the `|||IP_ADDRESS|||` placeholder but leaves `|||PHONE_NUMBER|||`
   and `|||EMAIL_ADDRESS|||` bit-identical; DeepSeek V4-Flash-Vision-Exp leaves `<｜begin▁sys｜>`/`<｜end▁sys｜>`
   bit-identical relative to V4-Flash-0731 (poor shrink fit, 84%: treat with care). Hy3 (vs Hy3-preview) updates ~440
   more placeholder slots than Hy3-preview did. All of these come from pairs where under 1% of vectors are barely
   changed, so bit-identical here is not a rounding artefact.
   *Sentence:* "Ling-3.1-flash's post-training left its three fill-in-the-middle tokens bit-identical to the base."

8. **Not readable.** LiquidAI LFM2/LFM2.5 and IBM Granite 4.1/4.2 share one matrix for input and output (no record).
   Kanana-2, Step-3.5-Flash (base -> midtrain -> release), Intern-S2 and Motif-3 changed every token vector but stay
   consistent with their parent plus decay: same lineage, no token list. Poolside Laguna-XS-2.1's vectors are unrelated
   to Laguna-XS.2's (cosine < 0.2 for 100,249 of 100,352) except 54 bit-identical slots.

Caveats: input embeddings only (no output heads); 16-bit releases (MagenticBrain is fp32 against a bf16 parent, compared
at a 64-ulp threshold); "bit-identical" means after removing the fitted uniform shrink; lineage statements are about
shared exact values, which a later checkpoint of the same line would also share.
