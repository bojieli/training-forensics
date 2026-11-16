# Roster sweep: interpretation notes (hand-written)

Every token in a model's vocabulary has its own vector of numbers. Comparing a fine-tune with its parent token
by token shows which vectors were updated; numbers below count distinct vocabulary tokens, not amounts of text.

- **Shared input/output matrix (tied embeddings, 32 pairs).** The output softmax sends a gradient to every token
  vector at every step, so the record of which tokens were in the data is gone (only 2 of 32 keep >= 10 untouched
  vectors; 2 are unchanged). The same holds for separate output layers: in 105 of 107 measured, >= 99% of output
  vectors were updated.
- **"Every token vector changed" (56 pairs)** mixes: tied output layers, re-initialised or rescaled vectors,
  vocabulary-wide signals (e.g. merges, distillation with logit losses through tied weights), and decay that cannot be
  identified. It is not evidence that every token was in the data: weight decay alone changes every vector. The shrink
  factor P is only reported when >= 10 vectors are reproduced exactly by round(P x parent).
- **Exactly-one-untouched pairs.** 3 pairs (Intern-S2-397B, Nemotron-3-Super-120B, Qwen3.5-9B) keep a single
  untouched vector; they are not counted among the 126 pairs with a token record.
- **Kwaipilot/KAT-Dev-72B-Exp:** the fit returns P = 0 on 2,171 vectors, i.e. those vectors are exactly zero in the
  fine-tune (zeroed, not trained). Its 22/22 "updated" special tokens are not a training signal, and it is not
  counted among the pairs with weight decay.
- **CORRECTED (exact ranges, `scripts/roster/p_interval.py` -> `p_intervals.json`).** The "identical decay across
  families" (P = 0.994148 for Llama-3.1-8B/70B-Instruct, Llama-3.3-70B-Instruct, Nemotron-3-Ultra-550B,
  LLaDA-MoE-v2-30B) is a fitting artifact. In all five, every number of the reproduced vectors shrank by exactly one
  smallest 16-bit step, which only confines P to [0.994141, 0.996124] (total weight decay sum lr*wd in
  [0.0039, 0.0059]); 0.994148 is the lower edge of that range, which any one-step shrink lands on. No shared recipe
  is implied.
- **Qwen3-8B / 14B / 30B-A3B:** P in [0.986911, 0.986979] for all three (total weight decay in [0.01311, 0.01318];
  vectors shrank 2-3 16-bit steps). The range is set by the 16-bit grid, so this means the three agree within 0.5%:
  consistent with one post-training schedule across sizes, not proof of it - in the same way Qwen2.5 Instruct updated
  the same 4/22 special tokens at 7B/14B/32B/72B.
- **Single-step tokens are rare in 16-bit releases** (5 pairs with >= 20). Learning rates estimated from them
  (~8e-7 for Llama-3-8B-Instruct and Mistral-Nemo-Instruct) rest on changes of only ~1-3 smallest 16-bit steps and
  are weak; Hermes-4.3-36B's change D = 2^-7 exactly (lr 2.5e-4) looks like a rounding plateau, not a timestamp. Do not
  cite these as recovered learning rates.
- **Pretraining decay, GLM.** GLM-4.5-Base, 4.6, 4.7, 5, 5.1, 5.2, 5.3 give total weight decay 5.98 -> 6.16, rising
  slightly with each release. That pattern fits never-trained vectors inherited through continued training of one
  base (each release adds a little decay), not a shared recipe - and every value assumes init std = initializer_range.
- **Pretraining decay, Kimi-Linear-48B:** P = 6.7e-8 (sum lr*wd 16.5 under the assumed init) is implausible for decay
  alone; both PRETRAIN_DECAY.md and COVERAGE.md flag it, with no decay value.
- **What is robust from public 16-bit releases:** the exact set of tokens whose vectors were never updated (never in a
  gradient), which special tokens a fine-tune used, and - when untouched vectors exist - the total weight decay as a range.
- Exact P ranges: `results/roster/p_intervals.json` lists 27 pairs; 23 give a range (18 tight, mostly 0.002-0.01%
  wide; 5 one-step bounds ~0.2% wide) and 4 give none (3 cache misses, 1 where no vector is reproduced by one factor).
