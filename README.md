# training-forensics

What a model's published weights reveal about how it was trained: which tokens were in the data, the
learning-rate schedule, the number of steps, and (given the dataset) which optimizer step each document
was used in. This repository is an independent replication of a public thread by Alapan Chaudhuri
([@alapanchaudhuri](https://x.com/alapanchaudhuri), [thread](https://x.com/alapanchaudhuri/status/2121439269772492848), 14 Nov 2026) that made these claims, tested three ways:
on the four public models it names, on controlled fine-tunes with full ground truth (Pythia-160m/410m
and Qwen2.5-7B), and on the 265-model roster of the sibling project
[checkpoint-forensics](https://github.com/bojieli/checkpoint-forensics), plus follow-up studies of frontier
releases and of fine-tunes whose training data is public.

**Timeline and status.** The thread appeared on 14 Nov 2026; everything here was done between 14 and 17 Nov 2026
(see the commit history), with heavy use of AI coding agents. A technical review of the first version raised
blockers and major issues; the answers are logged in [`FIXLOG/`](FIXLOG/) and this README reflects the corrected
numbers.
How to rerun everything: [`REPRODUCE.md`](REPRODUCE.md).

## Terms

- **Token vector (embedding).** Every token in a model's vocabulary has its own vector of numbers. In the
  code and tables this is one *row* of the embedding matrix; counts of rows are counts of distinct
  vocabulary tokens, never amounts of text.
- **Updated / bit-identical.** A token vector is *updated* (*moved* in table columns) when training changed it
  beyond weight decay, and *bit-identical to the base* (*unmoved*) when it equals where it started (after the
  common weight-decay shrink). Bit-identical is an observation. In a 16-bit release a very small update can
  round away, so "never updated" is only said where the detection floor was checked.
- **Parent -> fine-tune pair.** A model and the model trained from it, e.g. Qwen2.5-7B -> Qwen2.5-7B-Instruct.
- **Shrink factor P and total weight decay.** With decoupled weight decay an untouched vector is multiplied by
  (1 - lr x wd) at every step; P is the product over all steps and -ln P = sum of lr x wd, the *total weight
  decay applied*. It gives the product of learning rate and weight decay, summed over the run, not either one alone.
  In a 16-bit release P = 1 means "no shrink resolvable" (below a bound, median 0.195%), not "no weight decay".
- **Tied embeddings.** Models that use one matrix for both the input tokens and the output layer. The output
  layer sends a gradient to every token at every step, which erases the record.
- **Special tokens.** Chat markers, tool-call, thinking, image-placeholder and fill-in-the-middle tokens.
- **16-bit release.** Weights published in bf16/fp16. One *16-bit step* is the smallest change a 16-bit number can show.
- **How sure.** *Recovered exactly* = bit-exact, checked; *estimated* = a fit with error; *bounded* = only a
  range; *detected* = a yes/no test with a measured false-alarm rate; *assumed* = rests on a stated assumption.

## The mechanism, in five lines

1. Fine-tune with Adam(W) and subtract the starting token vectors, token by token.
2. A token the data never contained gets no gradient: its vector is not updated (with decoupled weight
   decay it only shrinks by a factor that is the same for every such token).
3. A token seen in exactly one optimizer step changes by the same amount D in nearly every number of its
   vector: Adam divides a first gradient by its own magnitude, so only the sign survives.
4. D depends only on the learning rates from that step on: D(s) = sum_k lr(s+k) a(k, s). For a constant learning
   rate, default Adam settings and a late step (full bias correction) D/lr -> 31.77; early steps and gradients near
   Adam's eps give less, which the estimation handles with an intercept fit. So D says *when*; the sign pattern
   says *what* (minus the sign of that step's gradient).
5. Under decoupled weight decay every number goes through the same sequence of fp32 multiplications,
   so the untouched vectors can be replayed bit for bit once the schedule is known.

`trainforensics/adam.py` is the forward model; `trainforensics/blind.py` does the estimation;
`scripts/controlled/score.py` analyses a run blind and only then opens its ground truth.

## Results

### The four public claims (`results/public/PUBLIC_CLAIMS.md`)

| Claim | Verdict | What the weights show |
|---|---|---|
| Pythia-1B peak learning rate is 2.5e-4, not the card's 3.0e-4 | **confirmed** | 2.50002e-4, estimated from how the unused padding vectors shrank over 154 checkpoints; the yml config agrees. All 8 sizes land within 0.2% of their configs. |
| Pythia's learning rate is flat for its last ~28,000 steps (cosine to 0 clipped at 0.1x peak) | **confirmed** | clip at 0.1000x peak at every size; flat tail 27,657-27,865 steps (predicted 27,861). Today's gpt-neox scheduler fits 47-227x worse. |
| OLMo-2 first SFT release: ~20,341 tokens that should have been trained were never updated, incl. the chat-turn closer | **confirmed, count differs** | 23,064 real tokens bit-identical to base; 20,878 of them occur in the SFT data (20,358-20,878 under three definitions; the gap to 20,341 is unexplained). `>Ċ` (id 397, 2.39M occurrences) is bit-identical to the base in the first release (a 16-bit release); it was updated in the retrain. |
| Qwen2.5 instruct updated exactly 4 of 22 special tokens, same 4 at 7B/14B/32B/72B | **confirmed** | `<\|im_start\|>`, `<\|im_end\|>`, `<tool_call>`, `</tool_call>` at all four sizes; the other 18 bit-identical. |

Side findings: pythia-12b checkpoints `step50000`/`step58000` contain the final model's padding vectors;
pythia-2.8b's padding vectors are bit-identical to step 0 in 85 of 153 later checkpoints (incl. the final one).

### Controlled fine-tunes (`results/controlled/CONTROLLED.md`, `results/gaps/GAPS.md`, `results/fixc/FIXC.md`)

24 fine-tunes with full ground truth: Pythia-160m/410m (full fine-tuning) and Qwen2.5-7B (token vectors +
output layer, LoRA with trained token vectors, weight decay, batch 128, two epochs), linear / cosine / constant /
cosine-to-floor / warmup-stable-decay schedules, batch 4-128, weight decay 0-0.1. Each run is analysed
from (start, release) and only then scored against the trainer's logged learning rates and batches. All 24 are
scored with one version of the code (the five Qwen2.5-7B runs were retrained for this).

**Scope of these numbers.** The 24 runs are single-epoch (one is two-epoch and unsolved), use round
hyper-parameters, seeds below 100,000 and one of the reader's five schedule families. On five further runs
trained *after* the reader was frozen, with non-round learning rates and weight decay, random 32-bit seeds and one
schedule outside the five families, the reader does much worse (row "fresh runs" below).

| What is recovered or estimated | Result |
|---|---|
| Tokens seen (updated vectors = tokens in the data) | **recovered exactly in 24 / 24**, no false positive: directly where there was no weight decay, and via the bit-exact decay replay (which needs the full recipe) where there was. The direct comparison alone misses 10-469 late-updated tokens in decay runs. |
| Recipe: schedule, steps, warm-up, peak (and weight decay) | **exact in 22 / 24**; peak within 0.12% before rounding to a round value. Misses: r05 (501 steps read for 500) and the two-epoch run. |
| Step count from the weights alone (no claimed dataset or batch size) | Kind, step count and warm-up right in **24 of 25** pre-existing Pythia runs; the dataset-implied step count was never needed. |
| Fresh runs trained after the reader was frozen | Full schedule recovered in **2 of 5**. Two with non-round weight decay misread (0.013 read as 0.015; 738 steps for 500), one out-of-family schedule wrong. Each wrong recipe shows in the replay check (28 to 802,448 of about 24.4M weights bit-exact, against 24,419,699 when right). The direct token list had no false positive in any of the five. |
| Shuffle seed (HF Trainer) | Recovered in every one of the 24 runs where it was searched (all but the two-epoch run). The search covers seeds 0-99,999; a random 32-bit seed is outside it and is not found (fresh runs). |
| Documents in their exact step, from the weights alone | 4,000-document runs: 767-1,714 (19-43%); 8,000 docs: 16%; batch 128 (52,000 docs): 4.5%. Wrong placements are 0-7 in most runs; 51 in r09, 232 in r05, 7-253 in the Qwen2.5-7B runs. **All documents** once the seed is known. |
| Replay of untouched weights | bit-exact: 473,769,019 of 473,804,800 weights (Qwen2.5-7B); one step off leaves ~1,100. Pythia-160m: about 24.4M; one step off leaves 26-279. |
| 40 secretly swapped documents (s05) | v1: 24 caught, 3 false flags; **v2: 27 caught, 0 false flags**. Over five swap runs v2 catches 137 / 200 with 7 false flags. |
| False swap flags on runs with no swap (v2) | 3 in total over 18 clean Pythia runs; 0 / 0 / 0 / 2 on the four clean Qwen2.5-7B runs; 266 on r05, whose step count is one off. |
| Swap test on the fresh runs | **Not usable**: the seed was not recovered with confidence on any of the five, and without that gate the test flags 2,314-2,558 clean documents per run. |
| Membership inference (was this document in the training data?) | With the step curve taken from the weights alone, at a 0.1% false-positive rate: **43.6% [42.1, 45.2]** (m01, Alpaca), **45.0% [43.5, 46.6]** (g05, Wikipedia) and **46.0% [44.4, 47.6]** (n06, trained after the method was frozen); 95% Clopper-Pearson intervals. LiRA with 8 shadow fine-tunes finds 51.4%, 82.5% and 77.0% at the same point, so it is more accurate; this method needs no shadow training. Loss, reference-model and Min-K% attacks: 0.5-4.2%. The zero-false-positive point is unstable (5.9% on g05, 27.3% on n06) and is not a headline. |

Misses: r05 (warmup-stable-decay at batch 8) is estimated one step long (a 380-step flat stage leaves the
length bounded to about one step); the batch-4 WSD run is exact. The two-epoch Qwen run is not solved, also not
by a joint two-epoch fit (`results/gaps/GAPS.md`). r12 (cosine to a 0.1x floor with weight decay 0.1, the
Pythia-style schedule) was a regression in the first published results and is now exact.

What made the estimates work, beyond the thread's description: each token's change estimated as the *intercept*
of its sorted changes (Adam's eps shrinks small-gradient numbers), a mixture likelihood over steps with a
detection floor, decay strength tied to the total decay P for every candidate schedule, a coarse first pass
over step counts, an exact decay map taken from the untouched vectors (fp32 rounding drift is otherwise as
large as the changes), and weight decay and step count settled by bit-exact replay. The reader rounds the peak
and the weight decay to round values, which is why non-round recipes fail.

**Training setups** ([`results/stacks/STACKS.md`](results/stacks/STACKS.md), Pythia-160m, weight decay 0.1):

- Gradient clipping at 1.0 (active every step), gradient accumulation 4, two data-parallel ranks x accumulation 4,
  and DeepSpeed ZeRO-2 with DeepSpeed's fused AdamW: tokens (via replay), recipe and seed recovered, about 24.4M
  untouched weights bit-exact. The DeepSpeed run replays only with DeepSpeed's own weight-decay formula and the
  start cast to 16 bits as DeepSpeed holds it.
- Packed sequences: tokens, recipe and seed recovered, 24,357,120 weights bit-exact. Packing changes the
  last-token rule: all 68 tokens that occur only at the end of a document were updated, because only each
  packed chunk's last token gets no gradient.
- Pure 16-bit training (no 32-bit master copy): the token list is detected by a plain bit-identity test
  (18,538 / 18,538 and 18,526 / 18,538, no false positive); the recipe, seed and document steps are not recoverable.
- Clipping at 0.01 fails (Adam's eps dominates). Not tested: FSDP, ZeRO-3, Muon, full 7B fine-tuning.

### The 265-model roster

Every model of the checkpoint-forensics roster has a line in
[`results/roster/COVERAGE.md`](results/roster/COVERAGE.md): **186 of 265** give a pretraining estimate or a
fine-tune comparison, 52 more serve only as the reference parent of an analysed fine-tune, and the remaining
27 have no record (no untouched vectors, vectors rewritten after training, JAX/TPU or Ascend initialisation),
each with its reason.

**Base models: pretraining weight decay from never-trained token vectors** ([`PRETRAIN_DECAY.md`](results/roster/PRETRAIN_DECAY.md)).
Tokens that never occur in pretraining (and unused padding slots) keep their random starting values, shrunk
only by weight decay. checkpoint-forensics regenerates those starting values bit for bit; the spread that
reproduces the published vectors is init_std x P. So, *if init_std is known*, the weights give the total weight
decay applied in pretraining with no checkpoints at all.

- 175 models give an estimate: 37 only the list of never-trained tokens, 138 a comparison with the regenerated
  starting values. Of the 138, by how sure:
  - **8 Pythia sizes, 7 validated**: 70M, 160M, 410M, 1B, 1.4B, 6.9B and 12B agree with an independent
    measurement from the public checkpoints to 1e-4; 2.8B's checkpoint series is unusable, so it is not
    validated. The init that makes them agree is gpt-neox small_init for <= 2.8B and std 0.02 for 6.9B/12B;
  - **5 exactly no decay**: the spread equals the configured init exactly (OLMo-3 / OLMo-3.1 / OLMo-7B-Twin among them);
  - **24 no decay, inferred**: the spread is a round number, e.g. exactly 0.006 for DeepSeek-V2/V3/V4 (the init
    std in DeepSeek's paper); a decayed value is almost never round, but this is an inference;
  - **69 assumed**: a decay value that assumes the vectors started at the config's `initializer_range`; a wrong
    init convention moves P by ~2x, as Pythia shows;
  - **20 implausible for decay alone** (P < 3e-7 under the assumed init, e.g. Kimi-Linear): rescaled or zeroed otherwise;
  - **12 init unknown**: no value.
- GLM-4.5 through GLM-5.3 give 5.98-6.16 (assumed init), rising slightly release by release: most likely
  never-trained vectors inherited through continued training of one base, not a shared recipe.

**Fine-tunes: parent -> fine-tune comparisons** ([`SWEEP.md`](results/roster/SWEEP.md)).

Every public parent -> fine-tune pair in the roster whose token vectors can be compared: **229 pairs**
(197 with separate input and output matrices, 32 tied), token vectors fetched by HTTP range requests and
compared token by token. Across the roster and all follow-up studies there are 421 distinct comparisons over
580 models.

- **Seven classes** ([`results/frontier/FRONTIER.md`](results/frontier/FRONTIER.md) section 7,
  `scripts/frontier/comparability.py`, thresholds calibrated in `results/frontier/calibration.json`):

  | Class | Pairs | What it means |
  |---|---|---|
  | Token record | 97 | At least 50 *trainable* token vectors bit-identical after removing decay (unreachable, reserved, never-trained, zero and duplicate vectors are not counted), and the decay fit reproduces >= 99% or no shrink is resolvable. The list of tokens the fine-tune's data contained is detected. This shows shared ancestry, not a direct parent. |
  | Only untrainable tokens shared | 33 | The shared vectors are all untrainable (shared random start): no data or lineage evidence. |
  | Some bit-identical, decay fit below 99% | 25 | Token lists to be read with care. |
  | Every token changed, fit above every separately trained pair | 6 | Fit above 0.521 (largest value among 25 separately trained pairs: 0.471). |
  | Not a decayed copy of that base | 31 | e.g. Qwen3-235B-2507, Kimi-K2.5. |
  | Embedding unchanged | 5 | Re-uploads, conversions, frozen vectors; light training with a shrink below the 16-bit bound is not excluded. |
  | Shared input/output matrix (tied) | 32 | No record (e.g. Gemma). |

  Calibration: of 25 separately trained pairs sharing a tokenizer, 0 are classed "token record" (1 before the
  vectors were cleaned). This null is mostly Pythia variants (21 of 25); only Qwen2-7B vs Qwen2.5-7B shares any
  vector, so it is the one informative null pair. Of 39 distinct sibling pairs (two children of one parent), 12 are
  classed "token record". So no class names a direct parent. Before the review the classes were 114 / 11 / 25 / 42 / 5 / 32.
- **Lineage uses a margin rule** ([`results/frontier/lineage.md`](results/frontier/lineage.md)): a starting point is
  named only if its cleaned count of bit-identical vectors leads every other candidate (siblings, ancestors) by
  1.5x and 200 vectors. 16 statements hold; 9 are reworded to "shares never-trained vectors with X and Y".
- **Tied embeddings erase the record** (28 of 32 tied pairs keep fewer than 10 untouched vectors; 2 are
  unchanged), and so do separate output layers: in 105 of 107 measured, >= 99% of output vectors were updated or
  barely changed (93 if barely changed vectors are not counted). The output softmax sends a gradient to every token.
- **Special tokens.** Qwen2.5-Instruct updated the same 4 of 22 at 7B/14B/32B/72B (the thread's claim);
  Qwen2.5-Coder updated the fill-in-the-middle and repo/file tokens, Qwen2.5-Math only `<|endoftext|>`; the full
  table covers every family. An updated special token can also be a vendor edit (KAT-Dev's are zeroed).
- **Weight decay** is visible in 25 pairs (plus KAT-Dev, whose untouched vectors are zeroed, not decayed).
  21 pairs give a range of P that reproduces the same vectors (conditional on the fitted P; `p_intervals.json`,
  `results/frontier/decay_ranges.json`), widened by the largest simulated fp32 rounding drift. Qwen3-8B/14B/30B-A3B:
  a shrink of 1.30-1.31% for one multiplication, 1.25-1.37% with the drift allowance, for all three (the range is
  set by the 16-bit grid, so this means "agree within that range", not "same recipe"). Correction to our own first
  pass: five releases (Llama-3.1/3.3 Instruct, Nemotron-3-Ultra, LLaDA-MoE-v2) seemed to share P = 0.994148; their
  untouched vectors shrank by exactly one 16-bit step, which only bounds the total weight decay to [0.0039, 0.0059]
  ([0.0033, 0.0065] with drift), and the shared value is that range's edge, not a shared recipe.
- **Single-step timestamps do not survive public 16-bit releases** at today's batch sizes: only 5 pairs
  have >= 20 tokens with the single-step signature, and their changes are 1-3 16-bit steps. No learning rates
  are estimated from them.

### Follow-up studies

Each has its own generated report with sources; wording rules in [`STYLE.md`](STYLE.md). Article figures:
[`results/figures/`](results/figures/).

- **Frontier claims checked** ([`results/frontier/FRONTIER.md`](results/frontier/FRONTIER.md)). Qwen3-8B/14B
  leave the same 11 image/grounding tokens bit-identical; Qwen3-30B-A3B's special-token vectors were replaced,
  so it cannot be tested; Qwen3.5-35B-A3B leaves 11 of 18 image/audio tokens bit-identical. In 16-bit releases
  many vectors change too little to be told apart from untouched: DeepSeek-R1-Zero has 61.5% of its vectors with
  under 10% of their numbers changed (R1: 8.4%, V3-0324: 2.8%).
- **DeepSeek special tokens** ([`results/frontier_ds/DEEPSEEK.md`](results/frontier_ds/DEEPSEEK.md)).
  The nine tool tokens are at their regenerated random start in V3-Base; V3 updated the five tool-call tokens
  and left the four tool-output tokens bit-identical, V3-0324 updated all nine, R1 and R1-Zero none. V3-Base's
  800 reserved slots are about 5x their starting size with changes in a shared direction, consistent with
  appearing in pretraining input; DeepSeek's V4 releases updated all 800 and left 417 newer slots bit-identical.
  **V4 task tokens.** DeepSeek's own [encoding README](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md)
  documents `<｜action｜>`, `<｜query｜>`, `<｜authority｜>`, `<｜domain｜>`, `<｜title｜>` and `<｜read_url｜>` as tokens
  for auxiliary classification and generation tasks. The same block of the tokenizer holds `<｜task｜>`,
  `<｜political｜>`, `<｜entity｜>`, `<｜safety｜>`, `<｜answer｜>` and `<｜search｜>`, which that README does not
  describe. In both V4 bases all of these are at the *size* of a random start (the V4 starting values were not
  regenerated bit for bit, so light use in pretraining is not excluded). In V4-Flash and V4-Pro, documented and
  undocumented tokens changed alike: 93-99% of their numbers, by similar amounts, not zeroed, copied or rescaled,
  with changes that point in shared directions (38 of 45 pairs among ten of them, at both sizes). That is
  consistent with gradient updates on inputs that contain them; a deliberate small edit with a shared component
  is not ruled out. `<｜political｜>`'s change lines up with the documented authority and domain classifiers at
  both sizes, `<｜safety｜>`'s at one of the two. A third-party fine-tune of the same base (Nex-N2.5-Max) left all
  of them bit-identical. The weights do not say what the undocumented tasks are.
- **58 more frontier comparisons** ([`results/frontier_x/FRONTIER_X.md`](results/frontier_x/FRONTIER_X.md)):
  18 token records, 14 with only untrainable tokens shared, 8 with a poor fit, 3 changed throughout, 8 not
  comparable, 7 tied. Small-update footprints like R1-Zero's also occur in documented RL-only runs (Olmo-3-7B
  RL-Zero-Math: 53.6% of vectors with under 10% of their numbers changed, RL-Zero-Mix 91.9%, R1-Zero 61.5%), so
  the footprint is not by itself evidence of narrow data; it does not separate RL from low-learning-rate or
  narrow-data training either. Several successive releases are not decayed copies of the previous public
  checkpoint (MiniMax M2->M2.1 and M2.5->M2.7, GLM-5.1->5.2->5.3, Kimi K2-Thinking->K2.5, the LongCat-Flash
  chain). Kimi K2.5 is K2.6's closest published starting point (628 cleaned bit-identical vectors against at most
  13 for any other candidate). **Withdrawn:** "the DeepSeek chain V3->V3-0324, V3.1-Base->V3.1,
  V3.2-Exp->V3.2->V3.2-Speciale are direct descendants". V3.1-Base->V3.1 and V3.2-Exp-Base->V3.2-Exp hold under
  the margin rule; V3-0324 shares more cleaned vectors with V3-Base (1,160) and R1 (1,015) than with V3 (754),
  and V3.1-Terminus, V3.2 and V3.2-Speciale share never-trained vectors with two candidates each.
- **Which tokens each fine-tune's data contained** ([`results/tokens/TOKENS.md`](results/tokens/TOKENS.md)):
  116 fine-tunes with >= 10 bit-identical vectors, by script, language, code, digits, emoji and special tokens,
  with detection-floor, reachability, starting-size and hand-edit checks and a negative control
  (unrelated models sharing a tokenizer). Qwen2.5's base holds its chat and tool-call tokens at ~0; in the
  instruct releases they are updated, at all four sizes.
- **Quantized releases and merges** ([`results/quant_merge/QUANT_MERGE.md`](results/quant_merge/QUANT_MERGE.md)):
  of 77 quantized releases of 22 models, 42 keep the input token vectors equal to the official release, 13
  quantize them but still give the updated tokens with no false flags against the base quantized the same way
  (tokens with tiny updates are missed: about 0.1% in GGUF Q4_K/IQ3_S and MLX, 0.2-0.9% in Q8_0 copies of
  DeepSeek models), 6 are reproduced bit for bit from the official release, 6 could not be assessed because
  weight decay changed every vector, 6 are unscored, 3 not checked, 1 changed.
  `baseten/DeepSeek-V3-FP4`'s token vectors equal DeepSeek-V3-0324's for all 129,280 tokens.
  Merges (partial, on our own models): linear, SLERP, task-arithmetic, TIES and DARE keep tokens no
  parent updated bit-identical; model_stock does not.
- **Pretraining schedules from public checkpoint series** ([`results/series/SERIES.md`](results/series/SERIES.md)),
  learning rates assuming the published weight decay, with 16-bit bounds: OLMo-7B-0724's published schedule
  (warm-up from 0.1x peak) is reproduced with a median ratio of 1.0000 over 805 intervals, and 771 of 818 intervals
  lie inside the exact bounds; OLMoE to 0.08% per interval, with the cosine planned for 5.0T tokens though training
  ran to 5.13T; Apertus-8B's stable learning rate is 1.100e-4; Viking-7B follows its card within 1% in 8 of 9
  intervals (one interval 3.7% low, cause unknown); Amber's untouched vectors do not shrink for 259 of 360 chunks,
  which coincides with its checkpoints switching from 16-bit to 32-bit. Withdrawn: an earlier "OLMo-7B-0724 config
  disagrees with its card" (we compared the wrong config file) and "K2's floor is 0.092x" (within 16-bit rounding).
- **Public fine-tunes vs their published training data** ([`results/published/PUBLISHED.md`](results/published/PUBLISHED.md),
  controls in [`CONTROLS.md`](results/published/CONTROLS.md)).
  With a trainable input embedding, the tokens whose vectors changed should be exactly the tokens in the training
  inputs. Tokenizing each project's published data with its own training code (trl for s1, LLaMA-Factory for the
  others; the last position of each sequence excluded, since without packing it gets no gradient):

  | Fine-tune | Release | Tokens in data | Vectors changed | Verdict |
  |---|---|---|---|---|
  | simplescaling/s1-32B | float32 | 16,573 | 16,573 | exact match |
  | simplescaling/s1.1-32B | float32 | 27,154 | 27,154 | exact match at the documented 20,000-token block size |
  | NovaSky-AI/Sky-T1-32B-Preview | float16 | 48,015 | 48,015 | exact match |
  | bespokelabs/Bespoke-Stratos-32B | bfloat16 | 43,761 | 43,761 | exact match |
  | bespokelabs/Bespoke-Stratos-7B | bfloat16 | 43,761 | 43,761 | exact match |
  | open-thoughts/OpenThinker-7B | bfloat16 | 65,554 | 65,554 | exact match with the framework's default system prompt |

  Each match holds at exactly one tested combination of cutoff and system prompt, except s1, which matches at every
  block size from 16,384 up because its documents are shorter. That combination is the one the project documents for
  s1 (block size 32,768 in `train/sft.sh`), s1.1 (20,000 in the s1 README), Sky-T1 (cutoff 16,384 and the data's
  system prompt in SkyThought's LLaMA-Factory yaml) and OpenThinker-7B (cutoff 16,384 and no system column in its
  yaml, so the framework default); for Bespoke-Stratos it is documented only by reference (its cards point to
  Sky-T1's training code and list the same hyper-parameters). Sources, pinned to commits, are in
  [PUBLISHED.md](results/published/PUBLISHED.md#where-the-matching-setting-comes-from). s1.1
  matches only at 20,000 (49 in-data tokens that stayed bit-identical all first occur at position 20,022 or later,
  every changed token by 19,901). Controls: scoring each model against a different published dataset gives
  166-49,147 mismatches; leaving out a random 1% of documents would remove 60-104 tokens. What set equality
  cannot show: the absence of individual documents with no unique token, or extra documents whose tokens all
  occur elsewhere (an extra OpenThoughts-style document would add a new token only 2-31% of the time).
- **WebWorld 8B/14B/32B vs the published 52 GB training set** ([`results/published/WEBWORLD.md`](results/published/WEBWORLD.md)).
  No changed token lies outside the published data at 8B and 14B (2 at 32B); 99.0% of the data's vocabulary was
  updated (141,536 of 142,952); the weights point to a training length limit near 20,000 tokens, with a 0.2%
  residual the weights do not explain. Scored against the wrong data, 114,364 changed tokens fall outside it.
  The Hugging Face card metadata names Qwen3-8B as the base of all three sizes; the project's GitHub README names
  Qwen3-8B/14B/32B, and the weights agree with the README.
- **LIMO and LIMO-v2 vs their published data** ([`results/published/PUBLISHED_LIMO.md`](results/published/PUBLISHED_LIMO.md),
  rerunnable check with pinned revisions in `scripts/limo_check/`, outputs in `results/limo_check/out/`).
  GAIR/LIMO changed 29,742 token vectors. The published 817-example set has 13,386 distinct tokens (at a
  16,384-token cutoff), of which 13,183 changed and 203 never received a gradient; 16,559 changed tokens
  (5,391 of them CJK) do not occur in it. The DeepSpeed optimizer state left in an earlier revision of the model
  repository shows exactly those 29,742 vectors received gradients, in one run of 801-832 examples per epoch
  (epoch 9 of a planned 15; output directory named `aime817_deepseek-r1`), so a larger corpus is excluded. The
  questions are consistent with the published ones. The solution text used in training differs from the
  published one for at least 61 examples at a 16,384 cutoff (19-76 depending on the cutoff; 76 at 32,768). The
  16,559 absent tokens would need about 2,336 differing examples at the average rate of new tokens per example,
  more than the set holds, so the trained text of the differing examples must have been very unlike the published
  text. The weights do not show what that text was or why it differs. We report no evaluation-set coverage
  counts and make no contamination claim. A check against the logged `total_flos` was inconclusive.
  LIMO-v2: 279 changed tokens are absent from its published set (261 after tokenization variants), all ASCII.
- **Seed and batch size of public fp32 fine-tunes** ([`results/fp32scan/FP32SCAN.md`](results/fp32scan/FP32SCAN.md)).
  Of the popular models scanned, 630 are stored in fp32 and 173 of those name a base model. For s1-32B and
  s1.1-32B (fp32 master weights, public data and script), the published weight decay of 1e-4 is a no-op in fp32.
  Assuming the published cosine schedule with 5% warm-up, beta2 0.95, 5 epochs and Trainer's seedable sampler,
  seed 42 with batch 16 fits best of 6,003 candidates for both, and stays first in 300 of 300 bootstrap
  resamples. Batch 16 matches the published script; seed 42 is the Trainer default (the script sets none).
  Individual documents' steps follow from the seed but are only weakly confirmed one at a time.
- **Reflection-Llama-3.1-70B** ([`results/fp32scan/FP32SCAN_REFLECTION.md`](results/fp32scan/FP32SCAN_REFLECTION.md),
  independent check with pinned revisions in `results/reflection_check/`). The card names Llama-3.1-70B-Instruct
  as base. The first upload (5 Sep 2024) is bit-identical to Llama-3.1-70B-Instruct in every tensor we sampled.
  The current weights (6 Sep 2024) derive from Llama-3-70B-Instruct, not 3.1: all normalisation weights and
  51,241 token vectors are bit-identical to it and none to any Llama 3.1/3.3 model; in 35 sampled projection
  matrices the difference has rank at most 256; the embedding and output matrix changed differently. The
  repository's config.json names Meta-Llama-3-70B-Instruct. A third upload is listed in the report. References
  were verified byte-identical to Meta's releases by sha256. This agrees with public reports from September 2024;
  the training method is not proven, and the weights say nothing about intent.
- **Large fine-tunes of frontier models** ([`results/large/LARGE.md`](results/large/LARGE.md)), 29 comparisons.
  Tulu-3-405B SFT against its published mixture: all but 245 of the 124,572 tokens in the data were updated
  (39 of the 245 occur more than 5 times) and no other regular token changed; "nothing outside" is tested on the
  3,684 vocabulary tokens the mixture lacks. The DPO stage does not match its mixture (2,188 in-data tokens
  bit-identical, 399 changed outside). Hermes-4 405B's cleaned set of 2,461 bit-identical tokens lies inside the
  70B release's 2,482; 97% of them are also bit-identical in Hermes 3, so this is weak evidence about Hermes 4's
  own data. r1-1776 and MAI-DS-R1 against DeepSeek-R1 are low-signal: category counts only. The end-of-sequence
  token is bit-identical after DPO/RL stages (Tulu, MAI-DS-R1, AceReason, Skywork-OR1), consistent with unpacked
  training. Lineage under the margin rule: AM-Thinking-v1 from the Qwen2.5-32B base, Nemotron-Ultra-253B from
  Llama-3.1-405B-Instruct; QwQ-32B and Hermes-4-405B share never-trained vectors with two candidates each.
  Kimi-Dev-72B and KAT-Dev-72B-Exp have the same 2,171 vectors set to zero (unused slots and unreachable
  tokens); an observation, with no claim about how.
- **2026 lab releases** ([`results/y2026_labs/LABS2026.md`](results/y2026_labs/LABS2026.md)): 435 releases from
  42 lab accounts found, 78 new comparisons. Qwen-AgentWorld-35B-A3B's card names the Qwen3.5 base; it shares
  614 never-trained vectors with Qwen3.6-35B-A3B and 594 with KAT-Coder-V2.5-Dev (another Qwen3.6 fine-tune) against 13
  with the Qwen3.5 models, so the weights place it in the Qwen3.6 line without naming a starting point. Atria-Dawn-Preview's
  closest published starting point is GLM-5.2; Sage-S2.1 shares never-trained vectors with both GLM-5.2 and
  Atria-Dawn-Preview. Several 2026 flagship successors are new token vectors, not trained copies of the previous
  public release (Kimi K3 line, MiniMax M3, Step-5, Qwen3.6/3.8, MiMo V2.6); some special tokens stay
  bit-identical after post-training (fill-in-the-middle tokens in Ling and Hy3).
- **2026 fine-tunes** ([`results/y2026_ft/FT2026.md`](results/y2026_ft/FT2026.md)), 32 comparisons of 29 distinct
  fine-tunes. Arete-27B's token vectors match its published SFT answers (63,392 of 63,900 tokens in the training
  text updated, nothing outside it; the reasoning traces were not trained on, as the dataset card states). The
  weights together with the per-example token counts Nous published are consistent with a plain concatenation
  without chat markers; the weights alone do not single that format out (`results/arete_check/`; Nous's data file
  is fetched by `scripts/arete_check/fetch_nous.py`, not redistributed). 13 of the 29 distinct fine-tunes (32 comparisons) leave the input embedding
  bit-identical (LoRA tunes, trace distills, refusal-removal edits), so they carry no token record.
  MemT-1.0-Flash's card names DeepSeek-V4-Flash; its closest published starting point is V4-Flash-0731 (16,806
  cleaned bit-identical vectors against 82).
- **Previously gated models** ([`results/gated/GATED.md`](results/gated/GATED.md)): open copies of Llama weights
  verified byte-identical to the official releases by sha256. Apertus-v1.5-70B base -> SFT -> RL -> release is a
  clean chain; Hydrogen-1.0's closest published starting point is GLM-5.3 (103,495 against 24), as its card says;
  Cohere's tiny-aya shares one matrix for input and output, so it has no record.

## Where the thread holds and where it does not (here)

Each line is a claim from the thread, our verdict, and the evidence. "Exact" means checked against ground truth
after the analysis; scope and exceptions are in the controlled-runs section above.

| Thread claim | Here | Evidence |
|---|---|---|
| Updated vectors = tokens in the data (39,663 = 39,663) | **holds** | Exact in all 24 controlled runs (14k-45k tokens, Pythia and Qwen2.5-7B, LoRA, batch up to 128) and with DeepSpeed ZeRO-2, two ranks and packing; with weight decay, exact via replay. Exception: gradient clipping at 0.01 (replay gives 31,766 false positives). |
| Recipe from the weights alone: schedule, steps, warm-up right in 15/15; peak within 0.01% | **mostly, for round recipes** | 22 / 24; step count from the weights alone right in 24 / 25 Pythia runs. On 5 fresh runs with non-round values: 2 / 5. |
| 3,121 / 4,000 docs placed in their exact step | **partly** | From the weights alone: 767-1,714 of 4,000 (19-43%), fewer at larger batch; wrong placements up to 253. With the recovered seed: all. |
| Shuffle seed found in every run | **holds within the search range** | Recovered in every pre-existing run where it was searched (seeds 0-99,999); random 32-bit seeds are not searched. |
| 38 / 40 secretly swapped docs exposed, no false flag | **partly** | v2 test: 27 / 40 caught, 0 false flags on s05; 137 / 200 with 7 false flags over five swap runs; 266 false flags when the step count is one off; unusable without a confident seed (fresh runs). |
| Replay of a stranger's public fine-tune: 25,463,296 weights matched bit for bit | **not tested on a public model yet** | On our own runs: Qwen2.5-7B, weight decay 0.1: 473,769,019 weights bit-exact; one step off: ~1,100. Pythia-160m: about 24.4M, one step off: 26-279. |
| Membership: 64% at zero false alarms; strongest standard test 0.5% | **lower here** | 44-46% at a 0.1% false-positive rate on three sets; LiRA with 8 shadow fine-tunes reaches 51-83% at the same point. The zero-false-positive point is unstable (5.9-41%). |
| Two epochs: both timestamps within 1% for 93-97% of twice-seen tokens | **lower here** | 61-69% when the two sightings are ~5,000 steps apart; 0.7% when 500 apart. The two-epoch Qwen run is unsolved, also by a joint fit. |
| Large batch: sorted changes follow a known curve whose intercept is the timestamp; OLMo 2's learning rates within 2% | **partly** | Recipe exact at batch 128 (our largest). The OLMo 2 learning-rate estimate was not tested. |
| 16-bit releases: token list stays exact | **holds, with one exception** | bf16 and fp16 exact without weight decay; with decay to lr ~0 at the end, 4-89 last-step tokens round away (their changes are below half a 16-bit step). |
| Adam + L2: replaying Adam on the decay term alone recovers the token list | **holds** | 31,766 / 31,766 never-seen tokens (24.4M weights) bit-exact, 0 seen tokens. |
| LoRA with trained token vectors: whole record there | **holds** | Recipe exact, seed found, tokens exact (Pythia and Qwen2.5-7B). |

## Reproduce

Full instructions, environment variables and pinned revisions: [`REPRODUCE.md`](REPRODUCE.md).

```bash
pip install -e '.[controlled,test]'
make test        # offline unit tests (CPU)
make all-offline # regenerate every report from the committed JSON (no network; uses data/ckptf if the sibling repo is absent)
make grid        # controlled fine-tunes with ground truth (one GPU)
make score table # blind analysis, scored
make public      # the four public claims (CPU + network; set HF_TOKEN)
make roster      # token-vector comparisons over the roster pairs (needs the checkpoint-forensics checkout)
```

## Limitations

- **Bit-identical counts do not name a parent.** Siblings and ancestors share never-trained vectors; lineage
  needs the margin rule, and the "token record" class only shows shared ancestry.
- **16-bit resolution.** A total shrink below about 0.2% and updates below the detection floor are invisible in a
  16-bit release; "bit-identical" there means "no change above the floor".
- **Round recipes only, so far.** The reader rounds the peak learning rate and weight decay to round values and
  searches five schedule families and seeds below 100,000; outside that it fails visibly (the replay check drops).
- **Weight decay needs the exact decay map and the trainer's own formula.** Every number is multiplied by an
  fp32-rounded (1 - lr x wd) each step; the rounding drift is comparable to the changes at low learning rates.
  DeepSpeed's fused AdamW needs its own decay formula and a 16-bit-cast start.
- **Multi-epoch recipes** are not solved here (q7_2ep).
- **Tied embeddings** (Qwen2.5-0.5B/1.5B/3B, Gemma, many small models): every token gets a gradient through
  the output softmax, so no vector stays untouched. **LoRA on attention only** leaves the token vectors untouched.
- **Full fine-tuning of a 7B on one GPU** did not fit (AdamW states); 7B runs train token vectors + output
  layer, LoRA, or token vectors only. The mechanism only involves the token vectors' gradients.
- **Controlled data** is Alpaca (short instructions), plus Wikipedia for two membership sets;
  placement, swap and membership rates depend on how many tokens are unique to a document.
- **Published-data matches are set equalities.** They cannot exclude a few missing documents without unique
  tokens, or extra documents inside the same token set.
- **Public 16-bit models** give the list of tokens and, with weight decay or checkpoints, the total weight decay
  or the schedule; document-level results need the training data.

## Credit and license

The method and the four public claims are from Alapan Chaudhuri's [thread](https://x.com/alapanchaudhuri/status/2121439269772492848) (14 Nov 2026); this is an
independent replication and extension. MIT. Not affiliated with any of the labs whose models are examined.
The results describe what the published weights record; they say nothing about intent. If we got your model
wrong, open an issue and we will correct it.
