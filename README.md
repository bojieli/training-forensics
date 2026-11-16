# training-forensics

What a model's published weights reveal about how it was trained: which tokens were in the data, the
learning-rate schedule, the number of steps, and (given the dataset) which optimizer step each document
was used in. This repository is an independent replication of a public thread by Alapan Chaudhuri
([@alapanchaudhuri](https://x.com/alapanchaudhuri), [thread](https://x.com/alapanchaudhuri/status/2121439269772492848), 14 Nov 2026) that made these claims, tested three ways:
on the four public models it names, on 24 controlled fine-tunes with full ground truth (Pythia-160m/410m
and Qwen2.5-7B), and on the 265-model roster of the sibling project
[checkpoint-forensics](https://github.com/bojieli/checkpoint-forensics).

## Terms

- **Token vector (embedding).** Every token in a model's vocabulary has its own vector of numbers. In the
  code and tables this is one *row* of the embedding matrix; counts of rows are counts of distinct
  vocabulary tokens, never amounts of text.
- **Updated / untouched.** A token vector is *updated* (*moved* in table columns) when training changed it
  beyond weight decay, and *untouched* (*unmoved*) when it is bit-identical to where it started (after the
  common weight-decay shrink).
- **Parent -> fine-tune pair.** A model and the model trained from it, e.g. Qwen2.5-7B -> Qwen2.5-7B-Instruct.
- **Shrink factor P and total weight decay.** With decoupled weight decay an untouched vector is multiplied by
  (1 - lr x wd) at every step; P is the product over all steps and -ln P = sum of lr x wd, the *total weight
  decay applied*. It gives the product of learning rate and weight decay, summed over the run, not either one alone.
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
4. D depends only on the learning rates from that step on: D(s) = sum_k lr(s+k) a(k, s). With
   default Adam settings and full bias correction D/lr -> 31.77. So D says *when*; the sign pattern says
   *what* (minus the sign of that step's gradient).
5. Under decoupled weight decay every number goes through the same sequence of fp32 multiplications,
   so the untouched vectors can be replayed bit for bit once the schedule is known.

`trainforensics/adam.py` is the forward model; `trainforensics/blind.py` does the estimation;
`scripts/controlled/score.py` analyses a run blind and only then opens its ground truth.

## Results

### The four public claims (`results/public/PUBLIC_CLAIMS.md`)

| Claim | Verdict | What the weights show |
|---|---|---|
| Pythia-1B peak learning rate is 2.5e-4, not the card's 3.0e-4 | **confirmed** | 2.50002e-4, estimated from how the unused padding vectors shrank over 154 checkpoints; the yml config agrees. All 8 sizes land within 0.2% of their configs. |
| Pythia's learning rate is flat for its last ~28,000 steps (cosine to 0 clipped at 0.1x peak) | **confirmed** | clip at 0.1000x peak at every size; flat tail 27,657-27,865 steps (predicted 27,861). Today's gpt-neox scheduler fits 50-200x worse. |
| OLMo-2 first SFT release: ~20,341 tokens that should have been trained were never updated, incl. the chat-turn closer | **confirmed, count differs** | 23,064 real tokens bit-identical to base; 20,878 of them occur in the SFT data (20,358-20,878 under three definitions; the gap to 20,341 is unexplained). `>Ċ` (id 397, 2.39M occurrences) never updated; it was updated in the retrain. |
| Qwen2.5 instruct updated exactly 4 of 22 special tokens, same 4 at 7B/14B/32B/72B | **confirmed** | `<\|im_start\|>`, `<\|im_end\|>`, `<tool_call>`, `</tool_call>` at all four sizes; the other 18 bit-identical. |

Side findings: pythia-12b checkpoints `step50000`/`step58000` contain the final model's padding vectors;
pythia-2.8b's padding vectors are bit-identical to step 0 in 85 of 153 later checkpoints (incl. the final one).

### Controlled fine-tunes (`results/controlled/CONTROLLED.md`, `results/gaps/GAPS.md`)

24 fine-tunes with full ground truth: Pythia-160m/410m (full fine-tuning) and Qwen2.5-7B (token vectors +
output layer, LoRA with trained token vectors, weight decay, batch 128, two epochs), linear / cosine / constant /
cosine-to-floor / warmup-stable-decay schedules, batch 4-128, weight decay 0-0.1. Each run is analysed blind
from (start, release) and only then scored against the trainer's logged learning rates and batches.
All 19 Pythia runs were re-scored with one version of the code (`results/gaps/rescore_commit.txt`). The five
Qwen2.5-7B runs were scored once on a borrowed GPU machine whose copy of the weights was later deleted; their
results are kept and marked *old*.

| What is recovered or estimated | Result |
|---|---|
| Tokens seen (updated vectors = tokens in the data) | **recovered exactly in 24 / 24**, no false positive, from the direct comparison or (with weight decay) the replay of the decay. |
| Recipe: schedule, steps, warm-up, peak (and weight decay) | **exact in 18 / 19 re-scored runs** (22 / 24 with the old Qwen2.5-7B results); peak within 0.016% before rounding to a round value in most runs (r09: 0.12%) |
| Shuffle seed (HF Trainer) | **recovered in every run where it was searched** (all but the two-epoch run, including r05 with its one-step-off recipe) |
| Documents in their exact step, from the weights alone | 4,000-document runs: 767-1,714 (19-43%); 8,000 docs: 16%; batch 128 (52,000 docs): 4.5%. Wrong placements are 0-7 in most runs; 51 in r09, 232 in r05, 33-254 in the old Qwen2.5-7B runs. **All documents** once the seed is known. |
| Replay of untouched weights | bit-exact: 473,769,020 weights (Qwen2.5-7B, old); one step off leaves ~1,100 (≈ the exact zeros). Pythia: 24.4M (r12 included); one step off leaves 26-279. |
| 40 secretly swapped documents (s05) | v1: 24 caught, 3 false flags; **v2: 27 caught, 0 false flags**. Over five swap runs v2 catches 137 / 200. |
| False swap flags on runs with no swap (v2) | 0-2 on 18 of 19 Pythia runs (3 in total); 266 on r05, whose step count is one off. Old Qwen2.5-7B (v1 only): 26-367. |
| Membership inference (was this document in the training data?), held out | Development set (Alpaca, m01): 42% at zero false alarms, 44% at a 0.1% false-alarm rate. **Held-out set (Wikipedia, g05), never used for development: 6.7% at zero false alarms (5-95% over splits: 5.4-44.3%), 45% at 0.1%.** Loss, reference-model and Min-K% attacks: <= 2.2%. LiRA with 8 shadow fine-tunes on the true recipe beats this method at 0.1% false alarms (54% / 83%) and is unstable at zero. |

Misses: r05 (warmup-stable-decay at batch 8) is estimated one step long (a 380-step flat stage leaves the
length bounded to about one step); the batch-4 WSD run is exact. The two-epoch Qwen run (most tokens seen
twice) is not solved. r12 (cosine to a 0.1x floor with weight decay 0.1, the Pythia-style schedule) was a
regression in the first published results and is now exact: recipe, seed, 0 wrong tokens, all 24,396,288
untouched weights replayed bit for bit (`results/gaps/GAPS.md` section 2).

What made the estimates work, beyond the thread's description: each token's change estimated as the *intercept*
of its sorted changes (Adam's eps shrinks small-gradient numbers), a mixture likelihood over steps with a
detection floor, decay strength tied to the total decay P for every candidate schedule, a coarse first pass
over step counts, an exact decay map taken from the untouched vectors (fp32 rounding drift is otherwise as
large as the changes), and weight decay and step count settled by bit-exact replay.

**Training setups** ([`results/stacks/STACKS.md`](results/stacks/STACKS.md), Pythia-160m, weight decay 0.1):
gradient clipping at 1.0 (active every step), gradient accumulation 4, two data-parallel ranks x accumulation 4,
and DeepSpeed ZeRO-2 with DeepSpeed's fused AdamW: tokens, recipe and seed recovered, ~24.4M untouched
weights bit-exact. The DeepSpeed run replays only with DeepSpeed's own weight-decay formula and the start cast
to 16 bits as DeepSpeed holds it (PyTorch's formula: 968,587 weights). Clipping at 0.01 fails (Adam's eps
dominates). Not tested: FSDP, ZeRO-3, Muon, packed sequences, pure 16-bit training, full 7B fine-tuning.

### The 265-model roster

Every model of the checkpoint-forensics roster has a line in
[`results/roster/COVERAGE.md`](results/roster/COVERAGE.md): **182 of 265** give a pretraining estimate or a
fine-tune comparison, 52 more serve only as the reference parent of an analysed fine-tune, and the remaining
31 have no record (no untouched vectors, vectors rewritten after training, JAX/TPU or Ascend initialisation),
each with its reason.

**Base models: pretraining weight decay from never-trained token vectors** ([`PRETRAIN_DECAY.md`](results/roster/PRETRAIN_DECAY.md)).
Tokens that never occur in pretraining (and unused padding slots) keep their random starting values, shrunk
only by weight decay. checkpoint-forensics regenerates those starting values bit for bit; the spread that
reproduces the published vectors is init_std x P. So, *if init_std is known*, the weights give the total weight
decay applied in pretraining with no checkpoints at all.

- 175 models give an estimate: 37 only the list of never-trained tokens, 138 a comparison with the regenerated
  starting values. Of the 138, by how sure:
  - **8 validated** (Pythia): agree with an independent measurement from the public checkpoints to 1e-4 at
    70M, 160M, 410M, 1B, 1.4B, 6.9B and 12B (2.8B's checkpoint series is corrupted). The init that makes them
    agree is gpt-neox small_init for <= 2.8B and std 0.02 for 6.9B/12B;
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
compared token by token.

- **Six classes** ([`results/frontier/FRONTIER.md`](results/frontier/FRONTIER.md), `scripts/frontier/comparability.py`):

  | Class | Pairs | What it means |
  |---|---|---|
  | Token record | 114 | Some vectors bit-identical after removing decay, and the decay fit reproduces >= 99%; or no decay (shrink exactly 1) and at least 100 vectors bit-identical (e.g. DeepSeek-V4 vs its bases): the list of tokens the fine-tune's data contained is recovered. |
  | Some bit-identical, decay fit below 99% | 11 | Token lists to be read with care (e.g. GLM-4.5). |
  | Every token changed, consistent with the base plus decay | 25 | Same base, broad training (e.g. MiMo). |
  | Not a decayed copy of that base | 42 | e.g. Qwen3-235B-2507, Kimi-K2.5, MiniMax-M1. |
  | Embedding unchanged | 5 | Re-uploads, conversions, frozen vectors. |
  | Shared input/output matrix (tied) | 32 | No record (e.g. Gemma). |

  The per-category token analysis ([`results/tokens/TOKENS.md`](results/tokens/TOKENS.md)) uses 116 fine-tunes
  with >= 10 bit-identical vectors.
- **Tied embeddings erase the record** (28 of 32 tied pairs keep fewer than 10 untouched vectors; 2 are
  unchanged), and so do separate output layers (in 105 of 107, >= 99% of output vectors were updated): the
  output softmax sends a gradient to every token.
- **Special tokens.** Qwen2.5-Instruct updated the same 4 of 22 at 7B/14B/32B/72B (the thread's claim);
  Qwen2.5-Coder updated the fill-in-the-middle and repo/file tokens, Qwen2.5-Math only `<|endoftext|>`; the full
  table covers every family. An updated special token can also be a vendor rewrite (KAT-Dev's are zeroed).
- **Weight decay** is visible in 25 pairs (plus KAT-Dev, whose untouched vectors are zeroed, not decayed).
  23 pairs give an exact range for the total weight decay, e.g. Mistral-Nemo-Instruct [0.01993, 0.01997];
  Qwen3-8B/14B/30B-A3B all [0.01311, 0.01318] (the range is set by the 16-bit grid, so this means "agree within
  0.5%", not "same recipe"). Correction to our own first pass: five releases (Llama-3.1/3.3 Instruct,
  Nemotron-3-Ultra, LLaDA-MoE-v2) seemed to share P = 0.994148; their untouched vectors shrank by exactly one
  16-bit step, which only bounds the total weight decay to [0.0039, 0.0059], and the shared value is that
  range's edge, not a shared recipe.
- **Single-step timestamps do not survive public 16-bit releases** at today's batch sizes: only 5 pairs
  have >= 20 tokens with the single-step signature, and their changes are 1-3 16-bit steps. No learning rates
  are estimated from them.

### Follow-up studies

Each has its own generated report with sources; wording rules in [`STYLE.md`](STYLE.md).

- **Frontier claims checked** ([`results/frontier/FRONTIER.md`](results/frontier/FRONTIER.md), figures in
  `results/frontier/fig/`). A comparability map of all 229 roster comparisons: 114 with a clean token record
  (the decay fit reproduces >= 99% of the untouched vectors, or no decay and >= 100 bit-identical vectors;
  see `results/frontier_ds/RECONCILE.md`), 11 partial, 25 with every vector changed but
  consistent with the base plus decay, 42 not a decayed copy of that base, 5 unchanged, 32 tied. Qwen3-8B/14B
  leave the same 11 image/grounding tokens bit-identical; Qwen3-30B-A3B's special-token vectors were replaced,
  so it cannot be tested; Qwen3.5-35B-A3B leaves 11 of 18 image/audio tokens bit-identical. Qwen3-8B/14B/30B-A3B
  vectors shrank by 1.30-1.31% (equal within the 16-bit grid). In 16-bit releases many vectors change too
  little to be told apart from untouched: DeepSeek-R1-Zero has 61.5% of its vectors with under 10% of their
  numbers changed (R1: 8.4%, V3-0324: 2.8%).
- **DeepSeek special tokens** ([`results/frontier_ds/DEEPSEEK.md`](results/frontier_ds/DEEPSEEK.md)).
  The nine tool tokens are at their regenerated random start in V3-Base; V3 updated the five tool-call tokens
  and left the four tool-output tokens bit-identical, V3-0324 updated all nine, R1 and R1-Zero none. V3-Base's
  800 reserved slots have moved from their start (5x larger, changes in a shared direction), consistent with
  appearing in pretraining input; DeepSeek's V4 releases updated all 800 and left 417 newer slots bit-identical.
  V4's named special tokens (`<｜political｜>`, `<｜safety｜>`, `<｜authority｜>` and their group) start at their
  random init in both V4 bases and were updated in DeepSeek's V4-Flash and V4-Pro like trained tokens (not
  zeroed, copied or rescaled); a third-party fine-tune of the same base (Nex-N2.5-Max) left them bit-identical.
  Names and status only.
- **58 more frontier comparisons** ([`results/frontier_x/FRONTIER_X.md`](results/frontier_x/FRONTIER_X.md),
  rows in `results/roster/sweep_frontier_x.jsonl`, not yet in the coverage table): 24 clean records, 3 partial,
  8 trained throughout, 16 not comparable, 7 tied. RL stages leave a distinctive footprint of bit-identical and
  barely changed vectors (Olmo-3-7B RL-Zero-Math: 38% bit-identical, 45% with 1-10% of numbers changed), the
  same shape as R1-Zero, so narrow data is not needed to explain R1-Zero. Several successive releases are not
  decayed copies of the previous public checkpoint (MiniMax M2->M2.1 and M2.5->M2.7, GLM-5.1->5.2->5.3,
  Kimi K2-Thinking->K2.5, the LongCat-Flash chain); the DeepSeek chain V3->V3-0324, V3.1-Base->V3.1,
  V3.2-Exp->V3.2->V3.2-Speciale are direct descendants.
- **Which tokens each fine-tune's data contained** ([`results/tokens/TOKENS.md`](results/tokens/TOKENS.md)):
  116 fine-tunes with >= 10 bit-identical vectors, by script, language, code, digits, emoji and special tokens,
  with detection-floor, reachability, starting-size and hand-edit checks and a clean negative control
  (unrelated models sharing a tokenizer). Qwen2.5's base holds its chat and tool-call tokens at ~0; the
  instruct tune first trained them, at all four sizes.
- **Quantized releases and merges** ([`results/quant_merge/QUANT_MERGE.md`](results/quant_merge/QUANT_MERGE.md)):
  42 of 77 quantized releases of 22 models keep the input token vectors equal to the official release (every
  FP8, NVFP4, MXFP4, AWQ, GPTQ, w4a16 and bitsandbytes release checked). llama.cpp Q8_0 is reproduced bit for bit,
  and against the base quantized the same way the updated tokens come back with zero false flags (also MLX and
  GGUF Q4_K/IQ3_S, <= 0.1% missed). `baseten/DeepSeek-V3-FP4`'s token vectors equal DeepSeek-V3-0324's.
  Merges (partial): every mergekit method except model_stock keeps tokens no parent updated bit-identical.
- **Pretraining schedules from public checkpoint series** ([`results/series/SERIES.md`](results/series/SERIES.md)),
  learning rates assuming the published weight decay, with exact 16-bit bounds: OLMo-7B-0724's published schedule
  (warm-up from 0.1x peak) is reproduced to within 0.1% over 818 intervals; OLMoE to 0.08% per interval, with
  the cosine planned for 5.0T tokens though training ran to 5.13T; Apertus-8B's stable learning rate is 1.100e-4;
  Viking-7B follows its card within 1% in 8 of 9 intervals (one interval 3.7% low, cause unknown); Amber's
  untouched vectors do not shrink for 259 of 360 chunks, which coincides with its checkpoints switching from
  16-bit to 32-bit. Withdrawn: an earlier "OLMo-7B-0724 config disagrees with its card" (we compared the wrong
  config file) and "K2's floor is 0.092x" (within 16-bit rounding).
- **Public fine-tunes vs their published training data** ([`results/published/PUBLISHED.md`](results/published/PUBLISHED.md)).
  With a trainable input embedding, the tokens whose vectors changed should be exactly the tokens in the training
  inputs. Tokenizing each project's published data with its own training code (trl for s1, LLaMA-Factory for the
  others; the last position of each sequence excluded, since it gets no gradient):

  | Fine-tune | Release | Tokens in data | Vectors changed | Verdict |
  |---|---|---|---|---|
  | simplescaling/s1-32B | float32 | 16,573 | 16,573 | exact match |
  | simplescaling/s1.1-32B | float32 | 27,154 | 27,154 | exact match at the documented 20,000-token block size |
  | NovaSky-AI/Sky-T1-32B-Preview | float16 | 48,015 | 48,015 | exact match |
  | bespokelabs/Bespoke-Stratos-32B | bfloat16 | 43,761 | 43,761 | exact match |
  | bespokelabs/Bespoke-Stratos-7B | bfloat16 | 43,761 | 43,761 | exact match |
  | open-thoughts/OpenThinker-7B | bfloat16 | 65,554 | 65,554 | exact match with the framework's default system prompt |

  The same pipeline gives 0 changed-but-absent tokens on our own controlled runs. In s1.1-32B the 49 tokens of
  its data that stayed bit-identical all first occur at position 20,022 or later, and every changed token by
  position 19,901: the weights give the training context length (the s1 README documents 20,000 for s1.1).
- **LIMO and LIMO-v2 vs their published data** ([`results/published/PUBLISHED_LIMO.md`](results/published/PUBLISHED_LIMO.md),
  independent check in `results/limo_check/`). The released GAIR/LIMO checkpoint (epoch 9 of a 15-epoch run over
  about 817 examples, read from the DeepSpeed optimizer state left in an earlier revision of the model repository)
  received gradient updates on 16.5k vocabulary tokens, including about 5.4k CJK tokens, that do not occur in the
  published 817-example set, while about 200 tokens of the published solutions were never updated. The questions
  match; the solution texts used in training differ from the published ones for at least about 60 examples. The run
  size excludes a larger corpus. The weights do not show what the other text was or why it differs. Token sets
  cannot show that an evaluation question was trained on; the evaluation-set coverage counts in the file are not
  evidence of contamination and we make no such claim. LIMO-v2: 261 updated tokens not explained by its published set.
- **Seed and batch size of public fp32 fine-tunes** ([`results/fp32scan/FP32SCAN.md`](results/fp32scan/FP32SCAN.md)).
  630 popular fine-tunes are stored in fp32. For s1-32B and s1.1-32B (fp32 master weights, public data and script),
  the published weight decay of 1e-4 is a no-op in fp32 (each step's factor rounds to exactly 1.0, and all
  never-updated vectors are bit-identical); a scan of 2,001 seeds x 3 batch sizes picks seed 42 (the Trainer
  default) and batch 16 (315 steps) for both, a cosine schedule and Adam beta2 0.95. Individual documents' steps
  follow from the seed but are only weakly confirmed one at a time. DeepScaleR-1.5B (RL) keeps a clean token
  record (103,986 bit-identical); UserLM-8b and marin-8b-instruct changed every token vector.
- **Reflection-Llama-3.1-70B** ([`results/fp32scan/FP32SCAN_REFLECTION.md`](results/fp32scan/FP32SCAN_REFLECTION.md),
  independent check in `results/reflection_check/`). The card names Llama-3.1-70B-Instruct as base. The first upload
  (5 Sep 2024) is bit-identical to Llama-3.1-70B-Instruct in every tensor we sampled. The current weights (6 Sep 2024)
  match Llama-3-70B-Instruct instead: all normalisation weights and 51,241 token vectors are bit-identical to it, none
  to any Llama 3.1/3.3 model, the remaining matrices differ from it by a rank-256 update, and the repo's config.json
  names Meta-Llama-3-70B-Instruct. References were verified byte-identical to Meta's releases by sha256. This agrees
  with public reports from September 2024; the training method (e.g. LoRA) is not proven, and the weights say nothing
  about intent.
- **Large fine-tunes of frontier models** ([`results/large/LARGE.md`](results/large/LARGE.md)), 29 comparisons.
  Tulu-3-405B SFT against its published mixture: all but 245 of the 124,572 tokens in the data were updated
  (mostly tokens occurring once or twice) and none outside it. Hermes-4 70B and 405B leave the identical set of
  2,739 tokens bit-identical. r1-1776 and MAI-DS-R1 against DeepSeek-R1 are low-signal (most changed vectors
  differ in < 10% of their numbers): category counts only. The end-of-sequence token is bit-identical after
  DPO/RL stages (Tulu, MAI-DS-R1, AceReason, Skywork-OR1), consistent with unpacked training: a sequence's last
  token gets no gradient. Lineage agrees with the cards (QwQ-32B and AM-Thinking-v1 from the Qwen2.5-32B base;
  Nemotron-Ultra-253B from Llama-3.1-405B-Instruct, shrink 0.765-0.769%). Kimi-Dev-72B and KAT-Dev-72B-Exp zero
  the identical 2,171 vectors (unused slots and unreachable tokens).
- **2026 lab releases** ([`results/y2026_labs/LABS2026.md`](results/y2026_labs/LABS2026.md)): 435 releases from
  42 lab accounts found, 78 new comparisons. Weights point to bases the cards do not name, or name differently
  (e.g. WebWorld-14B: 10,389 vectors bit-identical to Qwen3-14B, while its card names Qwen3-8B, whose vectors have
  a different width), consistent with card mistakes; several 2026 flagship successors are new token vectors, not
  trained copies of the previous public release (Kimi K3 line, MiniMax M3, Step-5, Qwen3.6/3.8, MiMo V2.6);
  some special tokens stay bit-identical after post-training (fill-in-the-middle tokens in Ling and Hy3).
- **2026 fine-tunes** ([`results/y2026_ft/FT2026.md`](results/y2026_ft/FT2026.md)), 32 compared. Arete-27B's
  token vectors match its published SFT answers (99.2% of the vocabulary in the training text updated, nothing
  outside it; the reasoning traces were not trained on, as the dataset card states), and the weights give the
  training text format: a plain concatenation without chat markers, confirmed by the per-example token counts
  Nous published (`results/arete_check/`). 13 of the 32 leave the input embedding bit-identical (LoRA tunes,
  trace distills, refusal-removal edits), so they carry no token record.

## Where the thread holds and where it does not (here)

Each line is a claim from the thread, our verdict, and the evidence. "Exact" always means checked against
ground truth that the analysis never saw.

| Thread claim | Here | Evidence |
|---|---|---|
| Updated vectors = tokens in the data (39,663 = 39,663) | **holds** | Exact in every run (18.5k-45k tokens, Pythia and Qwen2.5-7B, LoRA, batch up to 128, DeepSpeed ZeRO-2, two ranks); with weight decay, exact via replay. |
| Recipe from the weights alone: schedule, steps, warm-up right in 15/15; peak within 0.01% | **mostly** (18 / 19 re-scored; 22 / 24 with old Qwen runs) | Peak typically within 0.016% before rounding. Misses: r05 (one step long), q7_2ep (two epochs). |
| 3,121 / 4,000 docs placed in their exact step | **partly** | From the weights alone: 767-1,714 of 4,000 (19-43%), fewer at larger batch; wrong placements up to 254. With the recovered seed: all. |
| Shuffle seed found in every run | **holds** | Recovered in every run where it was searched (not attempted on the two-epoch run) (HF Trainer = `torch.randperm` seeded with seed + epoch). |
| 38 / 40 secretly swapped docs exposed, no false flag | **partly** | v2 test: 27 / 40 caught, 0 false flags on s05; 137 / 200 over five swap runs. False flags on clean runs: 3 over 18 Pythia runs, 266 on r05 (step count one off). |
| Replay of a stranger's public fine-tune: 25,463,296 weights matched bit for bit | **not tested on a public model yet** | On our own runs: Qwen2.5-7B, weight decay 0.1: 473,769,020 weights bit-exact; one step off: ~1,100 (1,095 exact zeros). Pythia: 24.4M, one step off: 26-279. |
| Membership: 64% at zero false alarms; strongest standard test 0.5% | **lower here** | Held-out Wikipedia set: 6.7% at zero false alarms (5-95%: 5.4-44.3%), 45% at a 0.1% false-alarm rate; development set 42%. Loss / reference / Min-K%: <= 2.2%, but LiRA with 8 shadow fine-tunes reaches 54-83% at 0.1% false alarms. |
| Two epochs: both timestamps within 1% for 93-97% of twice-seen tokens | **lower here** | 61-69% when the two sightings are ~5,000 steps apart; 0.7% when 500 apart (Adam's second moment still remembers the first: beta2^500 = 0.61). A joint fit was written but not finished. |
| Large batch: sorted changes follow a known curve whose intercept is the timestamp; OLMo 2's learning rates within 2% | **partly** | Intercept fit estimates 93.6% of single-step tokens within 0.3% even when only 3% have flat changes; recipe exact at batch 128 (our largest). The OLMo 2 learning-rate estimate was not tested. |
| 16-bit releases: token list stays exact | **holds, with one exception** | bf16 and fp16 exact without weight decay; with decay to lr ~0 at the end, 51-89 last-step tokens round away (their changes are below half a 16-bit step). |
| Adam + L2: replaying Adam on the decay term alone recovers the token list | **holds** | 31,766 / 31,766 never-seen tokens (24.4M weights) bit-exact, 0 seen tokens. |
| LoRA with trained token vectors: whole record there | **holds** | Recipe exact, seed found, tokens exact (Pythia and Qwen2.5-7B). |


## Reproduce

```bash
pip install -e '.[controlled]'
make test        # offline unit tests of the forward model (CPU)
make grid        # controlled fine-tunes with ground truth (one GPU)
make score table # blind analysis, scored
make public      # the four public claims (CPU + network; set HF_TOKEN)
make roster      # token-vector comparisons over the roster pairs
```

## Limitations

- **Weight decay needs the exact decay map and the trainer's own formula.** Every number is multiplied by an
  fp32-rounded (1 - lr x wd) each step; the rounding drift (up to ~N/2 smallest fp32 steps) is comparable to the
  changes at low learning rates. We take the map from the untouched vectors as a lookup table (exact when the
  start was stored in 16 bits, as released models are) and pick weight decay by bit-exact replay. DeepSpeed's
  fused AdamW needs its own decay formula and a 16-bit-cast start.
- **Multi-epoch recipes** are not solved here: with most tokens seen twice, the one-sighting curve does not
  apply (q7_2ep).
- **Tied embeddings** (Qwen2.5-0.5B/1.5B/3B, Gemma, many small models): every token gets a gradient through
  the output softmax, so no vector stays untouched. **LoRA on attention only** leaves the token vectors untouched.
- **Full fine-tuning of a 7B on one GPU** did not fit (AdamW states); 7B runs train token vectors + output
  layer, LoRA, or token vectors only. The mechanism only involves the token vectors' gradients.
- **Controlled data** is Alpaca (short instructions), plus Wikipedia for the held-out membership test;
  placement, swap and membership rates depend on how many tokens are unique to a document.
- **Public 16-bit models** give the list of tokens and, with weight decay or checkpoints, the total weight decay
  or the schedule; document-level results need the training data.


## Credit and license

The method and the four public claims are from Alapan Chaudhuri's [thread](https://x.com/alapanchaudhuri/status/2121439269772492848) (14 Nov 2026); this is an
independent replication and extension. MIT. Not affiliated with any of the labs whose models are examined.
The results describe what the published weights record; they say nothing about intent.
