# Public claims, checked from public weights

Four claims from the thread, each tested from published checkpoints alone (HTTP range reads
of the embedding rows; nothing was trained). Every number below is in the JSON next to this file.

| # | Claim | Verdict |
|---|-------|---------|
| 1 | Pythia-1B was trained at peak LR 2.5e-4, not the 3.0e-4 on its model card | **CONFIRMED** |
| 2 | Pythia's LR sat flat for its last ~28,000 steps: a cosine aimed at zero, clipped at 0.1x peak, not today's gpt-neox scheduler | **CONFIRMED** |
| 3 | OLMo 2's first instruct release left ~20,341 embedding rows frozen that should have trained, among them the token that closes every chat-turn marker | **CONFIRMED** (count within 2.6%, depending on definition) |
| 4 | Qwen2.5 instruct tuning touched exactly 4 of 22 control tokens, the same four at 7B/14B/32B/72B | **CONFIRMED** |

## 1-2. Pythia learning-rate schedule (`claim1_pythia_lr.json`, `claim1b_pythia_clean_fit.json`)

**Method.** gpt-neox pads the 50,277-token vocabulary to 50,304+ rows. Rows >= 50,277 are never
looked up, so they get no gradient; with decoupled weight decay (wd = 0.1) they only shrink by
(1 - lr_t * wd) each step. Reading those rows at the 154 public `stepN` branches gives
-log(<W_s,W_0>/<W_0,W_0>) = wd * sum_{t<=s} lr_t: the cumulative LR, sampled every 1,000 steps.
We fit candidate schedules (and a free 3-parameter clipped cosine: peak, clip level, horizon).

| model | yml config lr | model card lr | **recovered peak** (free fit) | clip / peak | flat tail (steps) |
|---|---|---|---|---|---|
| 70m | 1.0e-3 | 1.0e-3 | 9.983e-4 | 0.1000 | 27,657 |
| 160m | 6.0e-4 | 6.0e-4 | 5.991e-4 | 0.1001 | 27,694 |
| 410m | 3.0e-4 | 3.0e-4 | 2.995e-4 | 0.1000 | 27,685 |
| **1b** | **2.5e-4** | **3.0e-4** | **2.50002e-4 ± 5e-9** | 0.1000 | 27,865 |
| 1.4b | 2.0e-4 | 2.0e-4 | 1.997e-4 | 0.1000 | 27,681 |
| 2.8b | 1.6e-4 | 1.6e-4 | 1.597e-4 (tail-only, see below) | 0.1000 | 27,678 |
| 6.9b | 1.2e-4 | 1.2e-4 | 1.198e-4 | 0.0999 | 27,686 |
| 12b | 1.2e-4 | 1.2e-4 | 1.198e-4 | 0.1000 | 27,705 |

- Pythia-1B: the weights give 2.500e-4 and agree with `models/1B/pythia-1b.yml` (0.00025); the
  model card's 3.0e-4 is not what was run. The whole-run ratio alone (step 143000 vs step 0,
  ratio 0.157203) gives 2.4999e-4 under the Pythia-era shape.
- Flat tail. The cosine 0.5(1+cos(pi x/T)) with T = 143,000 reaches 0.1 at x = 113,709, i.e.
  step 115,139 after the 1,430-step warm-up: predicted flat tail 27,861 steps. Measured: free
  fits 27,657-27,865 at every size; model-free, every 1,000-step interval from 115,000-116,000
  onward has the same mean LR within 2% (0.1000 x peak), i.e. 28,000 steps at checkpoint resolution.
- Not today's scheduler. Pythia-era gpt-neox (`d7af1e7a8e`) computes
  `max(peak/2*(cos(pi*n/T)+1), min_lr)` (cosine aimed at 0, clipped). gpt-neox `main` computes
  `min_lr + (peak-min_lr)/2*(cos(pi*n/(T-W))+1)` (no flat tail when lr_decay_iters = train_iters).
  Fit residuals (RMS per 1,000 steps / peak): Pythia-era 0.005-0.023 vs today's 0.52-1.11 at every size.

**Side findings (checkpoint hygiene, not in the thread).**
- pythia-12b branches `step50000` and `step58000` carry the *final* model's padding rows
  (ratio to step 0 = 0.41153, equal to step143000), not step-50000/58000 weights.
- pythia-2.8b: padding rows are bit-identical to `step0` in 85 of the other 153 branches (all 75
  up to step65000, plus 70000, 80000, 90000, 100000, 103000, 110000, 120000, 130000, 140000 and
  the final 143000), while the remaining branches decay normally. In every other size the rows
  have left step 0 by step 1000. The fit uses only the decaying series from step66000 (68 checkpoints),
  so 2.8b's peak comes from the tail shape alone; it still lands at 1.597e-4 (config 1.6e-4).
- pythia-1b stores fp32 tensors with bf16 precision (low 16 bits zero in 100% of values) and
  `step116000` is off-series. 214 real vocabulary rows of pythia-1b are pure rescales of their
  initial values (never trained in pretraining); 191 of them are whitespace / CRLF runs.

## 3. OLMo 2 first SFT release (`claim2_olmo2_sft.json`)

**Method.** Compare `model.embed_tokens.weight` of `allenai/OLMo-2-1124-{7B,13B}` (base) with the
first SFT release (`-SFT-Preview`, uploaded 2024-11-25) and the retrained `-SFT` (2024-12-18),
row by row, bit for bit after rounding the base to bf16. Tokenize the full
`allenai/tulu-3-sft-olmo-2-mixture` (939,344 examples, ~0.8B tokens, chat template applied)
with the base tokenizer (cl100k-style pre-tokenizer) and with the preview's tokenizer
(vocab.json + merges.txt only, so transformers falls back to the GPT-2 pre-tokenizer regex).

| | first release (Preview) | retrained SFT |
|---|---|---|
| rows bit-identical to base (of 100,352) | 23,138 (74 padding + 23,064 real) | 2,876 (74 + 2,802) |
| frozen rows whose token occurs in the data (base tokenizer) | **20,878** | 554 |
| ... and absent from the data under the preview tokenizer | 20,632 | 334 |
| ... and occurring >= 2 times | 20,358 | 359 |
| rows that moved although their token is absent from the data under the tokenizer used | **0** | 0 |
| `>Ċ` (id 397), which closes every `<\|role\|>\n` marker (2,389,430 occurrences) | **never moved** | moved |

13B gives the same counts (20,878 / 2,803). The preview tokenizer splits `|>\n` into `>` + `Ċ`,
so `>Ċ` and ~20.6k other merged tokens never appeared in training; the moved set is exactly the
set of tokens present under the preview tokenization. AI2's card on `-SFT`: "the post-trained
models did not share the pre-tokenization logic that the base models use. As a result, we have
trained new post-trained models."

**Caveat.** The thread's 20,341 is not reproduced exactly; our counts under three reasonable
definitions are 20,358-20,878 (0.1-2.6% off). The difference is probably definitional (e.g. last
token per example excluded, truncation at the training max length, dataset snapshot); we did not
pin it down.

## 4. Qwen2.5 control tokens (`claim3_qwen25_control_tokens.json`, `claim4_qwen25_adam_signature.json`)

**Method.** Compare `model.embed_tokens.weight` of `Qwen/Qwen2.5-{S}` vs `-Instruct`, bit for bit,
for the 22 added tokens (ids 151643-151664).

- 7B, 14B, 32B, 72B: exactly 4 of 22 moved, the same four at every size: `<|im_start|>`,
  `<|im_end|>`, `<tool_call>`, `</tool_call>`. Bit-identical: `<|endoftext|>`, the vision/object/
  box/quad tokens, the FIM tokens, `<|repo_name|>`, `<|file_sep|>`. The four moved rows are
  ~zero in the base models.
- Untouched real-vocabulary rows: 7,247 / 7,362 / 7,370 / 7,198 of 151,643 (7B/14B/32B/72B), all
  bit-identical (scale 1.0): no decoupled weight decay reached the embedding. One padding row
  (>= 151,665) moved at 72B.
- 0.5B/1.5B/3B tie embeddings to the LM head, so every row moves (gradient through the softmax);
  the test does not apply there. We did not diff `lm_head` at 7B+: every head row gets gradient
  through the softmax denominator, so it carries no token-presence signal.
- Extra check: no single-touch "flat" rows in Qwen2.5-7B-Instruct (per-row CV of |delta| ~0.71-0.75,
  like Gaussian noise 0.756), as expected for large-batch tuning where every row is hit many times.

## Links
- Pythia-1B card (3.0e-4): https://huggingface.co/EleutherAI/pythia-1b ; config (0.00025): https://github.com/EleutherAI/pythia/blob/main/models/1B/pythia-1b.yml
- gpt-neox scheduler, Pythia era: https://github.com/EleutherAI/gpt-neox/blob/d7af1e7a8e/megatron/learning_rates.py ; today: https://github.com/EleutherAI/gpt-neox/blob/main/megatron/learning_rates.py
- OLMo 2 SFT: https://huggingface.co/allenai/OLMo-2-1124-7B-SFT , https://huggingface.co/allenai/OLMo-2-1124-7B-SFT-Preview ; data: https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture
- Qwen2.5: https://huggingface.co/Qwen/Qwen2.5-7B , https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

Reproduce: `scripts/public/claim1_pythia_lr.py`, `claim1b_pythia_clean_fit.py`, `claim2_olmo2_sft.py`,
`claim3_qwen25_control_tokens.py`, `claim4_qwen25_adam_signature.py` (CPU only; set `HF_TOKEN` to avoid rate limits).
