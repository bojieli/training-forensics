# Frontier releases: lab-specific claims, checked

Scripts: `scripts/frontier/`. Data: `results/frontier/*.json`. Figures: `results/frontier/fig/`.
Wording follows `STYLE.md`.

**Method.** Every token in a model's vocabulary has its own vector of numbers. For each token, we compare the
release with its base model after removing the uniform shrink that weight decay applies. We then report the
fraction of the token's numbers that still differ:

| State | Fraction of numbers that differ |
|---|---|
| bit-identical | 0 |
| barely changed | 1% or less |
| partly changed | 1–10% |
| updated | more than 10% |

These checks apply Fable's editorial checklist:
- a per-token change histogram;
- whether the weight-decay fit reproduces the base (at least 99%);
- the starting size of the token vector in the base;
- the detection floor (`detection_floor.json`);
- whether the token appears in the release's own chat template;
- agreement across model sizes.

"Bit-identical" is an observation about the released files. It says nothing about intent.

## Verdicts

| # | Claim | Verdict |
|---|---|---|
| 1 | DeepSeek tool tokens: V3 left the tool-output tokens bit-identical, V3-0324 updated them, R1 updated none | **holds, with wording** |
| 2 | DeepSeek reserved placeholder slots | moved to another agent |
| 3 | DeepSeek V4 named special tokens | moved to another agent |
| 4 | Qwen3 vision tokens left bit-identical | **holds for 8B and 14B; 30B-A3B is not testable** (its special-token vectors were replaced) |
| 5 | Qwen3 8B, 14B and 30B-A3B share the same total weight decay | **holds** |
| 6 | Llama-4 Scout tokens that grew from near zero | **holds, with wording; partly unexplained** |
| 7 | Comparability map of all 229 base → release comparisons | built (figure d); classes recalibrated after the review |

---

## 1. DeepSeek tool-call tokens: holds, with wording

**Data:** `special_tokens.json`, `detection_floor.json`.

DeepSeek-V3-Base has nine tool tokens: tool-calls begin/end, tool-call begin/end, tool-outputs begin/end,
tool-output begin/end, and tool-sep. The table gives the state of each release relative to its base.

| Release (vs its base) | 5 tool-call tokens | 4 tool-output tokens | Comparison usable? |
|---|---|---|---|
| DeepSeek-V3 | updated (88–96% of numbers changed) | **bit-identical** | yes: decay fit reproduces 100% |
| DeepSeek-V3-0324 | updated | updated (87–92%) | yes: 100% |
| DeepSeek-R1 | **bit-identical** | **bit-identical** | yes: 100% |
| DeepSeek-R1-Zero | **bit-identical** | **bit-identical** | yes: 100% |
| DeepSeek-V3.2 (vs V3.2-Exp-Base) | updated | updated (49–61%) | yes: 100% |
| DeepSeek-V4-Pro / V4-Flash (vs their bases) | updated | updated | yes: 446/447 vectors bit-identical, no decay (shrink 1); the 512-vector fit score (92%/91%) is capped by how few vectors are untouched, not a sign of a different base |
| V4-Pro-0813, V4-Flash-0731, V4.1-Flash-1017 (vs the previous release) | updated | updated | yes: 100% |

**Checks**
- **Detection floor.** A change of 1e-6 to every number would have altered at least 449 (V3) and 414 (R1) of each
  vector's 7,168 numbers. Any single number would show a change as small as 7.5e-9. Typical post-training learning
  rates are 1e-6 to 1e-5, so if these tokens had been in the training data their vectors would have changed. Here,
  bit-identical does mean "received no update".
- **Starting size.** In V3-Base all nine tool-token vectors have size 0.504–0.516. That equals the starting size,
  init std 0.006 × √7168 = 0.508. A typical trained token has size 3.2. So pretraining did not train these tokens
  either.
  - In the V4 bases the same tokens have size 1.3–1.7: they were trained before V4's post-training.
- **Reachability.** All nine tokens appear in the chat templates of V3 and R1.
- **R1-Zero.** It also left its three fill-in-the-middle code tokens bit-identical. R1 updated them, by 2–24%.

**Sentence the article may use:**
> "Compared with DeepSeek-V3-Base, the released DeepSeek-V3 left the vectors of its four tool-output tokens
> bit-identical, while updating the five tool-call tokens. DeepSeek-V3-0324 updated all nine. DeepSeek-R1 and
> R1-Zero left all nine bit-identical. These tokens were also untouched by pretraining: their vectors are still at
> their random starting size in the base. This is consistent with R1's post-training data containing no tool
> calls, and with V3's containing tool calls but no tool outputs."

**Do not say:**
- "R1 cannot use tools."
- "DeepSeek removed tool use."

---

## 4. Qwen3 vision and grounding tokens

**Data:** `special_tokens.json`, `qwen3_vision.json`, `detection_floor.json`.

### Qwen3-8B and Qwen3-14B: holds

- The same 11 tokens are bit-identical after the decay shrink, in both models:
  object_ref start/end, box start/end, quad start/end, vision start/end, vision_pad, image_pad and video_pad.
- All other 15 named special tokens were updated, by 88–99% of their numbers:
  - chat markers;
  - tool call and tool response;
  - think;
  - the six code fill-in tokens.
- The comparisons are usable: decay fit 99.98% and 100%.
- Detection floor: a 1e-6 change would have altered at least 172 numbers (8B) and 216 numbers (14B) per vector.
- None of the 11 tokens appears in the chat templates.

### Qwen3-30B-A3B: not testable for special tokens

The claim that "all 26 were updated" is wrong. The special-token vectors in the release are unrelated to those in
the base:

| Measure | Value |
|---|---|
| Cosine between release and base special-token vectors (median, 26 tokens) | 0.001 |
| Same, for ordinary tokens | 0.998 |
| Same, for Qwen3-8B special tokens (control) | 0.999 |
| Cosine between the change and the base vector | −0.71 |

- A cosine of −0.71 between the change and the base vector is what an independent new vector of the same size
  produces (−1/√2).
- The new vectors are the same size as the old ones (ratio 0.987).
- The Instruct-2507 and Thinking-2507 releases also carry special-token vectors unrelated to both the base and the
  original 30B-A3B release (cosine 0.002).
- So the 30B-A3B special tokens were replaced, or come from a different starting point than the published base. We
  cannot say whether they were trained.
- The ordinary vocabulary does follow the base: decay fit 99.9%, total weight decay identical to 8B/14B (claim 5).

### Qwen3.5-35B-A3B (vs its base)

- 11 of the 18 vision/audio tokens are bit-identical: vision_pad, the three object_ref/quad tokens, and all seven
  audio/TTS tokens.
- vision start/end, image_pad and video_pad were updated. These four appear in the chat template.
- Two code fill-in tokens are also bit-identical: fim_pad and repo_name.

**Sentence the article may use:**
> "Qwen3-8B and Qwen3-14B left the same 11 image, video and grounding tokens bit-identical to their base models
> (apart from the uniform weight-decay shrink), while updating every chat, tool, thinking and code token. That is
> consistent with text-only post-training. For Qwen3-30B-A3B the test cannot be run: its special-token vectors are
> unrelated to the published base's."

---

## 5. Qwen3 total weight decay: holds

**Data:** `decay_ranges.json`, from `results/roster/p_intervals.json`.

- For Qwen3-8B, Qwen3-14B and Qwen3-30B-A3B the never-updated token vectors all shrank by a factor in the same
  exact interval, [0.98691, 0.98698]:
  - 8B: 2,160 vectors;
  - 14B: 2,126 vectors;
  - 30B-A3B: 833 vectors.
- That is a shrink of 1.30–1.31%, or total weight decay Σ lr·wd between 0.01311 and 0.01318, for one multiplication.
  Allowing for fp32 rounding drift over many training steps (`drift_sim.json`) widens it to 1.25–1.37% for all three.
- This is the range of shrink that reproduces the same vectors, set by the 16-bit grid. So "identical" means the three
  agree within that range, not proof of one recipe.

**Sentence the article may use:**
> "In Qwen3-8B, 14B and 30B-A3B, the token vectors that post-training never updated shrank by the same amount, between
> 1.25% and 1.37%: the same total weight decay to within the precision of the 16-bit release."

---

## 6. Llama-4 Scout: holds, with wording; partly unexplained

**Data:** `scout.json`.

- In Llama-4-Scout-17B-16E (base), 1,117 of the 1,135 special and reserved token vectors are essentially zero:
  size about 1.3e-13.
- In the Instruct release they fall into three groups:

| Group | Count | What happened |
|---|---|---|
| Chat and tool tokens: header_start, header_end, eom, eot, python_start, python_end | 6 | Grew from zero to size 0.085–0.10, with Gaussian-like values (spread 0.76; a Gaussian gives 0.756) and mutually unrelated directions (median cosine 0.06). Three of them are in the chat template. |
| Reserved slots: 5 text_post_train_reserved, 43 vision_reserved | 48 | Grew from zero to tiny sizes, 1e-5 to 1.4e-4, about 1/1000 of a working token. Their values are irregular (spread 0.15–1.1). **Unexplained.** |
| Other reserved slots | 1,064 | Shrank exactly as weight decay predicts; "barely changed" only through rounding. |

**Sentence the article may use:**
> "Llama-4 Scout's base model ships its chat-header and tool tokens as all-zero vectors. In the Instruct release
> those six vectors are fully formed, so they were created during post-training. 48 reserved slots also moved off
> zero, but only to about a thousandth of a normal token's size. We cannot explain that."

---

## 7. Comparability map: built

**Data:** `comparability.json`. **Figure:** `fig/d_comparability.png`.

All 229 base → release comparisons fall into seven classes (`comparability.json`, lane `roster`; rules and thresholds in
`scripts/frontier/comparability.py`, calibrated in `calibration.json`):

| Class | Comparisons | What it means |
|---|---|---|
| Token record | 97 | At least 50 trainable token vectors are bit-identical after removing decay (unreachable, reserved, never-trained, zero and duplicate vectors are not counted), and the decay fit reproduces at least 99% or no shrink is resolvable. Evidence of shared ancestry, not of a direct parent. |
| Only untrainable tokens shared | 33 | The bit-identical vectors are all untrainable (shared random start). No data or lineage evidence. |
| Some bit-identical, decay fit below 99% | 25 | Treat token lists with care. |
| Every token changed, fit above every separately trained pair | 6 | Fit above 0.521, the threshold set from 25 separately trained pairs (largest 0.471). |
| Not a decayed copy of that base | 31 | e.g. Qwen3-235B-2507, Kimi-K2.5. |
| Embedding unchanged | 5 | |
| Shared input/output matrix | 32 | No record, e.g. Gemma. |

- **Calibration.** 25 separately trained pairs with the same tokenizer: 0 classed "token record" (1 before the cleaning). 40 sibling
  pairs (two children of one parent): 13 classed "token record". So the class shows shared ancestry and never names a direct parent;
  lineage statements use the margin rule in `lineage.md`.
- **Before the review** the classes were 114 / 11 / 25 / 42 / 5 / 32 (`comparability_before_after.json`).
- **No resolvable shrink.** In a 16-bit release a fitted shrink of exactly 1 means "below the resolvable bound" (median 0.195% over the
  180 such comparisons, `no_shrink_resolvable_below_pct`), not "no weight decay".

- **The fit metric** is the fraction of numbers reproduced exactly by round(shrink × base), in the 512 vectors
  closest to a pure shrink.
- **It is lenient.** A low value is strong evidence that the release is not a decayed copy of that base.
- **Class boundaries** are set in `scripts/frontier/comparability.py`.
- **No-decay rule** (`results/frontier_ds/RECONCILE.md` section 1): with shrink exactly 1, a bit-identical vector is an
  exact copy of the base, which only a continuation of that base produces. The fit score looks at the 512 vectors
  closest to a pure shrink; when fewer than 512 are untouched, trained vectors fill the sample and cap the score.
  With at least 100 bit-identical vectors such a comparison now counts as a token record (16 moved up, from 98/27
  to 114/11). Kimi K2's bit-identical vectors are mostly unreachable byte tokens: they
  prove the lineage but say nothing about data.
- **Stale fit scores fixed.** The first sweep batch (8 records) was scored by an earlier version of the fit code. Recomputed
  with the current code on the same cached files (`scripts/frontier/recheck_fit.py`, `recheck_fit*.json`): DeepSeek-V4-Flash
  0.293 -> 0.912, Trinity-Large 0.276 -> 0.895, Leanstral-1.5 0.608 -> 1.000, Trinity-Mini 0.998 -> 1.000, kanana-2 0.753 -> 0.763,
  INTELLECT-3 0.029 -> 0.133, Baichuan-M3 0.013 -> 0.000. Ten random later records reproduce exactly. The corrected values are in
  `results/roster/sweep.jsonl` (originals kept as `P_exact_frac_original`); no class count changes beyond the no-decay rule.

---

## Figures

| Figure | File | Data | Status |
|---|---|---|---|
| (a) Share of numbers changed per token vector, DeepSeek-V3-0324 vs R1 vs R1-Zero (all vs V3-Base) | `fig/a_change_histogram.png` | `fig/a_change_histogram.json` | ready |
| (b) Special-token grid for DeepSeek | built by the other agent | | not here |
| (b2) Special-token grid for other labs | `fig/b2_special_grid_nondeepseek.png` | `special_grid_nondeepseek.json` | ready |
| (c) Total weight decay ranges by release | `fig/c_decay_ranges.png` | `fig/c_decay_ranges.json` | ready |
| (d) Comparability map | `fig/d_comparability.png` | `fig/d_comparability.json` | ready |

Notes on (a):
- **Scope.** Each series covers all 129,280 vocabulary tokens.
- **The "61%" figure** for R1-Zero is the share of token vectors with under 10% of their numbers changed, excluding
  bit-identical ones: 14.8% barely + 46.6% partly = 61.5%.

| Release | Bit-identical | Under 10% changed (not bit-identical) | Over 10% changed |
|---|---|---|---|
| DeepSeek-R1-Zero | 8.0% | 61.5% | 30.5% |
| DeepSeek-R1 | 3.3% | 8.4% | 88.3% |
| DeepSeek-V3-0324 | 1.9% | 2.8% | 95.4% |

Notes on (b2):
- **Categories** are assigned by token name (`scripts/frontier/special_grid.py`).
- **Reserved slots** are left out of the grid.
- **Testable releases** are those in the "token record" class. GLM-4.5 is marked not testable; Kimi-K2-Thinking is now
  testable under the no-decay rule.
