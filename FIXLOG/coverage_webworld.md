# Coverage regeneration and WebWorld audit (Fork F1)

For the final integration pass; README.md was not edited (an earlier README edit in 1d26715 was reverted in e2b7042).

## Coverage (`scripts/roster/coverage.py` -> `results/roster/COVERAGE.md`, `coverage.json`, `comparisons_all.json`)

- Sources: roster `sweep.jsonl` + `sweep_cml3.jsonl`, `sweep_frontier_x.jsonl`, `sweep_2026_labs.jsonl`,
  `results/large/sweep_large.jsonl`, and `results/y2026_ft/pairs/*.json` (own `cls` field), deduplicated by
  (parent, fine-tune), first source wins (roster first).
- Classes are not hard-coded: each sweep record is classified by importing `classify` from
  `scripts/frontier/comparability.py` at run time (y2026_ft records keep their own `cls`). After Fix-A's classifier
  rework, rerun `python3 scripts/roster/coverage.py` and the counts below update; new class names are appended
  to the table automatically.
- Not included yet: `results/roster/sweep_gated.jsonl` (appeared during this pass, owned by another lane);
  add it to `SWEEPS` in coverage.py if it should count.

Totals with the classifier as of commit e2b7042:

| | before | after |
|---|---|---|
| roster models with a pretraining estimate or a fine-tune comparison | 182 / 265 | 186 / 265 |
| roster models compared as a fine-tune | 50 | 74 |
| roster models only a reference parent | 52 | 52 |
| roster models with no record | 31 | 27 |
| distinct parent -> fine-tune comparisons (all lanes) | 229 (roster only) | 421 over 580 models |

All 421 comparisons by class: token record 203, record with decay fit < 99% 17, changed throughout 50,
not comparable 85, unchanged 17, shared input/output matrix 49. By source: roster 229, frontier_x 58,
y2026_labs 77, large 28, y2026_ft 29. The roster's 229 keep 114 / 11 / 25 / 42 / 5 / 32.

Suggested README sentence (for the integration pass): "186 of 265 roster models give a pretraining estimate or a
fine-tune comparison, 52 more serve only as a reference parent, 27 have no record. Across the roster and all
follow-up studies there are 421 distinct parent -> fine-tune comparisons over 580 models."

## WebWorld audit (`results/published/WEBWORLD.md`)

Scripts: `scripts/published/ww_tokens.py` (token counts, earliest positions, Qwen3 chat template, whole
string), `ww_lengths.py` (per-sample lengths: drop-by-length test), `ww_fetch.py`, `ww_compare.py`,
`ww_finescan.py`, `ww_report.py`. Data and results in `results/published/webworld/`. PUBLISHED.md links it
(through `scripts/published/report.py`).

- Models Qwen/WebWorld-8B, -14B, -32B vs Qwen3-8B / -14B / -32B; data Qwen/WebWorldData (52.2 GB,
  1,107,028 conversations, 15.66B tokens with the chat template; the dataset card says 1,059,348).
- **Nothing outside the published set was updated**: counting whole conversations, all 141,536 changed regular
  tokens of the 8B occur in the data (14B: 0 outside, 32B: 2).
- **99.0% of the data's vocabulary was updated** (141,536 of 142,952); the 1,416 never-updated tokens occur mostly
  late in long conversations.
- **Length limit near 20,000 tokens**: a 250-token scan of the truncation length gives a sharp kink and the best
  fit at 20,000 at all three sizes (59 changed tokens beyond it, 266 in-data tokens within it never updated;
  145/232 at 19,000). Dropping long conversations entirely fits worse.
- Residual ~325 tokens (0.2%) not explained by one limit; candidate causes (other truncation rule, tokenization of
  some conversations, published conversations not in the run: 47,680 more than the card's count) are not
  distinguished by the weights.
- Same at three sizes (10,122-10,127 bit-identical): consistent with one training set.
- Wrong-data control (s1K-1.1 as if it were the data): 114,364 changed tokens "not in the data" instead of 0.
- Special tokens: chat markers, `<think>`/`</think>`, `<tool_call>`/`</tool_call>` updated; `<|endoftext|>`
  (one conversation per sequence), vision, FIM and `<tool_response>` tokens bit-identical.
- Card metadata names Qwen3-8B as the base of all three sizes; the GitHub README names Qwen3-8B/14B/32B and the
  weights agree with the README (14B/32B vectors are 5,120 wide).

Suggested README line: "WebWorld 8B/14B/32B match their published 52 GB training set: every updated token occurs in
it, 99.0% of its vocabulary was updated, and the weights point to a 20,000-token training length limit (0.2% residual)."

Downloads (52 GB dataset, 7 embeddings) deleted after the run; derived counts kept.
