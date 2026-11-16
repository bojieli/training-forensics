# Wording guide (all docs, README, article)

Write for a reader who knows what a language model is but not linear algebra.

Define once, early: "Every token in a model's vocabulary has its own vector of numbers (its embedding).
If a token never appears in training, its vector is never updated."

| Avoid | Use |
|---|---|
| row / rows (of the embedding) | token embedding, token vector; for counts: "N vocabulary tokens" / "N distinct tokens" |
| moved / unmoved | updated / bit-identical to the base model (never updated) |
| moved every row | every token vector changed |
| read (as in "read the recipe") | recover / reproduce exactly (bit-exact), estimate / infer (fit with error), bound (interval only), detect / flag (yes/no tests) |
| pair | base model and its fine-tune (parent -> child is fine in tables, define once) |
| P, sum lr*wd (in prose) | "total weight decay applied" (define the formula once, in a footnote or method section: shrink factor P = prod(1 - lr*wd), total = -ln P) |
| tied | models that share one matrix for input and output (tied embeddings) - this erases the record |
| control tokens | special tokens (chat markers, tool-call, thinking, image placeholders) |
| bf16 / ulp | 16-bit release format; "one smallest 16-bit step" instead of ulp |
| placement | which training step a document was used in |
| membership | membership inference: was this document in the training data |
| fingerprint card / init regenerated | the token vectors' random starting values, regenerated exactly |
| padding rows | unused vocabulary slots (padding / reserved tokens) |

Additions after the editorial review

| Avoid | Use |
|---|---|
| "never updated" as a bare observation | "bit-identical to the released base" (observation). Say "never updated" only after the detection-floor check; in 16-bit releases a tiny update can round to zero |
| near | "barely changed (under 1% of its numbers)" |
| special tokens (lumped) | named special tokens (chat, tool, thinking, image) vs reserved placeholder slots |
| total weight decay applied (bare) | add a gloss, e.g. "the vectors shrank by 1.3% in total" |
| recover (for token sets from 16-bit releases) | detect; "recover" only for fp32 / bit-exact results |
| shows (about data content) | "consistent with" |

Never make claims about a lab's intent.

Rules
- Counts of rows are counts of distinct vocabulary tokens, never amounts of text. Say so where it matters.
- Match the verb to the certainty: never "read" or "found" for something that is an assumption-based estimate.
- Keep column headers short but plain; table cells may use "params" etc. if a legend defines them.
- Code identifiers (function names, JSON keys) stay as they are; only human-facing prose changes.
