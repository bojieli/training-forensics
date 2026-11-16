# Article figures

One consistent set, drawn by `make_figures.py` and `make_fig0_cover.py` (run from the repository root:
`python3 results/figures/make_figures.py`; CPU only, matplotlib, Lato font).
Each figure has a PNG (1600 px wide), an SVG, and a sidecar JSON that lists the source file and
every value drawn. Wording follows `STYLE.md`.

Colours are Okabe-Ito and mean the same thing in figures 1 to 4: orange = bit-identical to the base,
yellow = barely changed, light blue = partly changed, blue = updated. Bit-identical cells also carry
an "=" sign in figure 2, so the reading does not depend on colour. Figures 5 and 6 reuse the palette
for their own categories, named in each figure.

| Figure | Files | Source | Caption |
|---|---|---|---|
| 0 (cover) | `fig0_cover_matches.{png,svg,json}` (1600×640), drawn by `make_fig0_cover.py` | `results/published/PUBLISHED.md` main table; `results/large/LARGE.md` (Tulu 3 405B, SFT) | Published training data against the released weights, in distinct vocabulary tokens. For s1, s1.1, Sky-T1, Bespoke-Stratos and OpenThinker the tokens in the published data equal the tokens whose vectors changed, at one setting of cutoff and system prompt (documented by the project except for Bespoke-Stratos, where it is documented by reference). For Tulu 3 405B's SFT stage all but 245 of 124,572 changed and none outside the data. |
| 1 | `fig1_mechanism.{png,svg,json}` (1600×900) | `README.md` "The mechanism, in five lines" (the 31.8 is D/lr = 31.77). Schematic: bar heights are illustrative, not data. | How training leaves a record. A token that never appears in the training data keeps a vector that is bit-identical to the base model. A token seen in one training step has every number shifted by the same small amount, up or down: about 31.8 times the learning rate at that step. With weight decay on, vectors that were never updated all shrink by one exact factor that can be replayed. From this one can detect which tokens the data contained and which special tokens were trained, recover the schedule (32-bit releases only), and see which base model a release started from. |
| 2 | `fig2_deepseek_special_tokens.{png,svg,json}` (1600×900) | `results/frontier_ds/fig_b_deepseek_special_tokens.json` (the four V3-Base comparisons) | DeepSeek's 18 special tokens in four releases, each compared with DeepSeek-V3-Base. Orange with "=": the token vector is bit-identical to the base. Blue: updated. V3 left the four tool-output tokens bit-identical, V3-0324 updated them, and R1 and R1-Zero left all nine tool tokens bit-identical. |
| 3 | `fig3_special_tokens_other_labs.{png,svg,json}` (1600×1218) | `results/frontier/special_grid_nondeepseek.json` (13 of its 22 comparisons; the omitted ones are listed in the sidecar) | Special tokens by kind for 13 releases from other labs, each compared with its own base model. Each number is a count of special tokens in that state. Qwen3-8B and 14B left all 11 image and video tokens bit-identical while updating every chat, tool, thinking and code token. Qwen3-30B-A3B cannot be tested: its special-token vectors are unrelated to the published base. |
| 4 | `fig4_change_share.{png,svg,json}` (1600×900) | `results/frontier/fig/a_change_histogram.json`; state definitions from `results/frontier/FRONTIER.md` | How much of each token vector changed, for three DeepSeek releases compared with DeepSeek-V3-Base. Each bar is all 129,280 vocabulary tokens. In R1-Zero, 8.0% of token vectors are bit-identical and another 61.5% have under 10% of their numbers changed. In R1 and V3-0324 most token vectors are fully updated. |
| 5 | `fig5_comparability.{png,svg,json}` (1600×1000) | `results/frontier/fig/d_comparability.json` (roster lane of `results/frontier/comparability.json`) | The 229 base-and-release comparisons in seven groups, in total (left) and by model family (right). 97 give a token record: trainable token vectors bit-identical to the base, which shows shared ancestry, not a direct parent. 33 share only untrainable tokens. Releases whose input and output share one matrix (tied embeddings, such as Gemma) carry no record. |
| 6 | `fig6_weight_decay_ranges.{png,svg,json}` (1600×1316) | `results/frontier/fig/c_decay_ranges.json`; "one smallest 16-bit step" from `results/roster/NOTES.md` | How much the never-updated token vectors shrank between base and release, for 20 releases (logarithmic scale). Each bar is the range of shrink that reproduces the same vectors, widened for fp32 rounding drift. Qwen3-8B, 14B and 30B-A3B share one range, 1.25–1.37%. For the releases in orange the vectors shrank by exactly one smallest 16-bit step, so only a wide bound is known (0.33–0.64% for five of them); they do not share a recipe. |

Not drawn: the DeepSeek V4 named tokens (67 tokens × 9 comparisons in the same source file). The article
reports those by name and status in the text only.

The earlier drafts in `results/frontier/fig/` and `results/frontier_ds/` are untouched; these replace them
in the article.
