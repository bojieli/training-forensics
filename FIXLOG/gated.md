# Gated lane summary (for the integration pass)

Source: results/gated/GATED.md, results/roster/sweep_gated.jsonl. Raw counts are authoritative; `class` is from scripts/frontier/comparability.py as of this commit and should be recomputed if that classifier changes.

| base | release | bit-identical / barely changed / updated token vectors | shrink factor P | fit | one matrix for input and output | share of vectors by fraction of numbers changed (0 / <=1% / 1-10% / 10-50% / >50%) | class |
|---|---|---|---|---|---|---|---|
| meta-llama/Llama-3.1-8B-Instruct | microsoft/UserLM-2.0 | 3,509 / 49 / 124,698 | 1 | 1 | no | 0.027 / 0.000 / 0.002 / 0.008 / 0.962 | token record |
| meta-llama/Llama-3.1-8B | microsoft/UserLM-2.0 | 0 / 0 / 128,256 | 1 | 0.03735 | no | 0.000 / 0.000 / 0.000 / 0.000 / 1.000 | not comparable |
| CohereLabs/tiny-aya-base | CohereLabs/tiny-aya-global | 0 / 0 / 262,144 | 0.850002 | 0.01881 | yes | 0.000 / 0.000 / 0.000 / 0.000 / 1.000 | shared matrix |
| CohereLabs/tiny-aya-base | CohereLabs/tiny-aya-earth | 0 / 0 / 262,144 | 0.860858 | 0.02063 | yes | 0.000 / 0.000 / 0.000 / 0.000 / 1.000 | shared matrix |
| CohereLabs/tiny-aya-base | CohereLabs/tiny-aya-fire | 0 / 0 / 262,144 | 0.882532 | 0.02483 | yes | 0.000 / 0.000 / 0.000 / 0.000 / 1.000 | shared matrix |
| CohereLabs/tiny-aya-base | CohereLabs/tiny-aya-water | 0 / 0 / 262,144 | 0.860698 | 0.02061 | yes | 0.000 / 0.000 / 0.000 / 0.000 / 1.000 | shared matrix |
| zai-org/GLM-5.3-BF16 | deep-principle/Hydrogen-1.0 | 103,828 / 0 / 51,052 | 1 | 1 | no | 0.670 / 0.000 / 0.000 / 0.005 / 0.325 | token record |
| swiss-ai/Apertus-v1.5-70B-base | swiss-ai/Apertus-v1.5-70B-SFT | 31,837 / 362 / 234,553 | 1 | 1 | no | 0.119 / 0.001 / 0.001 / 0.003 / 0.875 | token record |
| swiss-ai/Apertus-v1.5-70B-SFT | swiss-ai/Apertus-v1.5-70B-RL | 51,857 / 5,931 / 208,964 | 1 | 1 | no | 0.194 / 0.022 / 0.161 / 0.621 / 0.002 | token record |
| swiss-ai/Apertus-v1.5-70B-base | swiss-ai/Apertus-v1.5-70B | 31,492 / 212 / 235,048 | 1 | 1 | no | 0.118 / 0.001 / 0.001 / 0.005 / 0.875 | token record |
| swiss-ai/Apertus-v1.5-70B-RL | swiss-ai/Apertus-v1.5-70B | 44,444 / 5,178 / 217,130 | 1 | 1 | no | 0.167 / 0.019 / 0.110 / 0.701 / 0.003 | token record |

Lane reports updated: results/frontier_x/FRONTIER_X.md (Apertus), results/y2026_labs/LABS2026.md (tiny-aya, UserLM-2.0), results/y2026_ft/FT2026.md (Hydrogen-1.0), results/large/LARGE.md (405B copies verified).
README not edited (per coordinator). Suggested README line: "Gated models: open copies verified byte-identical to official releases by sha256; previously gated comparisons in results/gated/GATED.md."
