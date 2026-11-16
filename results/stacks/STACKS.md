# Training stacks: does the record survive real fine-tuning setups?

**Partial.** Several variants were not run; only the rows marked *scored* carry
results. The DeepSpeed ZeRO-2 run (t08) was trained with DeepSpeed itself, not replayed with its formulas. Everything ran on one RTX PRO 6000 (96 GB): no multi-GPU box was
available, so several ranks would have shared one GPU over gloo, and the 7B run would have kept its
optimizer state on NVMe.

Method: each run fine-tunes with full ground truth (`scripts/stacks/finetune_stack.py`); the unchanged
blind reader (`scripts/controlled/score.py`, wrapped by `scripts/stacks/score_stack.py`) recovers which
vocabulary tokens were trained, the learning-rate schedule, the shuffle seed, and replays the token
vectors that were never updated. Counts below are counts of distinct vocabulary tokens.

The wrapper adds two stack-aware steps: the starting point may be the base model cast to 16 bits
(as DeepSpeed does), and weight decay is replayed with three kernels (torch `p*(1-lr*wd)`, DeepSpeed
FusedAdam `fma(-lr, wd*p, p)`, DeepSpeed CPUAdam `fma(p, -lr*wd, p)`), keeping the one that is bit-exact.

| run | setup | status | tokens trained (recovered / true, false positives) | schedule recovered exactly | seed | never-updated vectors reproduced bit-exactly |
|---|---|---|---|---|---|---|
| t01_clip_off | Pythia-160m, no gradient clipping (control) | scored | 18,461 / 18,538; via replay 18,538 / 18,538 | yes | recovered | 24,396,288 of 24,455,424 weights (mul kernel) |
| t02_clip_on | Pythia-160m, gradient clipping at 1.0 (clips every step: norms 20-30) | scored | 18,468 / 18,538; via replay 18,538 / 18,538 | yes | recovered | 24,396,289 of 24,450,048 weights (mul kernel) |
| t03_clip_tight | Pythia-160m, clipping at 0.01 (tiny gradients, Adam eps matters) | scored | 18,231 / 18,538; via replay 18,538 / 18,538, 31766 fp | no (cosine, 1142 steps, warm-up 17, peak 7.15e-05) | missed | 505,961 of 24,632,064 weights (mul kernel) |
| t04_ga4 | Pythia-160m, gradient accumulation 4 x 2 documents | scored | 18,241 / 18,538; via replay 18,538 / 18,538 | yes | recovered | 24,396,529 of 24,624,384 weights (mul kernel) |
| t05_ddp2 | Pythia-160m, 2 data-parallel ranks (gloo, one GPU) | stopped mid-run | - | - | - | - |
| t06_ddp4_ga2 | Pythia-160m, 4 ranks x accumulation 2 | not run | - | - | - | - |
| t16_ddp2_ga4 | Pythia-160m, 2 data-parallel ranks x accumulation 4 (gloo, one GPU) | scored | 18,449 / 18,538; via replay 18,538 / 18,538 | yes | recovered | 24,396,288 of 24,464,640 weights (mul kernel) |
| t07_fsdp_410m | Pythia-410m, FSDP full_shard, bf16 mixed precision | not run | - | - | - | - |
| t08_ds_z2_dsopt | DeepSpeed ZeRO-2, bf16, DeepSpeed FusedAdam | scored | 18,490 / 18,538; via replay 18,538 / 18,538 | yes | recovered | 24,396,313 of 24,433,152 weights (fused kernel) |
| t09_ds_z2_torchopt | DeepSpeed ZeRO-2, bf16, torch AdamW | not run | - | - | - | - |
| t10_ds_z3_offload | DeepSpeed ZeRO-3, optimizer on CPU (CPUAdam) | not run | - | - | - | - |
| t11_muon_hidden | Muon on hidden matrices, AdamW on embeddings | not run | - | - | - | - |
| t12_muon_all | Muon on every matrix including the embedding | not run | - | - | - | - |
| t13_pack | packed 256-token sequences | scored | 18,521 / 18,589; via replay 18,589 / 18,589 | yes | recovered | 24,357,120 of 24,409,344 weights (mul kernel) |
| t14_purebf16_hi | 16-bit weights and optimizer, no 32-bit master copy, lr 2e-4 | scored (tokens only) | 18,538 / 18,538 (bit-identity test) | not readable | not readable | weight decay rounds away: all 31,766 never-trained vectors bit-identical |
| t15_purebf16_lo | same, lr 2e-5 | scored (tokens only) | 18,526 / 18,538 (bit-identity test) | not readable | not readable | weight decay rounds away: all 31,766 never-trained vectors bit-identical |
| q7_ds_z3_nvme | Qwen2.5-7B full fine-tune, ZeRO-3, optimizer state on NVMe | stopped mid-run | - | - | - | - |

Gradient norms logged during training (clipping threshold in brackets): t02_clip_on: 12-33.3 [1.0]; t03_clip_tight: 14.1-61.4 [0.01]; t04_ga4: 13.8-40.4 [1.0]; t16_ddp2_ga4: 12.8-25.5 [1.0]; t08_ds_z2_dsopt: 82.2-1.79e+03 [1.0]; t13_pack: 8.73-13.7 [1.0]

Note: the 24 earlier controlled runs (`results/controlled/`) already used the Hugging Face default
clipping at 1.0 with gradient norms of 13-19, so clipping was active on every step there too.

t03 (clipping at 0.01) fails, as expected: clipped gradients are so small that Adam's epsilon dominates
the update, so a token seen once no longer moves by about 31.8 x the learning rate and the schedule cannot
be estimated. Real fine-tunes clip at about 1.0, as t02 does.

t08 is trained with DeepSpeed ZeRO-2 and its FusedAdam: the base is cast to 16 bits before the 32-bit
master copy is built, and weight decay rounds as fma(-lr, wd*p, p). With that starting point and kernel
the never-updated vectors replay bit-exactly; with the torch formula they do not. Its warm-up was first
estimated one step long (21 vs 20); trying warm-up +-3 and steps +-1 against the bit-exact replay
(`refine_by_replay` in score_stack.py) fixed it, and the seed and document steps then followed.

## Packed sequences (t13)

Documents joined with an end-of-text token into 256-token chunks. The reader recovers the schedule, the
seed (as the chunk order) and the bit-exact replay unchanged. The last-token rule changes: without packing a
document's final token is the last input of its sequence and gets no gradient, so the published-data audits
drop final positions. Packed, it is followed by the end-of-text token and is trained: 68 of the
68 tokens that occur in the training documents only in final position were updated, and the
end-of-text token itself was updated. Only each chunk's last token gets no gradient.
For an audit: if tokens that occur only at the ends of published documents were updated, the run was packed
(or appended an end token after each document); if they are all bit-identical, it was not. Either way the
token-set comparison stays exact once the matching rule is applied (`scripts/stacks/pack_lasttoken.py`).
The reader's direct reading misses exactly these 68 tokens; its replay reading, which does not assume the
rule, finds all of them.

## Pure 16-bit training (t14, t15)

16-bit weights and optimizer state with no 32-bit master copy. Each update is rounded to whole 16-bit steps,
so a token seen once no longer moves by a measurable 31.8 x the learning rate: the schedule fit finds nothing
to fit, and the recipe, seed and document steps are not recoverable. Weight decay (lr x wd at most 2e-05, 2e-06 per step) is below half a 16-bit step
(about 0.2%), so it rounds away entirely and every never-trained vector stays bit-identical. The token list is
therefore read by a plain bit-identity test, with no replay (`scripts/stacks/purebf16_tokens.py`):

| run | peak lr | tokens trained (found / true) | false positives | missed | share of numbers changed per updated vector (median) | relative change (median) |
|---|---|---|---|---|---|---|
| t14_purebf16_hi | 0.0002 | 18,538 / 18,538 | 0 | 0 | 94% | 0.044 |
| t15_purebf16_lo | 2e-05 | 18,526 / 18,538 | 0 | 12 | 13% | 0.0016 |

At the low learning rate, trained tokens whose updates never reached one 16-bit step stay bit-identical:
that is the detection floor.
