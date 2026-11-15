# training-forensics

Reading a model's training history from its embedding rows: which tokens were in the data, the
learning-rate schedule, the number of steps, and (given the dataset) which optimizer step each
document was in. This repository is an independent replication of a public thread making these
claims, tested three ways: on the four public models it names, on 24 controlled fine-tunes with
full ground truth (Pythia-160m/410m and Qwen2.5-7B), and on parent-to-child pairs from the
265-model roster of the sibling project [checkpoint-forensics](https://github.com/bojieli/checkpoint-forensics).

## The mechanism, in five lines

1. Fine-tune with Adam(W) and subtract the starting embedding matrix. Each row is one token.
2. A token the data never contained gets no gradient: its row does not move (with decoupled weight
   decay it only shrinks by a factor that is the same for every such row).
3. A token seen in exactly one optimizer step moves by the same amount D in nearly every coordinate:
   Adam divides a first gradient by its own magnitude, so only the sign survives.
4. D depends only on the learning rates from that step on: D(s) = sum_k lr(s+k) a(k, s). With
   default betas and full bias correction D/lr -> 31.77. So D says *when*; the sign pattern says
   *what* (minus the sign of that step's gradient).
5. Under decoupled weight decay every coordinate goes through the same sequence of fp32 multiplies,
   so the never-touched rows can be replayed bit for bit once the schedule is known.

`trainforensics/adam.py` is the forward model; `trainforensics/blind.py` does the reading;
`scripts/controlled/score.py` reads a run blind and only then opens its ground truth.

## Results

### The four public claims (`results/public/PUBLIC_CLAIMS.md`)

| Claim | Verdict | What the weights say |
|---|---|---|
| Pythia-1B peak LR is 2.5e-4, not the card's 3.0e-4 | **confirmed** | 2.50002e-4 from the padding rows' decay over 154 checkpoints; the yml config agrees. All 8 sizes land within 0.4% of their configs. |
| Pythia's LR is flat for its last ~28,000 steps (cosine to 0 clipped at 0.1x peak) | **confirmed** | clip at 0.1000x peak at every size; flat tail 27,657-27,865 steps (predicted 27,861). Today's gpt-neox scheduler fits 50-200x worse. |
| OLMo-2 first SFT release: ~20,341 rows that should have trained never moved, incl. the chat-turn closer | **confirmed, count differs** | 23,064 real-token rows bit-identical to base; 20,878 of them occur in the SFT data (20,358-20,878 under three definitions). `>Ċ` (id 397, 2.39M occurrences) never moved; it moved in the retrain. |
| Qwen2.5 instruct touched exactly 4 of 22 control tokens, same 4 at 7B/14B/32B/72B | **confirmed** | `<\|im_start\|>`, `<\|im_end\|>`, `<tool_call>`, `</tool_call>` at all four sizes; the other 18 bit-identical. |

Side findings: pythia-12b branches `step50000`/`step58000` contain the final model's padding rows;
pythia-2.8b's padding rows are bit-identical to step 0 in 85 of 153 later branches (incl. the final one).

### Controlled fine-tunes (`results/controlled/CONTROLLED.md`)

24 fine-tunes with full ground truth: Pythia-160m/410m (full fine-tuning) and Qwen2.5-7B (embeddings +
head, LoRA with trained embeddings, weight decay, batch 128, two epochs), linear / cosine / constant /
cosine-to-floor / warmup-stable-decay schedules, batch 4-128, weight decay 0-0.1. Each run is read blind
from (start, release) and only then scored against the trainer's logged learning rates and batches.

| What is read | Result |
|---|---|
| Tokens seen (moved rows = tokens in the data) | **exact in 24 / 24** (no false positive); with weight decay via bit-exact replay of the decay |
| Recipe: schedule, steps, warm-up, peak (and wd) | **exact in 22 / 24**; peak within 0.001-0.016% before snapping to a round value |
| Shuffle seed (HF Trainer) | **found in 22 / 22** runs with a recipe |
| Documents in their exact step, from the weights alone | 19-43% (≤ 4 wrong in most runs); **100%** once the seed is known |
| Replay of never-touched weights | bit-exact: 473,769,020 weights (Qwen2.5-7B), ±1 step leaves only ~1,100 (≈ the exact zeros) |
| 40 secretly swapped documents | 24 caught, 3 false flags |
| Membership at zero false alarms | 41.4% (best standard attack 0.4%) |

Misses: WSD at bs 8 read one step long (a 380-step flat stage makes the length weakly identifiable
without weight decay; the bs 4 WSD run is exact), and the two-epoch Qwen run (most rows touched twice).
Full table: [`results/controlled/CONTROLLED.md`](results/controlled/CONTROLLED.md).

What made the readings work, beyond the thread's description: per-row moves read as the *intercept* of
the sorted-moves curve (Adam's eps shrinks small-gradient coordinates), a mixture likelihood over
steps with a detection floor, decay strength tied to the total decay P for every candidate schedule,
an exact decay map read off the never-touched rows (fp32 rounding drift is otherwise as large as the
moves), and wd and step count settled by bit-exact replay.

### The 265-model roster (`results/roster/SWEEP.md`)

Every public parent -> child pair in the checkpoint-forensics roster whose embeddings can be compared:
**229 pairs** (197 untied, 32 tied), embeddings fetched by HTTP range requests and diffed row by row.

- **129 pairs** keep exactly unmoved rows next to moved ones: the set of tokens the fine-tune's data
  contained is readable bit for bit. 5 more pairs did not change the embedding at all (re-uploads,
  conversions, frozen embeddings); 56 moved every row.
- **Tied embeddings erase the record** (28 of 32 tied pairs keep fewer than 10 unmoved rows; 2 are unchanged), and so do untied output heads
  (105 of 107 move ≥ 99% of rows): the output softmax sends a gradient to every row.
- **Control tokens.** Qwen2.5-Instruct touches the same 4 of 22 at 7B/14B/32B/72B (the thread's claim);
  Qwen2.5-Coder touches the FIM and repo/file tokens, Qwen2.5-Math only `<|endoftext|>`; the full table
  covers every family.
- **Weight decay** is visible in 26 pairs; 23 give an exact interval for the integrated sum(lr x wd),
  e.g. Mistral-Nemo-Instruct [0.01993, 0.01997], Qwen3-8B/14B/30B-A3B all [0.01311, 0.01318].
  Correction to our own first pass: five releases (Llama-3.1/3.3 Instruct, Nemotron-3-Ultra, LLaDA-MoE-v2)
  seemed to share P = 0.994148; their rows moved exactly one bf16 step, which only bounds sum(lr x wd) to
  [0.0039, 0.0059], and the shared value is that interval's edge, not a shared recipe.
- **Single-step timestamps do not survive public 16-bit releases** at today's batch sizes: only 5 pairs
  have ≥ 20 single-touch rows, and their moves are 1-3 bf16 steps. Learning rates are not read from them.

## Where the thread holds and where it does not (here)

Each line is a claim from the thread, our verdict, and the evidence. "Exact" always means checked
against ground truth that the reader never saw.

| Thread claim | Here | Evidence |
|---|---|---|
| Moved rows = tokens in the data (39,663 = 39,663) | **holds** | Exact in every run (18.5k-45k tokens, Pythia and Qwen2.5-7B, LoRA, bs up to 128); with weight decay, exact via replay. |
| Recipe from the weights alone: schedule, steps, warm-up right in 15/15; peak within 0.01% | **holds** (22 / 24) | Peak typically within 0.001-0.016% before snapping to a round value. Misses listed below. |
| 3,121 / 4,000 docs placed in their exact step | **partly** | From D alone: 770-1,700 of 4,000 (19-43%), almost all correct. With the recovered seed: 4,000 / 4,000. |
| Shuffle seed found in every run | **holds** | Found in every run whose recipe was read (HF Trainer = `torch.randperm` seeded with seed + epoch). |
| 38 / 40 secretly swapped docs exposed, no false flag | **partly** | 24 / 40 caught; the other 16 contain no token unique to them among the claimed docs. 0-5 false flags per clean run (more under weight decay and large batch). |
| Replay: 25.5M weights matched bit for bit; one step off, only exact zeros match | **holds** | Qwen2.5-7B, wd 0.1: 473,769,020 weights bit-exact; ±1 step: ~1,100 (1,095 exact zeros). Pythia: 24.4M, ±1 step: 26 (the zeros). |
| Membership: 64% at zero false alarms; strongest standard test 0.5% | **partly** | 41.4% at zero false alarms (71% of the docs that contain a single-touch token); standard attacks 0.025-0.4%. Only 58% of members are eligible here, so 64% is above this method's ceiling on this data. |
| Two epochs: both timestamps within 1% for 93-97% of twice-touched rows | **partly** | 61-69% when the touches are ~5,000 steps apart; 0.7% when 500 apart (Adam's second moment still remembers the first touch: beta2^500 = 0.61). |
| Large batch: sorted moves follow a known curve, intercept = timestamp | **holds** | Intercept fit (half-normal gradients, eps-limited Adam) reads 93.6% of single-touch rows within 0.3% even when only 3% of them are flat; recipe exact at bs 128. |
| 16-bit releases: token list stays exact | **holds, with one exception** | bf16 and fp16 exact without weight decay; with decay to lr ~0 at the end, 4-89 last-step tokens round away (their moves are below half a 16-bit step). |
| Adam + L2: replaying Adam on the decay term alone recovers the token list | **holds** | 31,766 / 31,766 never-seen rows (24.4M weights) bit-exact, 0 seen rows. |
| LoRA with trained embeddings: whole record there | **holds** | Recipe exact, seed found, tokens exact (Pythia and Qwen2.5-7B). |


## Reproduce

```bash
pip install -e '.[controlled]'
make test        # offline unit tests of the forward model (CPU)
make grid        # controlled fine-tunes with ground truth (one GPU)
make score table # blind reading, scored
make public      # the four public claims (CPU + network; set HF_TOKEN)
make roster      # embedding diffs over the roster pairs
```

## Limitations

- **Weight decay needs the exact decay map.** Every coordinate is multiplied by an fp32-rounded
  (1 - lr*wd) each step; the rounding drift (up to ~N/2 ulp) is comparable to the moves at low
  learning rates. We read the map off the never-touched rows as a lookup table (exact when the start
  was stored in 16 bits, as released models are) and pick wd by bit-exact replay.
- **Multi-epoch recipe reading** is not solved here: with most rows touched twice, the one-touch
  curve does not apply (q7_2ep).
- **Tied embeddings** (Qwen2.5-0.5B/1.5B/3B, Gemma, many small models): every row gets gradient through
  the output softmax, so no row is untouched. **LoRA on attention only** leaves embeddings untouched.
- **Full fine-tuning of a 7B on one GPU** did not fit (AdamW states); 7B runs train embeddings + head,
  LoRA, or embeddings only. The mechanism only involves the embedding rows' gradients.
- **Controlled data** is Alpaca (short instructions); placement, swap and membership rates depend on how
  many tokens are unique to a document.
- Public models give tokens and, with weight decay or checkpoints, schedules; document-level readings
  need the training data.


## License

MIT. Independent work; not affiliated with any of the labs whose models are examined. The readings
describe what the published weights record; they say nothing about intent.
