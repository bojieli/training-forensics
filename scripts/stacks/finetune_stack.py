"""Controlled fine-tune on realistic training stacks, with full ground truth.

Same contract as scripts/controlled/finetune.py (writes emb.safetensors,
truth.json, tokens.json), plus the stack features vendors actually use:

  --ga N                 gradient accumulation (micro-batch --bs, update batch bs*ga*world)
  --max-grad-norm X      global-norm clipping (0 disables; HF default 1.0)
  --deepspeed CFG        DeepSpeed ZeRO-1/2/3 (+ CPU offload) via the HF Trainer
  --fsdp "full_shard"    torch FSDP via accelerate
  --muon hidden|all      Muon on 2-D hidden matrices (AdamW on embeddings/head/1-D), or on
                         every 2-D matrix including the embedding
  --pack                 concatenate the claimed documents (EOS-separated) and train on
                         fixed-length chunks
  --pure-bf16            bf16 weights and optimizer state, no fp32 master copy
  --cpu                  run on CPU (gloo) to emulate several ranks on one box

Run with torchrun for several ranks, e.g.
  torchrun --nproc_per_node 2 scripts/stacks/finetune_stack.py --cpu --out runs/stacks/t04 ...
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random

import torch
import torch.distributed as dist
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments)

p = argparse.ArgumentParser()
p.add_argument("--model", default="EleutherAI/pythia-160m")
p.add_argument("--out", required=True)
p.add_argument("--data-jsonl", default="runs/alpaca.jsonl")
p.add_argument("--pool-seed", type=int, default=0)
p.add_argument("--n-docs", type=int, default=4000)
p.add_argument("--max-len", type=int, default=256)
p.add_argument("--epochs", type=float, default=1)
p.add_argument("--sched", default="linear")
p.add_argument("--warmup", type=int, default=0)
p.add_argument("--min-lr-ratio", type=float, default=0.1)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--bs", type=int, default=8, help="per-device micro-batch")
p.add_argument("--ga", type=int, default=1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--max-grad-norm", type=float, default=1.0)
p.add_argument("--deepspeed", default=None)
p.add_argument("--fsdp", default="")
p.add_argument("--fsdp-config", default=None)
p.add_argument("--muon", default="none", choices=["none", "hidden", "all"])
p.add_argument("--pack", action="store_true")
p.add_argument("--pure-bf16", action="store_true")
p.add_argument("--bf16", action="store_true", help="bf16 mixed precision (fp32 master)")
p.add_argument("--cpu", action="store_true")
p.add_argument("--same-gpu", action="store_true", help="all ranks on cuda:0 over gloo (NCCL refuses two ranks per GPU)")
p.add_argument("--grad-ckpt", action="store_true")
p.add_argument("--load-bf16", action="store_true", help="load the (bf16-published) base in bf16; mixed precision unchanged")
p.add_argument("--optim", default="adamw_torch")
args = p.parse_args()
if args.same_gpu:
    os.environ["LOCAL_RANK"] = "0"           # every rank uses cuda:0

rank = int(os.environ.get("RANK", 0))
world = int(os.environ.get("WORLD_SIZE", 1))
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ------------------------------------------------------------------ documents
rng = random.Random(args.pool_seed)
ds = load_dataset("json", data_files=args.data_jsonl, split="train")


def fmt(r):
    s = f"### Instruction:\n{r['instruction']}\n"
    if r["input"]:
        s += f"### Input:\n{r['input']}\n"
    return s + f"### Response:\n{r['output']}"


pool_idx = list(range(len(ds)))
rng.shuffle(pool_idx)
claimed = pool_idx[: args.n_docs]
outside = pool_idx[args.n_docs: 2 * args.n_docs]
enc = {i: tok(fmt(ds[i]), truncation=True, max_length=args.max_len)["input_ids"] for i in set(claimed) | set(outside)}
pack_note = None
if args.pack:
    # deterministic packing of the claimed list (the reader rebuilds the same chunks)
    eos = tok.eos_token_id
    stream = []
    for i in claimed:
        stream += enc[i] + [eos]
    L = args.max_len
    chunks = [stream[a:a + L] for a in range(0, len(stream) - L + 1, L)]
    base = 10_000_000
    for j, c in enumerate(chunks):
        enc[base + j] = c
    trained = [base + j for j in range(len(chunks))]
    claimed_units = list(trained)
    pack_note = f"{len(claimed)} docs -> {len(chunks)} chunks of {L} tokens (EOS-joined, claimed order, tail dropped)"
else:
    trained = list(claimed)
    claimed_units = list(claimed)


class Docs(torch.utils.data.Dataset):
    def __len__(self):
        return len(trained)

    def __getitem__(self, k):
        return {"input_ids": enc[trained[k]], "pos": k}


seen_batches: list[list[int]] = []


def collate(feats):
    seen_batches.append([f["pos"] for f in feats])
    L = max(len(f["input_ids"]) for f in feats)
    ids = torch.full((len(feats), L), tok.pad_token_id)
    att = torch.zeros((len(feats), L), dtype=torch.long)
    for r, f in enumerate(feats):
        ids[r, : len(f["input_ids"])] = torch.tensor(f["input_ids"])
        att[r, : len(f["input_ids"])] = 1
    labels = ids.masked_fill(att == 0, -100)
    return {"input_ids": ids, "attention_mask": att, "labels": labels}


# ------------------------------------------------------------------ model
dtype = torch.bfloat16 if (args.pure_bf16 or args.load_bf16) else torch.float32
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
emb_name = [n for n, m in model.named_modules() if m is model.get_input_embeddings()][0]
W0 = model.get_input_embeddings().weight.detach().clone().float().cpu()
assert model.get_output_embeddings().weight.data_ptr() != model.get_input_embeddings().weight.data_ptr(), "tied"
if args.grad_ckpt:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

lrs: list[float] = []


class LrLog(TrainerCallback):
    def on_step_begin(self, a, state, control, lr_scheduler=None, **kw):
        lrs.append(float(lr_scheduler.get_last_lr()[0]))


sched_kwargs = {}
sched = args.sched
if sched == "cosine_min":
    sched, sched_kwargs = "cosine_with_min_lr", {"min_lr_rate": args.min_lr_ratio}

optimizers = (None, None)
emb_param = model.get_input_embeddings().weight
head_param = model.get_output_embeddings().weight


class MuonAdamW(torch.optim.Optimizer):
    """Muon on the chosen 2-D matrices, AdamW (torch, foreach) on everything else.
    Both inner optimizers share their param_group dicts with this wrapper, so one
    LR scheduler drives both (Moonlight-style: Muon lr matched to AdamW RMS)."""

    def __init__(self, muon_params, adam_params, lr, wd, betas, eps):
        self.adam = torch.optim.AdamW(adam_params, lr=lr, weight_decay=wd, betas=betas, eps=eps, foreach=True)
        self.muon = torch.optim.Muon(muon_params, lr=lr, weight_decay=wd, adjust_lr_fn="match_rms_adamw")
        self.defaults = dict(lr=lr)
        self.param_groups = self.adam.param_groups + self.muon.param_groups
        self.state = {}
        self._optimizer_step_pre_hooks = {}
        self._optimizer_step_post_hooks = {}
        self._optimizer_state_dict_pre_hooks = {}
        self._optimizer_state_dict_post_hooks = {}
        self._optimizer_load_state_dict_pre_hooks = {}
        self._optimizer_load_state_dict_post_hooks = {}
        self._zero_grad_profile_name = "MuonAdamW.zero_grad"

    @torch.no_grad()
    def step(self, closure=None):
        self.adam.step()
        self.muon.step()

    def zero_grad(self, set_to_none=True):
        self.adam.zero_grad(set_to_none)
        self.muon.zero_grad(set_to_none)


if args.muon != "none":
    mu, ad = [], []
    for n, q in model.named_parameters():
        is2d = q.ndim == 2
        if args.muon == "hidden":
            use = is2d and q is not emb_param and q is not head_param
        else:
            use = is2d
        (mu if use else ad).append(q)
    opt = MuonAdamW(mu, ad, args.lr, args.wd, (0.9, 0.999), 1e-8)
    optimizers = (opt, None)

ta = TrainingArguments(
    output_dir=str(out / "trainer"), per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.ga,
    num_train_epochs=args.epochs, learning_rate=args.lr, weight_decay=args.wd,
    lr_scheduler_type=sched, lr_scheduler_kwargs=sched_kwargs, warmup_steps=args.warmup,
    adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, optim=args.optim,
    seed=args.seed, logging_steps=25, save_strategy="no", report_to=[], remove_unused_columns=False,
    bf16=args.bf16, max_grad_norm=args.max_grad_norm, dataloader_num_workers=0, disable_tqdm=True,
    deepspeed=args.deepspeed, fsdp=args.fsdp, fsdp_config=args.fsdp_config, use_cpu=args.cpu,
    ddp_backend="gloo" if (args.cpu or args.same_gpu) else None, gradient_checkpointing=False,
)
trainer = Trainer(model=model, args=ta, train_dataset=Docs(), data_collator=collate, callbacks=[LrLog()],
                  optimizers=optimizers)
trainer.train()

# ------------------------------------------------------------------ final embedding
w = trainer.model.get_input_embeddings().weight
W1_master = None
W1_param = None
if args.deepspeed:
    import deepspeed
    from deepspeed.utils import safe_get_full_fp32_param
    W1_master = safe_get_full_fp32_param(w)
    W1_master = None if W1_master is None else W1_master.detach().float().cpu().contiguous()
    with deepspeed.zero.GatheredParameters([w], modifier_rank=None):
        W1_param = w.detach().clone().cpu().contiguous()
elif args.fsdp:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    with FSDP.summon_full_params(trainer.model_wrapped, writeback=False):
        ww = trainer.model_wrapped.module.get_input_embeddings().weight if hasattr(trainer.model_wrapped, "module") \
            else trainer.model_wrapped.get_input_embeddings().weight
        W1_master = ww.detach().clone().float().cpu().contiguous()
else:
    W1_param = w.detach().clone().cpu().contiguous()
    if W1_param.dtype == torch.float32:
        W1_master = W1_param

# ------------------------------------------------------------------ truth (gather per-rank batches)
steps = len(lrs)
ga = args.ga
if world > 1 and dist.is_initialized():
    allb = [None] * world
    dist.all_gather_object(allb, seen_batches)
else:
    allb = [seen_batches]
if rank == 0:
    docs_per_step = []
    for k in range(steps):
        s = []
        for r in range(len(allb)):
            s += sum(allb[r][k * ga:(k + 1) * ga], [])
        docs_per_step.append(s)
    W1 = W1_master if W1_master is not None else W1_param.float()
    save = {"W0": W0.contiguous(), "W1": W1.contiguous(), "W1_bf16": W1.bfloat16().contiguous(),
            "W1_fp16": W1.half().contiguous()}
    if W1_param is not None and W1_param.dtype != torch.float32:
        save["W1_param"] = W1_param.contiguous()          # what a 16-bit release of this run contains
    save_file(save, str(out / "emb.safetensors"))
    o = trainer.optimizer
    while hasattr(o, "optimizer"):
        o = o.optimizer
    stack = dict(world=world, same_gpu=args.same_gpu, ga=ga, max_grad_norm=args.max_grad_norm, deepspeed=args.deepspeed and json.load(open(args.deepspeed)),
                 fsdp=args.fsdp, muon=args.muon, pack=pack_note, pure_bf16=args.pure_bf16, bf16=args.bf16,
                 master_saved=W1_master is not None, optimizer=type(o).__name__)
    a = vars(args).copy()
    a["bs"] = args.bs * world            # update batch = bs * ga * world; reader multiplies by ga
    gt = dict(args=a, steps=steps, lr=lrs, emb=emb_name, claimed=claimed_units, trained=trained,
              swapped_positions=[], outside=outside if not args.pack else [], docs_per_step=docs_per_step,
              optimizer=type(o).__name__, stack=stack,
              grad_norms=[h.get("grad_norm") for h in trainer.state.log_history if "grad_norm" in h])
    json.dump(gt, open(out / "truth.json", "w"))
    json.dump({str(i): enc[i] for i in enc}, open(out / "tokens.json", "w"))
    print("done", out, "steps", steps, stack)
