"""Controlled fine-tune with full ground truth.

Fine-tunes a causal LM with the Hugging Face Trainer on N documents and records,
for every optimizer step: the learning rate used, and the documents (hence
tokens) in it. Saves the starting and final input-embedding matrices in fp32,
plus bf16 / fp16 casts of the final one, so the forensics can be run blind and
scored against the truth.

Example:
  python3 scripts/controlled/finetune.py --out runs/r01 --sched linear --lr 5e-5 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments)

p = argparse.ArgumentParser()
p.add_argument("--model", default="EleutherAI/pythia-160m")
p.add_argument("--out", required=True)
p.add_argument("--source", default="alpaca", choices=["alpaca", "wiki"])
p.add_argument("--data-jsonl", default=None, help="local alpaca-format jsonl (offline boxes)")
p.add_argument("--pool-seed", type=int, default=0, help="which documents are drawn from the source")
p.add_argument("--n-docs", type=int, default=4000)
p.add_argument("--max-len", type=int, default=256)
p.add_argument("--epochs", type=float, default=1)
p.add_argument("--max-steps", type=int, default=-1)
p.add_argument("--sched", default="linear")
p.add_argument("--warmup", type=int, default=0)
p.add_argument("--min-lr-ratio", type=float, default=0.1)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--wd", type=float, default=0.0)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--ga", type=int, default=1)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--b1", type=float, default=0.9)
p.add_argument("--b2", type=float, default=0.999)
p.add_argument("--eps", type=float, default=1e-8)
p.add_argument("--optim", default="adamw_torch", help="adamw_torch | adamw_torch_fused | adam_l2")
p.add_argument("--lora", action="store_true", help="LoRA on attention + full embedding (modules_to_save)")
p.add_argument("--train-head", action="store_true", help="with --freeze-body: also train lm_head")
p.add_argument("--freeze-body", action="store_true", help="train only the input embedding (speed)")
p.add_argument("--swap", type=int, default=0, help="secretly replace this many claimed docs with outside docs")
p.add_argument("--bf16", action="store_true", help="bf16 autocast (fp32 master weights)")
p.add_argument("--save-model", action="store_true", help="also save the full fine-tuned model (membership baseline)")
p.add_argument("--loss-scale", default="mean", choices=["mean"], help="token-mean loss (HF default)")
args = p.parse_args()

out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ------------------------------------------------------------------ documents
rng = random.Random(args.pool_seed)
if args.source == "alpaca":
    ds = load_dataset("json", data_files=args.data_jsonl, split="train") if args.data_jsonl else \
        load_dataset("tatsu-lab/alpaca", split="train")
    def fmt(r):
        s = f"### Instruction:\n{r['instruction']}\n"
        if r["input"]:
            s += f"### Input:\n{r['input']}\n"
        return s + f"### Response:\n{r['output']}"
    pool_idx = list(range(len(ds)))
else:
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    def fmt(r):
        return r["text"].strip()
    pool_idx = [i for i in range(0, len(ds)) if len(ds[i]["text"]) > 400][:200000]
rng.shuffle(pool_idx)
claimed = pool_idx[: args.n_docs]
outside = pool_idx[args.n_docs: 2 * args.n_docs]          # never trained: for membership tests
swap_pos = sorted(rng.sample(range(args.n_docs), args.swap)) if args.swap else []
trained = list(claimed)
for j, pos in enumerate(swap_pos):
    trained[pos] = pool_idx[2 * args.n_docs + j]            # replacement doc, not in claimed nor outside

texts = {i: fmt(ds[i]) for i in set(claimed) | set(outside) | set(trained)}
enc = {i: tok(texts[i], truncation=True, max_length=args.max_len)["input_ids"] for i in texts}


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
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
emb_name = [n for n, m in model.named_modules() if m is model.get_input_embeddings()][0]
W0 = model.get_input_embeddings().weight.detach().clone().float().cpu()
assert model.get_output_embeddings().weight.data_ptr() != model.get_input_embeddings().weight.data_ptr(), "tied"
if args.freeze_body:
    for n, prm in model.named_parameters():
        prm.requires_grad = n.startswith(emb_name) or (args.train_head and "lm_head" in n)
if args.lora:
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["query_key_value"] if "pythia" in args.model else
                     ["q_proj", "k_proj", "v_proj", "o_proj"], modules_to_save=[emb_name.split(".")[-1]])
    model = get_peft_model(model, cfg)

lrs: list[float] = []


class LrLog(TrainerCallback):
    def on_step_begin(self, a, state, control, lr_scheduler=None, **kw):
        lrs.append(float(lr_scheduler.get_last_lr()[0]))


sched_kwargs = {}
sched = args.sched
if sched == "cosine_min":
    sched, sched_kwargs = "cosine_with_min_lr", {"min_lr_rate": args.min_lr_ratio}
elif sched == "wsd":
    sched, sched_kwargs = "warmup_stable_decay", {"num_decay_steps": None, "min_lr_ratio": args.min_lr_ratio}

ta = TrainingArguments(
    output_dir=str(out / "trainer"), per_device_train_batch_size=args.bs, gradient_accumulation_steps=args.ga,
    num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr, weight_decay=args.wd,
    lr_scheduler_type=sched, lr_scheduler_kwargs=sched_kwargs, warmup_steps=args.warmup,
    adam_beta1=args.b1, adam_beta2=args.b2, adam_epsilon=args.eps,
    optim=args.optim if args.optim != "adam_l2" else "adamw_torch",
    seed=args.seed, logging_steps=50, save_strategy="no", report_to=[], remove_unused_columns=False,
    bf16=args.bf16, max_grad_norm=1.0, dataloader_num_workers=0, disable_tqdm=True,
)
if sched == "warmup_stable_decay":
    steps_total = args.max_steps if args.max_steps > 0 else int(len(trained) / (args.bs * args.ga) * args.epochs)
    ta.lr_scheduler_kwargs["num_decay_steps"] = int(0.2 * steps_total)
    ta.lr_scheduler_kwargs["num_stable_steps"] = steps_total - args.warmup - int(0.2 * steps_total)

optimizers = (None, None)
if args.optim == "adam_l2":
    from transformers import get_scheduler
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr, betas=(args.b1, args.b2), eps=args.eps, weight_decay=args.wd)
    steps_total = args.max_steps if args.max_steps > 0 else int(len(trained) / (args.bs * args.ga) * args.epochs)
    sch = get_scheduler(sched, opt, num_warmup_steps=args.warmup, num_training_steps=steps_total,
                        scheduler_specific_kwargs=sched_kwargs)
    optimizers = (opt, sch)

trainer = Trainer(model=model, args=ta, train_dataset=Docs(), data_collator=collate, callbacks=[LrLog()],
                  optimizers=optimizers)
trainer.train()

m = trainer.model
if args.lora:
    m = m.merge_and_unload()
W1 = m.get_input_embeddings().weight.detach().float().cpu().contiguous()
steps = len(lrs)
ga = args.ga
docs_per_step = [sum(seen_batches[k * ga:(k + 1) * ga], []) for k in range(steps)]
save_file({"W0": W0.contiguous(), "W1": W1, "W1_bf16": W1.bfloat16().contiguous(),
           "W1_fp16": W1.half().contiguous()}, str(out / "emb.safetensors"))
gt = dict(args=vars(args), steps=steps, lr=lrs, emb=emb_name,
          claimed=claimed, trained=trained, swapped_positions=swap_pos, outside=outside,
          docs_per_step=docs_per_step,              # positions k into `trained`
          optimizer=type(trainer.optimizer.optimizer if hasattr(trainer.optimizer, "optimizer") else trainer.optimizer).__name__)
json.dump(gt, open(out / "truth.json", "w"))
if args.save_model:
    m.save_pretrained(str(out / "model"), safe_serialization=True)
json.dump({str(i): enc[i] for i in enc}, open(out / "tokens.json", "w"))
print("done", out, "steps", steps)
