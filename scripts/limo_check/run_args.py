#!/usr/bin/env python3
"""Step 2: run metadata of the release, read without executing any pickle.

training_args.bin is a torch zip archive whose data.pkl holds the TrainingArguments __dict__; we walk its opcodes
with pickletools and keep (key, scalar value) pairs. Paths are reduced to their last component (the release's
config.json and training_args.bin contain cluster paths of the authors; we do not copy them). -> out/run_args.json"""
import json, os, pickletools, zipfile
from common import D, save
import pins

F = D + "limo_files/"
z = zipfile.ZipFile(F + "training_args.bin")
pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]
ops = list(pickletools.genops(z.read(pk)))
SCALAR = {"BININT", "BININT1", "BININT2", "LONG1", "BINFLOAT", "NONE", "NEWTRUE", "NEWFALSE",
          "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"}
STR = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"}
seq = [(op.name, a) for op, a, _ in ops if op.name not in ("BINPUT", "LONG_BINPUT", "MEMOIZE", "PUT", "FRAME")]
args = {}
for (n1, k), (n2, v) in zip(seq, seq[1:]):
    if n1 in STR and n2 in SCALAR and isinstance(k, str) and k.isidentifier() and k not in args:
        args[k] = {"NONE": None, "NEWTRUE": True, "NEWFALSE": False}.get(n2, v)
KEEP = ["per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay", "num_train_epochs",
        "max_steps", "warmup_ratio", "warmup_steps", "adam_beta1", "adam_beta2", "adam_epsilon", "max_grad_norm", "seed",
        "data_seed", "bf16", "fp16", "resume_from_checkpoint", "dataloader_drop_last", "save_strategy", "output_dir", "deepspeed"]
run = {}
for k in KEEP:
    if k in args:
        v = args[k]
        run[k] = os.path.basename(v.rstrip("/")) if isinstance(v, str) and "/" in v else v
ts = json.load(open(F + "trainer_state.json"))
cfg = json.load(open(F + "config.json"))
nproc = pins.N_RANKS  # one optimizer partition per process
spe = ts["max_steps"] // ts["num_train_epochs"]
bs = run.get("per_device_train_batch_size"); ga = run.get("gradient_accumulation_steps")
per_step = nproc * bs * ga
out = {
    "source": {"training_args.bin": f"{pins.LIMO}@{pins.LIMO_REV}", "trainer_state.json": f"{pins.LIMO}@{pins.LIMO_REV}",
               "config.json": f"{pins.LIMO}@{pins.LIMO_REV}", "num_processes": f"{pins.N_RANKS} ZeRO-3 optimizer partitions in {pins.LIMO}@{pins.LIMO_OPTIM_REV}"},
    "training_args": run,
    "trainer_state": {k: ts[k] for k in ("global_step", "epoch", "max_steps", "num_train_epochs", "train_batch_size", "total_flos", "num_input_tokens_seen")},
    "config": {"base_name_or_path_basename": os.path.basename(cfg.get("_name_or_path", "").rstrip("/")),
               "torch_dtype": cfg.get("torch_dtype"), "tie_word_embeddings": cfg.get("tie_word_embeddings"),
               "transformers_version": cfg.get("transformers_version")},
    "num_processes": nproc,
    "steps_per_epoch": spe,
    "examples_per_optimizer_step": per_step,
    # HF Trainer does not drop the last partial batch by default: ceil(n / per_step) = spe
    "implied_examples_per_epoch": [per_step * (spe - 1) + 1, per_step * spe] if not run.get("dataloader_drop_last") else [per_step * spe, per_step * (spe + 1) - 1],
    "lr_at_logged_steps": [(h["step"], h.get("learning_rate")) for h in ts["log_history"] if "learning_rate" in h][:3],
}
# total_flos: HF adds 6 * input tokens * model.num_parameters(exclude_embeddings=True) per process. Under ZeRO-3 the
# partitioned parameters report 0 elements, so the count depends on which small parameters stay unpartitioned.
f6 = ts["total_flos"] / 6
cand = {"q/k/v biases only (64 x 7,168)": 64 * 7168, "norms + q/k/v biases (64 x 17,408)": 64 * 17408,
        "norms + q/k/v biases + final norm": 64 * 17408 + 5120, "norms + final norm": 64 * 2 * 5120 + 5120}
out["total_flos_check"] = {"total_flos": ts["total_flos"], "epochs_done": ts["epoch"],
                           "tokens_per_epoch_if_counted_params": {k: round(f6 / n / ts["epoch"]) for k, n in cand.items()},
                           "integral_divisor": [k for k, n in cand.items() if f6 % n == 0]}
print(json.dumps(out, indent=1, default=str), flush=True)
save("run_args", out)
