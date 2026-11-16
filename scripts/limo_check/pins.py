"""Pinned inputs for the LIMO independent check. Every download in fetch.py uses these revisions."""
# GAIR/LIMO model repo: four commits. 1833babb94 uploaded the epoch-9 checkpoint *with* the DeepSpeed
# optimizer shards (global_step234/...optim_states.pt); 910230fdaa deleted them; 7b59ab38 is main
# (README only on top). Weights are identical in 1833babb94 and 7b59ab38 (same LFS sha256, checked in fetch.py).
LIMO = "GAIR/LIMO"
LIMO_REV = "7b59ab38ade231f371a420b2772fe51802a1319e"
LIMO_OPTIM_REV = "1833babb94e0b71051f8773911e2ece8ed6a0a51"
LIMO_DELETE_REV = "910230fdaaa8716a3fd8c521138cbe9e5019ddaf"
N_RANKS = 32
OPTIM_FILE = "global_step234/bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"

# embedding matrices fetched by ranged reads (tag -> (repo, revision))
MODELS = {
    "limo": (LIMO, LIMO_REV),
    "q32i": ("Qwen/Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"),
    "q32b": ("Qwen/Qwen2.5-32B", "1818d35814b8319459f4bd55ed1ac8709630f003"),
    "qc32i": ("Qwen/Qwen2.5-Coder-32B-Instruct", "381fc969f78efac66bc87ff7ddeadb7e73c218a7"),
    "qc32b": ("Qwen/Qwen2.5-Coder-32B", "2e12b5f7bc878d424d222e224ed40aee564ec45f"),
    "limo2": ("GAIR/LIMO-v2", "9510d41a1f5aca4bc86bbb52d63d0ab13da52ab1"),
    "s1": ("simplescaling/s1-32B", "569f1b8dc020ebf57c9996462f3b58b1eb8aec8f"),
}
# small files of the release (run metadata), read for fields only; not redistributed
RUN_FILES = ["trainer_state.json", "training_args.bin", "config.json", "tokenizer.json"]
DATASETS = {
    "limo_hf": ("GAIR/LIMO", "b60f4462da9d927930b9c9bd43399cf875564416", "limo.jsonl"),
    "limo2_hf": ("GAIR/LIMO-v2", "ef8bf143f8cc56ca8634795835b55083a3ae3061", "limo-v2.jsonl"),
    "s1k_tok": ("simplescaling/s1K_tokenized", "bd41c1e1caf2abec9535b9bbac1f2db7e960b77d", None),
}
# GitHub copy of the training set and evaluation sets. train/data/limo.json is unchanged since its first
# commit 70074cd (2025-02-05); we check out the later commit 2284c6a and verify the blob is identical.
GITHUB = "https://github.com/GAIR-NLP/LIMO"
GITHUB_REV = "2284c6a0e6653aa8894bd12fdecc1212ba706c3a"
GITHUB_FIRST = "70074cd1a654eff1cee17df6ed5e685ca198a499"
# train_limo.yaml: cutoff_len 32768 at 70074cd (2025-02-05, the day of the release), 16384 from 04efb71 (2025-02-14)
GITHUB_YAML_CHANGE = "04efb7180099439a5a3651b4256523bf0a9c8480"
