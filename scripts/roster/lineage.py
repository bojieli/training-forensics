"""Hand-curated lineage facts used by build_pairs.py.

OVERRIDES[name]   -> replaces automatic parent resolution for a roster model.
                     keys: role, stage, parent_repo, parent_revision, parent_alt, how,
                     confidence, note, children, mirror_of
CHILD_OVERRIDES   -> repo -> list of child repos (replaces automatic child search)
EXTRA_PAIRS       -> well-known public parent->child pairs that are not in the roster.

Roles: "base" = no public parent (pretrained from scratch, or post-trained model whose
base was never released: stage says which); "derived" = public parent;
"unknown" = lineage unclear (candidate parent may be given with confidence low).
"""

K = "known lineage"


def D(parent, conf="high", how=K, note=None, **kw):
    d = {"role": "derived", "parent_repo": parent, "confidence": conf, "how": how}
    if note:
        d["note"] = note
    d.update(kw)
    return d


def B(children=None, stage="pretrained", note=None, conf="high", how=K, **kw):
    d = {"role": "base", "parent_repo": None, "stage": stage, "confidence": conf, "how": how}
    if children:
        d["children"] = children
    if note:
        d["note"] = note
    d.update(kw)
    return d


def U(cand=None, note=None, **kw):
    d = {"role": "unknown", "parent_repo": None, "confidence": "low", "how": "unresolved"}
    if cand:
        d["candidate_parent"] = cand
    if note:
        d["note"] = note
    d.update(kw)
    return d


POST = "post-trained (base not public)"

OVERRIDES = {
    # ---------------- DeepSeek
    "DeepSeek-V3-Base": B(["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324", "deepseek-ai/DeepSeek-R1-Zero"]),
    "DeepSeek-R1": D("deepseek-ai/DeepSeek-V3-Base", note="R1 is RL/SFT on DeepSeek-V3-Base (R1 paper)"),
    "DeepSeek-V2": B(["deepseek-ai/DeepSeek-V2-Chat", "deepseek-ai/DeepSeek-V2-Chat-0628"]),
    "DeepSeek-V2-Lite": B(["deepseek-ai/DeepSeek-V2-Lite-Chat"]),
    "DeepSeek-Coder-V2": D("deepseek-ai/DeepSeek-V2", "medium",
                           note="Coder-V2 continued pre-training from an intermediate DeepSeek-V2 checkpoint "
                                "(paper); released V2 is a later checkpoint of the same run (same init draw)",
                           children=["deepseek-ai/DeepSeek-Coder-V2-Instruct"]),
    "DeepSeek-V4-Flash": D("deepseek-ai/DeepSeek-V4-Flash-Base", "high", how="same-family naming + shared init draw"),
    "DeepSeek-V4-Pro": D("deepseek-ai/DeepSeek-V4-Pro-Base", "high", how="same-family naming + shared init draw"),
    "DeepSeek-V4-Flash-0731": D("deepseek-ai/DeepSeek-V4-Flash", "medium", how="same-family naming + shared init draw",
                                note="dated refresh of V4-Flash; V4-Flash-Base is the pre-trained ancestor",
                                parent_alt="deepseek-ai/DeepSeek-V4-Flash-Base"),
    "DeepSeek-V4-Pro-0813": D("deepseek-ai/DeepSeek-V4-Pro", "medium", how="same-family naming + shared init draw",
                              note="dated refresh of V4-Pro; V4-Pro-Base is the pre-trained ancestor",
                              parent_alt="deepseek-ai/DeepSeek-V4-Pro-Base"),
    "DeepSeek-V4.1-Flash-1017": D("deepseek-ai/DeepSeek-V4.1-Flash", "medium", how="same-family naming"),
    "DeepSeek-Coder-1.3B": B(["deepseek-ai/deepseek-coder-1.3b-instruct"]),
    "DeepSeek-Coder-33B": B(["deepseek-ai/deepseek-coder-33b-instruct"]),
    "DeepSeek-LLM-7B": B(["deepseek-ai/deepseek-llm-7b-chat"]),
    "DeepSeek-LLM-67B": B(["deepseek-ai/deepseek-llm-67b-chat"]),
    "DeepSeek-MoE-16B": B(["deepseek-ai/deepseek-moe-16b-chat"]),
    # ---------------- Moonshot
    "Kimi-K2-Base": B(["moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Instruct-0905", "moonshotai/Kimi-K2-Thinking"]),
    "Kimi-K2-Thinking": D("moonshotai/Kimi-K2-Base", "medium",
                          note="K2-Thinking is post-trained from the K2 line; immediate parent may be "
                               "K2-Instruct-0905 (not stated); Base is the safe ancestor",
                          parent_alt="moonshotai/Kimi-K2-Instruct-0905"),
    "Kimi-K2.5": D("moonshotai/Kimi-K2-Base", "medium",
                   note="K2.5 = continued multimodal pre-training on top of Kimi-K2-Base (K2.5 report); shared init draw"),
    "Moonlight-16B-A3B": B(["moonshotai/Moonlight-16B-A3B-Instruct"]),
    "Kimi-Linear-48B": B(["moonshotai/Kimi-Linear-48B-A3B-Instruct"]),
    # ---------------- MiniMax
    "MiniMax-Text-01": B(stage=POST, note="hybrid-attention MoE; base not released"),
    "MiniMax-M1": D("MiniMaxAI/MiniMax-Text-01", "high",
                    note="M1 = continued pre-training + RL on MiniMax-Text-01 (M1 report); shared init draw"),
    "MiniMax-M2": B(stage=POST),
    # ---------------- Zhipu
    "GLM-4.5-Base": B(["zai-org/GLM-4.5"]),
    "GLM-4.5-Air": B(["zai-org/GLM-4.5-Air"]),
    "GLM-4-32B": B(["zai-org/GLM-4-32B-0414", "zai-org/GLM-Z1-32B-0414"]),
    "GLM-4-9B": B(["THUDM/glm-4-9b-chat"]),
    "ChatGLM3-6B": B(["THUDM/chatglm3-6b"]),
    # ---------------- Qwen
    "Qwen3-32B": B(stage=POST, note="Qwen3-32B-Base was never released"),
    "Qwen3-235B-A22B": B(stage=POST, children=["Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Thinking-2507"],
                         note="base not released; 2507 Instruct/Thinking are further-trained from this line "
                              "(summary note: April upload's unused rows overwritten with CPU noise, 2507 keeps original init)"),
    "Qwen3-0.6B": B(["Qwen/Qwen3-0.6B"]),
    "Qwen3-1.7B": B(["Qwen/Qwen3-1.7B"]),
    "Qwen3-8B": B(["Qwen/Qwen3-8B"]),
    "Qwen3-14B": B(["Qwen/Qwen3-14B"]),
    "Qwen3-30B-A3B": B(["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-30B-A3B-Thinking-2507"]),
    "Qwen3.5-0.8B": B(["Qwen/Qwen3.5-0.8B"]),
    "Qwen3.5-9B": B(["Qwen/Qwen3.5-9B"]),
    "Qwen3.5-35B-A3B": B(["Qwen/Qwen3.5-35B-A3B"]),
    "Qwen2.5-0.5B": B(["Qwen/Qwen2.5-0.5B-Instruct"]),
    "Qwen2.5-7B": B(["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-Math-7B", "Qwen/Qwen2.5-Coder-7B"],
                    note="shares its init draw with Qwen2-7B: either the same seed or continued from Qwen2 "
                         "(diff vs Qwen/Qwen2-7B would tell)", candidate_parent="Qwen/Qwen2-7B"),
    "Qwen2.5-32B": B(["Qwen/Qwen2.5-32B-Instruct", "Qwen/QwQ-32B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"]),
    "Qwen2.5-72B": B(["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen2.5-Math-72B"],
                     note="shares its init draw with Qwen2-72B (seed reuse or continued training)",
                     candidate_parent="Qwen/Qwen2-72B"),
    "Qwen2-7B": B(["Qwen/Qwen2-7B-Instruct", "Qwen/Qwen2-Math-7B"]),
    "Qwen2-72B": B(["Qwen/Qwen2-72B-Instruct", "Qwen/Qwen2-Math-72B"]),
    "Qwen2-57B-A14B": B(["Qwen/Qwen2-57B-A14B-Instruct"], note="upcycled MoE (from Qwen2-7B-class dense per report; "
                                                           "hidden 3584 matches Qwen2-7B)", candidate_parent="Qwen/Qwen2-7B"),
    "Qwen1.5-7B": B(["Qwen/Qwen1.5-7B-Chat"], note="shares init draw with Qwen-7B", candidate_parent="Qwen/Qwen-7B"),
    "Qwen1.5-110B": B(["Qwen/Qwen1.5-110B-Chat"], note="shares init draw with Qwen-72B (seed reuse; shapes differ)"),
    "Qwen-7B": B(["Qwen/Qwen-7B-Chat"]),
    "Qwen-72B": B(["Qwen/Qwen-72B-Chat"]),
    # ---------------- Meta (unsloth / Nous / huggyllama mirrors)
    "Llama-3-8B": B(["unsloth/llama-3-8b-Instruct"], mirror_of="meta-llama/Meta-Llama-3-8B"),
    "Llama-3.1-8B": B(["unsloth/Meta-Llama-3.1-8B-Instruct"], mirror_of="meta-llama/Llama-3.1-8B",
                      note="shares init draw with Llama-3-8B (seed reuse or continued training)",
                      candidate_parent="unsloth/llama-3-8b"),
    "Llama-3.1-70B": B(["unsloth/Meta-Llama-3.1-70B-Instruct", "unsloth/Llama-3.3-70B-Instruct"],
                       mirror_of="meta-llama/Llama-3.1-70B"),
    "Llama-3.3-70B": D("unsloth/Meta-Llama-3.1-70B", "medium",
                       note="Llama 3.3 70B = new post-training on the Llama 3.1 70B pre-trained base (shared init draw)",
                       mirror_of="meta-llama/Llama-3.3-70B-Instruct", parent_canonical="meta-llama/Llama-3.1-70B"),
    "Llama-3.2-1B": B(["unsloth/Llama-3.2-1B-Instruct"], mirror_of="meta-llama/Llama-3.2-1B",
                      note="pruned + distilled from Llama 3.1 8B (hidden 2048 vs 4096: embeddings not row-diffable)"),
    "Llama-3.2-3B": B(["unsloth/Llama-3.2-3B-Instruct"], mirror_of="meta-llama/Llama-3.2-3B",
                      note="pruned + distilled from Llama 3.1 8B (hidden 3072 vs 4096)"),
    "Llama-4-Scout": B(["unsloth/Llama-4-Scout-17B-16E-Instruct"], mirror_of="meta-llama/Llama-4-Scout-17B-16E"),
    "Llama-4-Maverick": D("unsloth/Llama-4-Maverick-17B-128E", "high", how="same-family naming",
                          mirror_of="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                          parent_canonical="meta-llama/Llama-4-Maverick-17B-128E"),
    "LLaMA-1-7B": B(["lmsys/vicuna-7b-v1.3"], note="mirror of original LLaMA-7B; vicuna-7b-v1.3 is a full-weight fine-tune"),
    "LLaMA-1-65B": B(),
    "Llama-2-7B": B(["NousResearch/Llama-2-7b-chat-hf", "lmsys/vicuna-7b-v1.5", "codellama/CodeLlama-7b-hf"],
                    mirror_of="meta-llama/Llama-2-7b-hf"),
    "Llama-2-13B": B(["NousResearch/Llama-2-13b-chat-hf", "lmsys/vicuna-13b-v1.5", "codellama/CodeLlama-13b-hf"],
                     mirror_of="meta-llama/Llama-2-13b-hf"),
    "Llama-2-70B": B(["NousResearch/Llama-2-70b-chat-hf", "codellama/CodeLlama-70b-hf"], mirror_of="meta-llama/Llama-2-70b-hf"),
    "CodeLlama-34B": B(["codellama/CodeLlama-34b-Instruct-hf", "codellama/CodeLlama-34b-Python-hf"],
                       note="initialised from Llama 2 34B, which was never released"),
    "Hermes-3-405B": D("meta-llama/Llama-3.1-405B", "high", how="card / known lineage",
                       note="Llama-3.1-405B weights are gated (and not range-readable in checkpoint-forensics)"),
    "Hermes-4-405B": D("meta-llama/Llama-3.1-405B", "high", how="card / known lineage",
                       note="Hermes 4 405B trained from Llama-3.1-405B base"),
    "Nemotron-70B": D("unsloth/Meta-Llama-3.1-70B-Instruct", "high", how="card base_model (meta-llama mirror)",
                      parent_canonical="meta-llama/Llama-3.1-70B-Instruct"),
    "DeepSeek-R1-Distill-Llama-70B": D("unsloth/Llama-3.3-70B-Instruct", "high", how="card base_model (meta-llama mirror)",
                                       parent_canonical="meta-llama/Llama-3.3-70B-Instruct"),
    # ---------------- Google (unsloth mirrors)
    "Gemma-3-270M": B(["unsloth/gemma-3-270m-it"], mirror_of="google/gemma-3-270m"),
    "Gemma-3-1B": B(["unsloth/gemma-3-1b-it"], mirror_of="google/gemma-3-1b-pt"),
    "Gemma-3-4B": B(["unsloth/gemma-3-4b-it"], mirror_of="google/gemma-3-4b-pt"),
    "Gemma-3-12B": B(["unsloth/gemma-3-12b-it"], mirror_of="google/gemma-3-12b-pt"),
    "Gemma-3-27B": B(["unsloth/gemma-3-27b-it"], mirror_of="google/gemma-3-27b-pt"),
    "Gemma-3n-E4B": B(["unsloth/gemma-3n-E4B-it"], mirror_of="google/gemma-3n-E4B"),
    "Gemma-2-2B": B(["unsloth/gemma-2-2b-it"], mirror_of="google/gemma-2-2b"),
    "Gemma-2-27B": B(["unsloth/gemma-2-27b-it"], mirror_of="google/gemma-2-27b"),
    "Gemma-4-26B-A4B": B(["google/gemma-4-26b-a4b-it"]),
    "Gemma-4-31B": B(["google/gemma-4-31b-it"]),
    # ---------------- AI2
    "OLMo-2-32B": B(["allenai/OLMo-2-0325-32B-SFT", "allenai/OLMo-2-0325-32B-DPO", "allenai/OLMo-2-0325-32B-Instruct"]),
    "Olmo-3-32B": B(["allenai/Olmo-3-32B-Think-SFT", "allenai/Olmo-3.1-32B-Instruct-SFT", "allenai/Olmo-3.1-32B-Think"]),
    "OLMo-7B": B(["allenai/OLMo-7B-SFT-hf", "allenai/OLMo-7B-Instruct-hf"], init_revision="step0-tokens0B",
                 note="intermediate checkpoints are public as revisions (stepN-tokensMB)"),
    "OLMo-7B-Twin-2T": B(note="same recipe as OLMo-7B on a different (AMD) cluster"),
    "OLMo-7B-0424": B(["allenai/OLMo-7B-0424-SFT-hf", "allenai/OLMo-7B-0424-Instruct-hf"],
                      note="OLMo 1.7 (0424) — new pre-training run"),
    "OLMo-1B-0724": B(),
    "OLMo-1B": B(init_revision="step20000-tokens84B", note="intermediate checkpoints public as revisions"),
    # ---------------- EleutherAI: Pythia has step0 (= the init) as a public revision
    **{f"Pythia-{s}": B(ch, init_revision="step0",
                        note="revision step0 is the exact initialisation; 154 intermediate checkpoints public")
       for s, ch in [("70M", None), ("160M", None), ("410M", None), ("1B", None), ("1.4B", None),
                     ("2.8B", ["databricks/dolly-v2-3b"]), ("6.9B", ["databricks/dolly-v2-7b"]),
                     ("12B", ["databricks/dolly-v2-12b", "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"])]},
    "GPT-NeoX-20B": B(["togethercomputer/GPT-NeoXT-Chat-Base-20B"]),
    "GPT-J-6B": B(["nomic-ai/gpt4all-j"]),
    # ---------------- Mistral
    "Mistral-7B-v0.1": B(["mistralai/Mistral-7B-Instruct-v0.1", "HuggingFaceH4/mistral-7b-sft-beta",
                          "HuggingFaceH4/zephyr-7b-beta"]),
    "Mistral-7B-v0.3": D("mistralai/Mistral-7B-v0.1", "medium",
                         note="v0.3 extends vocab 32000->32768 by inserting 768 control tokens near the start, "
                              "so v0.1 rows sit 768 rows later in v0.3 (checkpoint-forensics note): use a shifted "
                              "row map, not common_rows. Immediate parent is the v0.2 base "
                              "(mistral-community/Mistral-7B-v0.2, community upload of the raw weights)",
                         row_map="parent row i (i>=3?) -> child row i+768; verify on special tokens",
                         parent_alt="mistral-community/Mistral-7B-v0.2",
                         children=["mistralai/Mistral-7B-Instruct-v0.3"]),
    "Mixtral-8x7B": D("mistralai/Mistral-7B-v0.1", "medium",
                      note="sparse upcycle: embedding shares Mistral-7B-v0.1's init draw",
                      children=["mistralai/Mixtral-8x7B-Instruct-v0.1"]),
    "Mixtral-8x7B-v0.1-base": D("mistralai/Mistral-7B-v0.1", "medium",
                                note="duplicate roster entry of Mixtral-8x7B; upcycled from Mistral-7B-v0.1",
                                children=["mistralai/Mixtral-8x7B-Instruct-v0.1"]),
    "Mixtral-8x22B": B(["mistralai/Mixtral-8x22B-Instruct-v0.1"]),
    "Mistral-Nemo-12B": B(["mistralai/Mistral-Nemo-Instruct-2407"]),
    "Mistral-Small-24B": B(["mistralai/Mistral-Small-24B-Instruct-2501"]),
    "Mistral-Large-2411": B(stage=POST, note="base never released; Mistral-Large-Instruct-2407 is a sibling, "
                                             "possibly ancestor", candidate_parent="mistralai/Mistral-Large-Instruct-2407"),
    "Ministral-8B-2512": B(["mistralai/Ministral-3-8B-Instruct-2512", "mistralai/Ministral-3-8B-Reasoning-2512"]),
    "Ministral-3B-2512": B(["mistralai/Ministral-3-3B-Instruct-2512", "mistralai/Ministral-3-3B-Reasoning-2512"]),
    # ---------------- Microsoft
    "Phi-4": B(stage=POST, children=["microsoft/Phi-4-reasoning", "microsoft/Phi-4-reasoning-plus"]),
    "Phi-3-mini": B(stage=POST),
    "Phi-3-medium": B(stage=POST),
    "Phi-2": B(),
    # ---------------- misc older bases
    "Falcon-7B": B(["tiiuae/falcon-7b-instruct"]),
    "Falcon-40B": B(["tiiuae/falcon-40b-instruct"]),
    "OPT-30B": B(["facebook/opt-iml-30b"]),
    "OPT-66B": B(),
    "StarCoder2-15B": B(["bigcode/starcoder2-15b-instruct-v0.1"]),
    "StableLM-2-12B": B(["stabilityai/stablelm-2-12b-chat"]),
    "OpenLLaMA-13B": B(),
    "RedPajama-7B": B(["togethercomputer/RedPajama-INCITE-7B-Instruct", "togethercomputer/RedPajama-INCITE-7B-Chat"]),
    "Arctic": B(["Snowflake/snowflake-arctic-instruct"]),
    "Jamba-v0.1": B(),
    "Grok-1-hf": B(mirror_of="xai-org/grok-1"),
    "Mamba-2.8B": B(),
    "Aquila2-34B": B(["BAAI/AquilaChat2-34B"]),
    "Skywork-13B": B(["Skywork/Skywork-13B-Math"]),
    "XVERSE-65B": B(["xverse/XVERSE-65B-Chat"]),
    "XVERSE-MoE-A36B": B(["xverse/XVERSE-MoE-A36B-Chat"]),
    "InternLM-20B": B(["internlm/internlm-chat-20b"]),
    "InternLM2-20B": B(["internlm/internlm2-chat-20b", "internlm/internlm2-chat-20b-sft"]),
    "InternLM2.5-7B": B(["internlm/internlm2_5-7b-chat"]),
    "Baichuan2-13B": B(["baichuan-inc/Baichuan2-13B-Chat"]),
    "Yi-34B": B(["01-ai/Yi-34B-Chat"]),
    "Yi-1.5-34B": D("01-ai/Yi-34B", "medium",
                    note="Yi-1.5 = continued pre-training of Yi on 500B more tokens (Yi-1.5 card); shared init draw",
                    children=["01-ai/Yi-1.5-34B-Chat"]),
    "BLOOM-176B": B(["bigscience/bloomz"]),
    "DBRX": B(mirror_of="databricks/dbrx-base"),
    "Apertus-70B": B(["swiss-ai/Apertus-70B-Instruct-2509"]),
    "SmolLM2-135M": B(["HuggingFaceTB/SmolLM2-135M-Instruct"]),
    "SmolLM2-360M": B(["HuggingFaceTB/SmolLM2-360M-Instruct"]),
    "SmolLM2-1.7B": B(["HuggingFaceTB/SmolLM2-1.7B-Instruct"]),
    "Granite-3.3-2B": B(["ibm-granite/granite-3.3-2b-instruct"]),
    "Hunyuan-A13B": B(["tencent/Hunyuan-A13B-Instruct"]),
    "Seed-OSS-36B-Base": B(["ByteDance-Seed/Seed-OSS-36B-Instruct"]),
    "gpt-oss-120b": B(stage=POST, children=["openai/gpt-oss-safeguard-120b"]),
    "gpt-oss-20b": B(stage=POST, children=["openai/gpt-oss-safeguard-20b"]),
    "Qwen3-Next-80B-A3B": B(stage=POST, note="Qwen3-Next base not released"),
    "Qwen3.5-27B": B(stage=POST),
}

CHILD_OVERRIDES: dict = {}

# --------------------------------------------------------------------------- extra pairs
# stage: what the child adds. datasets/training_config are what the authors publish.
EXTRA_PAIRS = []


def _e(parent, child, family, stage, datasets=None, config=None, note=None, parent_revision=None):
    EXTRA_PAIRS.append({"parent": parent, "child": child, "family": family, "stage": stage,
                        "parent_revision": parent_revision,
                        "datasets": datasets or [], "training_config": config, "note": note})


for sz in ("7B", "14B", "32B", "72B"):
    _e(f"Qwen/Qwen2.5-{sz}", f"Qwen/Qwen2.5-{sz}-Instruct", "Qwen2.5", "SFT+RLHF (closed data)")
for sz in ("4B", "8B", "14B"):
    _e(f"Qwen/Qwen3-{sz}-Base", f"Qwen/Qwen3-{sz}", "Qwen3", "4-stage post-training: long-CoT cold start, "
       "reasoning RL, thinking-mode fusion, general RL (closed data)")
_e("Qwen/Qwen3-30B-A3B-Base", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen3", "non-thinking post-training (closed)")

# OLMo 2 7B / 13B : base -> SFT -> DPO -> Instruct (RLVR); all data public (Tulu 3 mixtures)
OLMO2 = {
    "SFT": (["allenai/tulu-3-sft-olmo-2-mixture"], "open-instruct SFT; lr 2e-5 (7B) / 7.5e-6 (13B), 2 epochs, "
            "seq 4096, eff. batch 128 (OLMo 2 paper / model card)"),
    "DPO": (["allenai/olmo-2-1124-7b-preference-mix"], "length-normalised DPO, beta 5, lr 1e-6, 1 epoch (card)"),
    "Instruct": (["allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"], "RLVR with PPO (card)"),
}
for sz, tag in (("7B", "1124-7B"), ("13B", "1124-13B")):
    prev = f"allenai/OLMo-2-{tag}"
    for st in ("SFT", "DPO", "Instruct"):
        cur = f"allenai/OLMo-2-{tag}-{st}"
        ds, cfg = OLMO2[st]
        if sz == "13B" and st == "DPO":
            ds = ["allenai/olmo-2-1124-13b-preference-mix"]
        _e(prev, cur, "OLMo 2", st, ds, cfg)
        prev = cur
# OLMo 3 7B : base -> Instruct-SFT -> Instruct-DPO -> Instruct ; Think chain
prev = "allenai/Olmo-3-1025-7B"
for st, ds in (("Instruct-SFT", ["allenai/Dolci-Instruct-SFT"]), ("Instruct-DPO", ["allenai/Dolci-Instruct-DPO"]),
               ("Instruct", ["allenai/Dolci-Instruct-RL"])):
    cur = f"allenai/Olmo-3-7B-{st}"
    _e(prev, cur, "OLMo 3", st, ds, "open-instruct / OLMo-core; recipe in OLMo 3 report")
    prev = cur
prev = "allenai/Olmo-3-1025-7B"
for st, ds in (("Think-SFT", ["allenai/Dolci-Think-SFT"]), ("Think-DPO", ["allenai/Dolci-Think-DPO"]),
               ("Think", ["allenai/Dolci-Think-RL"])):
    cur = f"allenai/Olmo-3-7B-{st}"
    _e(prev, cur, "OLMo 3", st, ds, "open-instruct / OLMo-core; recipe in OLMo 3 report")
    prev = cur
# Tulu 3 on Llama-3.1-8B
_e("meta-llama/Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-SFT", "Tulu 3", "SFT",
   ["allenai/tulu-3-sft-mixture"], "lr 5e-6, 2 epochs, eff. batch 128, max seq 4096, linear warmup 0.03 (Tulu 3 paper Tab. 6)",
   note="parent gated (meta-llama); ungated mirror unsloth/Meta-Llama-3.1-8B")
_e("allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai/Llama-3.1-Tulu-3-8B-DPO", "Tulu 3", "DPO",
   ["allenai/llama-3.1-tulu-3-8b-preference-mixture"], "length-normalised DPO, beta 5, lr 5e-7, 1 epoch, batch 128")
_e("allenai/Llama-3.1-Tulu-3-8B-DPO", "allenai/Llama-3.1-Tulu-3-8B", "Tulu 3", "RLVR (PPO)",
   ["allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"], "PPO with verifiable rewards, lr 3e-7, KL beta 0.05")
_e("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", "Llama 3.1", "SFT+RS+DPO (closed)",
   note="both gated; ungated mirrors unsloth/Meta-Llama-3.1-8B -> unsloth/Meta-Llama-3.1-8B-Instruct")
_e("mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral", "instruct (closed)")
# Zephyr: fully public recipe (alignment-handbook)
_e("mistralai/Mistral-7B-v0.1", "HuggingFaceH4/mistral-7b-sft-beta", "Zephyr", "SFT",
   ["HuggingFaceH4/ultrachat_200k"], "alignment-handbook recipes/zephyr-7b-beta/sft/config_full.yaml: lr 2e-5, "
   "1 epoch, cosine, warmup 0.1, bf16, seq 2048, global batch 128")
_e("HuggingFaceH4/mistral-7b-sft-beta", "HuggingFaceH4/zephyr-7b-beta", "Zephyr", "DPO",
   ["HuggingFaceH4/ultrafeedback_binarized"], "alignment-handbook recipes/zephyr-7b-beta/dpo/config_full.yaml: "
   "beta 0.01, lr 5e-7, 3 epochs, linear, warmup 0.1, global batch 32")
# Gemma (tied)
_e("google/gemma-2-2b", "google/gemma-2-2b-it", "Gemma 2", "SFT+RLHF (closed)", note="tied embeddings; gated")
_e("unsloth/gemma-3-1b-pt", "unsloth/gemma-3-1b-it", "Gemma 3", "SFT+RL (closed)", note="tied embeddings")
# SmolLM
_e("HuggingFaceTB/SmolLM2-1.7B", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM2", "SFT+DPO",
   ["HuggingFaceTB/smoltalk", "HuggingFaceH4/ultrafeedback_binarized"], "alignment-handbook SmolLM2 recipe",
   note="tied embeddings")
_e("HuggingFaceTB/SmolLM3-3B-Base", "HuggingFaceTB/SmolLM3-3B", "SmolLM3", "mid-training + SFT + APO",
   ["HuggingFaceTB/smoltalk2"], "full recipe published (smollm repo)", note="tied embeddings")
# Falcon3
_e("tiiuae/Falcon3-7B-Base", "tiiuae/Falcon3-7B-Instruct", "Falcon3", "SFT+DPO (closed)")
_e("tiiuae/Falcon3-10B-Base", "tiiuae/Falcon3-10B-Instruct", "Falcon3", "SFT+DPO (closed)")
# Pythia fine-tunes with public data
_e("EleutherAI/pythia-6.9b", "databricks/dolly-v2-7b", "Pythia", "SFT", ["databricks/databricks-dolly-15k"],
   "dolly repo train_dolly.py: deepspeed, lr 5e-6, 2 epochs (7B/12B)")
_e("EleutherAI/pythia-2.8b", "databricks/dolly-v2-3b", "Pythia", "SFT", ["databricks/databricks-dolly-15k"],
   "dolly repo train_dolly.py")
_e("EleutherAI/pythia-12b", "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", "Pythia", "SFT",
   ["OpenAssistant/oasst1"], "Open-Assistant model/model_training configs (oasst-sft-4)")
_e("EleutherAI/pythia-1.4b", "EleutherAI/pythia-1.4b", "Pythia", "pre-training (step0 -> final)",
   note="same repo: parent revision step0 is the exact init, child is main", parent_revision="step0")
# community fine-tunes with public data
_e("mistralai/Mistral-7B-v0.1", "teknium/OpenHermes-2.5-Mistral-7B", "OpenHermes", "SFT",
   ["teknium/OpenHermes-2.5"], "axolotl; config in card (lr 2e-5? see card)", note="vocab extended by ChatML tokens")
_e("allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-SFT", "OLMo 2", "SFT", ["allenai/tulu-3-sft-olmo-2-mixture-0225"],
   "open-instruct SFT", note="tied? check")
_e("Qwen/Qwen2.5-Math-7B", "Qwen/Qwen2.5-Math-7B-Instruct", "Qwen2.5-Math", "SFT+GRPO (closed)")
