"""Candidate frontier comparisons (parent -> child) not in the original roster sweep.
Base -> released model where a base is published; otherwise older -> newer release."""
B = [  # (parent, child, kind)
 ("baidu/ERNIE-4.5-300B-A47B-Base-PT", "baidu/ERNIE-4.5-300B-A47B-PT", "base"),
 ("baidu/ERNIE-4.5-21B-A3B-Base-PT", "baidu/ERNIE-4.5-21B-A3B-PT", "base"),
 ("baidu/ERNIE-4.5-21B-A3B-Base-PT", "baidu/ERNIE-4.5-21B-A3B-Thinking", "base"),
 ("baidu/ERNIE-4.5-VL-424B-A47B-Base-PT", "baidu/ERNIE-4.5-VL-424B-A47B-PT", "base"),
 ("tencent/Hunyuan-A13B-Pretrain", "tencent/Hunyuan-A13B-Instruct", "base"),
 ("tiiuae/Falcon-H1-34B-Base", "tiiuae/Falcon-H1-34B-Instruct", "base"),
 ("swiss-ai/Apertus-70B-2509", "swiss-ai/Apertus-70B-Instruct-2509", "base"),
 ("swiss-ai/Apertus-v1.5-70B-base", "swiss-ai/Apertus-v1.5-70B-SFT", "base"),
 ("swiss-ai/Apertus-v1.5-70B-SFT", "swiss-ai/Apertus-v1.5-70B-RL", "stage"),
 ("swiss-ai/Apertus-v1.5-70B-base", "swiss-ai/Apertus-v1.5-70B", "base"),
 ("Qwen/Qwen3-Coder-Next-Base", "Qwen/Qwen3-Coder-Next", "base"),
 ("arcee-ai/Trinity-Mini-Base", "arcee-ai/Trinity-Mini", "base"),
 ("arcee-ai/Trinity-Large-Base", "arcee-ai/Trinity-Large-Thinking", "base"),
 ("arcee-ai/Trinity-Large-Base", "arcee-ai/Trinity-Large-Preview", "base"),
 ("arcee-ai/Trinity-Large-TrueBase", "arcee-ai/Trinity-Large-Base", "stage"),
 ("mistralai/Mistral-Large-3-675B-Base-2512", "mistralai/Mistral-Large-3-675B-Instruct-2512-BF16", "base"),
 ("inclusionAI/Ling-2.6-1T-base", "inclusionAI/Ling-2.6-1T", "base"),
 ("inclusionAI/Ling-2.6-1T", "inclusionAI/Ring-2.6-1T", "sibling"),
 ("inclusionAI/Ling-1T", "inclusionAI/Ring-1T", "sibling"),
 ("inclusionAI/Ling-flash-base-2.0", "inclusionAI/Ling-flash-2.0", "base"),
 ("inclusionAI/Ling-2.6-flash-base", "inclusionAI/Ling-2.6-flash", "base"),
 ("inclusionAI/Ling-mini-base-2.0", "inclusionAI/Ling-mini-2.0", "base"),
 ("inclusionAI/Ling-3.0-flash-base", "inclusionAI/Ling-3.0-flash", "base"),
 ("inclusionAI/Ling-3.0-tiny-base", "inclusionAI/Ling-3.0-tiny", "base"),
 ("ibm-granite/granite-4.0-h-small-base", "ibm-granite/granite-4.0-h-small", "base"),
 ("ibm-granite/granite-4.0-h-tiny-base", "ibm-granite/granite-4.0-h-tiny", "base"),
 ("ibm-granite/granite-4.1-8b-base", "ibm-granite/granite-4.1-8b", "base"),
 ("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16", "base"),
 ("nvidia/Nemotron-3-Labs-Ultra-Math-SFT", "nvidia/Nemotron-3-Labs-Ultra-Math-RL", "stage"),
 ("rednote-hilab/dots.llm1.base", "rednote-hilab/dots.llm1.inst", "base"),
 ("meta-llama/Llama-3.1-405B", "meta-llama/Llama-3.1-405B-Instruct", "base"),
 ("Qwen/Qwen3.5-4B-Base", "Qwen/Qwen3.5-4B", "base"),
 ("Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-2B", "base"),
 ("deepseek-ai/DeepSeek-V3.1-Base", "deepseek-ai/DeepSeek-V3.1", "base"),
 ("deepseek-ai/DeepSeek-V3-Base", "deepseek-ai/DeepSeek-V3.1-Base", "stage"),
 ("deepseek-ai/DeepSeek-V3.2-Exp-Base", "deepseek-ai/DeepSeek-V3.2-Exp", "base"),
 ("ByteDance-Seed/Seed-OSS-36B-Base-woSyn", "ByteDance-Seed/Seed-OSS-36B-Instruct", "base"),
 ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-RL-Zero-Math", "rl-zero"),
 ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-RL-Zero-Mix", "rl-zero"),
 ("XiaomiMiMo/MiMo-V2.6-Pro-MOPD", "XiaomiMiMo/MiMo-V2.6-Pro-RL", "stage"),
 ("XiaomiMiMo/MiMo-V2.6-Flash-MOPD", "XiaomiMiMo/MiMo-V2.6-Flash-RL", "stage"),
 ("LGAI-EXAONE/EXAONE-4.0-32B", "LGAI-EXAONE/EXAONE-4.0.1-32B", "successive"),
]
SUCC = [  # successive releases (older -> newer)
 ["zai-org/GLM-4.5", "zai-org/GLM-4.6", "zai-org/GLM-4.7", "zai-org/GLM-5", "zai-org/GLM-5.1", "zai-org/GLM-5.2", "zai-org/GLM-5.3-BF16"],
 ["MiniMaxAI/MiniMax-M2", "MiniMaxAI/MiniMax-M2.1", "MiniMaxAI/MiniMax-M2.5", "MiniMaxAI/MiniMax-M2.7"],
 ["moonshotai/Kimi-K2-Instruct", "moonshotai/Kimi-K2-Instruct-0905", "moonshotai/Kimi-K2-Thinking", "moonshotai/Kimi-K2.5", "moonshotai/Kimi-K2.6", "moonshotai/Kimi-K2.7-Code"],
 ["Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-235B-A22B-Thinking-2507"],
 ["meituan-longcat/LongCat-2.0", "meituan-longcat/LongCat-2.5"],
 ["meituan-longcat/LongCat-Flash-Chat", "meituan-longcat/LongCat-Flash-Thinking", "meituan-longcat/LongCat-Flash-Thinking-2601"],
 ["deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3-0324", "deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1-Terminus", "deepseek-ai/DeepSeek-V3.2-Exp", "deepseek-ai/DeepSeek-V3.2", "deepseek-ai/DeepSeek-V3.2-Speciale"],
 ["Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3-Next-80B-A3B-Thinking"],
]
EXTRA_BASE = [  # also base -> later release where the base is shared
 ("moonshotai/Kimi-K2-Base", "moonshotai/Kimi-K2.6"),
 ("zai-org/GLM-4.5-Base", "zai-org/GLM-4.6"),
]


def all_pairs():
    out = list(B)
    for chain in SUCC:
        out += [(a, b, "successive") for a, b in zip(chain, chain[1:])]
    out += [(a, b, "base-later") for a, b in EXTRA_BASE]
    return out
