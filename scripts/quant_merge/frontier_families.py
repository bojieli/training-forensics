# Frontier families appended to quant.FAMILIES: child = official 16-bit embedding source, base = public base (or None)
DS3B = "deepseek-ai/DeepSeek-V3-Base"
FRONTIER = {
    "DeepSeek-V3": dict(child="deepseek-ai/DeepSeek-V3", base=DS3B, quants=[
        ("st", "baseten/DeepSeek-V3-FP4"), ("gguf", "unsloth/DeepSeek-V3-GGUF::Q8_0", None),
        ("gguf", "unsloth/DeepSeek-V3-GGUF::Q4_K_M", None)]),
    "DeepSeek-V3-0324": dict(child="deepseek-ai/DeepSeek-V3-0324", base=DS3B, quants=[
        ("st", "nvidia/DeepSeek-V3-0324-NVFP4"), ("gguf", "unsloth/DeepSeek-V3-0324-GGUF::Q8_0", None)]),
    "DeepSeek-R1": dict(child="deepseek-ai/DeepSeek-R1", base=DS3B, quants=[
        ("st", "nvidia/DeepSeek-R1-FP4"), ("gguf", "unsloth/DeepSeek-R1-GGUF::Q8_0", None),
        ("gguf", "unsloth/DeepSeek-R1-GGUF::Q4_K_M", None)]),
    "DeepSeek-V3.2": dict(child="deepseek-ai/DeepSeek-V3.2", base="deepseek-ai/DeepSeek-V3.2-Exp-Base", quants=[
        ("st", "nvidia/DeepSeek-V3.2-NVFP4"), ("st", "amd/DeepSeek-V3.2-mxfp4"),
        ("gguf", "unsloth/DeepSeek-V3.2-GGUF::Q8_0", None), ("mlx", "mlx-community/DeepSeek-V3.2-4bit", None)]),
    "Qwen3-235B-A22B": dict(child="Qwen/Qwen3-235B-A22B", base=None, quants=[
        ("st", "Qwen/Qwen3-235B-A22B-FP8"), ("st", "nvidia/Qwen3-235B-A22B-NVFP4"),
        ("gguf", "Qwen/Qwen3-235B-A22B-GGUF::Q8_0", None)]),
    "Qwen3-235B-A22B-Instruct-2507": dict(child="Qwen/Qwen3-235B-A22B-Instruct-2507", base=None, quants=[
        ("st", "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"), ("gguf", "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF::Q8_0", None)]),
    "Qwen3-30B-A3B": dict(child="Qwen/Qwen3-30B-A3B", base="Qwen/Qwen3-30B-A3B-Base", quants=[
        ("st", "Qwen/Qwen3-30B-A3B-FP8"), ("gguf", "Qwen/Qwen3-30B-A3B-GGUF::Q8_0", None)]),
    "Qwen3-30B-A3B-Instruct-2507": dict(child="Qwen/Qwen3-30B-A3B-Instruct-2507", base="Qwen/Qwen3-30B-A3B-Base", quants=[
        ("st", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"), ("gguf", "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF::Q8_0", None)]),
    "Qwen3-14B": dict(child="Qwen/Qwen3-14B", base="Qwen/Qwen3-14B-Base", quants=[
        ("st", "Qwen/Qwen3-14B-FP8"), ("st", "Qwen/Qwen3-14B-AWQ"), ("st", "unsloth/Qwen3-14B-unsloth-bnb-4bit"),
        ("gguf", "Qwen/Qwen3-14B-GGUF::Q8_0", None), ("mlx", "lmstudio-community/Qwen3-14B-MLX-4bit", None)]),
    "Qwen3.5-397B-A17B": dict(child="Qwen/Qwen3.5-397B-A17B", base=None, quants=[
        ("st", "Qwen/Qwen3.5-397B-A17B-FP8"), ("st", "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"),
        ("st", "nvidia/Qwen3.5-397B-A17B-NVFP4")]),
    "Llama-4-Scout-Instruct": dict(child="unsloth/Llama-4-Scout-17B-16E-Instruct", base="unsloth/Llama-4-Scout-17B-16E", quants=[
        ("st", "nvidia/Llama-4-Scout-17B-16E-Instruct-FP8"), ("st", "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic"),
        ("st", "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"),
        ("gguf", "ggml-org/Llama-4-Scout-17B-16E-Instruct-GGUF::Q8_0", None),
        ("mlx", "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit", None)]),
    "Llama-4-Maverick-Instruct": dict(child="unsloth/Llama-4-Maverick-17B-128E-Instruct", base="unsloth/Llama-4-Maverick-17B-128E", quants=[
        ("st", "unsloth/Llama-4-Maverick-17B-128E-Instruct-FP8"), ("st", "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8"),
        ("st", "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4"),
        ("st", "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-quantized.w4a16")]),
    "gpt-oss-120b": dict(child="openai/gpt-oss-120b", base=None, quants=[
        ("st", "unsloth/gpt-oss-120b-unsloth-bnb-4bit"), ("gguf", "ggml-org/gpt-oss-120b-GGUF::mxfp4", None),
        ("mlx", "mlx-community/gpt-oss-120b-MXFP4-Q8", None)]),
    "gpt-oss-20b": dict(child="openai/gpt-oss-20b", base=None, quants=[
        ("st", "unsloth/gpt-oss-20b-unsloth-bnb-4bit"), ("gguf", "ggml-org/gpt-oss-20b-GGUF::mxfp4", None),
        ("gguf", "unsloth/gpt-oss-20b-GGUF::Q8_0", None), ("mlx", "mlx-community/gpt-oss-20b-MXFP4-Q8", None)]),
    "Kimi-K2-Thinking": dict(child="moonshotai/Kimi-K2-Thinking", base="moonshotai/Kimi-K2-Base", quants=[
        ("st", "nvidia/Kimi-K2-Thinking-NVFP4"), ("gguf", "unsloth/Kimi-K2-Thinking-GGUF::Q8_0", None)]),
    "Kimi-K2.5": dict(child="moonshotai/Kimi-K2.5", base="moonshotai/Kimi-K2-Base", quants=[
        ("st", "nvidia/Kimi-K2.5-NVFP4"), ("st", "amd/Kimi-K2.5-MXFP4")]),
    "GLM-4.5": dict(child="zai-org/GLM-4.5", base="zai-org/GLM-4.5-Base", quants=[
        ("st", "zai-org/GLM-4.5-FP8"), ("gguf", "unsloth/GLM-4.5-GGUF::Q8_0", None)]),
    "GLM-4.5-Air": dict(child="zai-org/GLM-4.5-Air", base="zai-org/GLM-4.5-Air-Base", quants=[
        ("st", "zai-org/GLM-4.5-Air-FP8"), ("gguf", "unsloth/GLM-4.5-Air-GGUF::Q8_0", None)]),
    "GLM-5.3": dict(child="zai-org/GLM-5.3-BF16", base=None, quants=[
        ("st", "zai-org/GLM-5.3"), ("st", "nvidia/GLM-5.3-NVFP4"), ("gguf", "unsloth/GLM-5.3-GGUF::Q8_0", None)]),
}
