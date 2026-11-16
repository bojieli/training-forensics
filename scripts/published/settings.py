"""Training settings each project documents, with sources pinned to a commit.

Used by report.py (PUBLISHED.md) and controls.py (CONTROLS.md) to say, for every published-data match,
whether the setting at which the token sets match is documented by the project or was found only by our sweep.
"""
SKY_YAML = ("https://github.com/NovaSky-AI/SkyThought/blob/0d190f11fd8e885bbe113aeccacba5ccde5b1102/"
            "skythought/train/LLaMA-Factory/examples/train_full/qwen2_full_sft.yaml")
SKY_DSINFO = ("https://github.com/NovaSky-AI/SkyThought/blob/0d190f11fd8e885bbe113aeccacba5ccde5b1102/"
              "skythought/train/LLaMA-Factory/data/dataset_info.json")
OT_YAML = ("https://github.com/open-thoughts/open-thoughts/blob/e86cf2dbbe69ae178aec302be33a3454a7183457/"
           "train/OpenThinker-7B.yaml")
S1_SH = "https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/train/sft.sh"
S1_README = "https://github.com/simplescaling/s1/blob/77272c6e925d610257a50b520bad15330b513389/README.md"

# child -> (matching setting, documentation status, sources)
DOCUMENTED = {
    "simplescaling/s1-32B": (
        "trl, block size 32,768 (any block size >= 16,384 matches)",
        "documented",
        f"block_size=32768 in [train/sft.sh]({S1_SH})"),
    "simplescaling/s1.1-32B": (
        "trl, block size 20,000",
        "documented",
        f"\"For s1.1, we set the block size to 20000\" in the [s1 README]({S1_README})"),
    "NovaSky-AI/Sky-T1-32B-Preview": (
        "system prompt from the data, cutoff 16,384",
        "documented",
        f"cutoff_len: 16384 in [qwen2_full_sft.yaml]({SKY_YAML}); the Sky-T1 entry of "
        f"[dataset_info.json]({SKY_DSINFO}) maps the `system` column"),
    "bespokelabs/Bespoke-Stratos-32B": (
        "system prompt from the data, cutoff 16,384",
        "documented by reference only",
        "the dataset card points to Sky-T1's codebase for training, and the model card's hyper-parameters "
        f"(lr 1e-5, 3 epochs, batch 96, cosine, warm-up 0.1) equal [Sky-T1's yaml]({SKY_YAML}); "
        "no Stratos-specific config is published"),
    "bespokelabs/Bespoke-Stratos-7B": (
        "system prompt from the data, cutoff 16,384",
        "documented by reference only",
        "as Bespoke-Stratos-32B"),
    "open-thoughts/OpenThinker-7B": (
        "framework default system prompt, cutoff 16,384",
        "documented",
        f"cutoff_len: 16384 and no `system` column mapping (so the template default is used) in "
        f"[train/OpenThinker-7B.yaml]({OT_YAML}); the exact default string (template qwen25) is not "
        "distinguished by the token set"),
}
