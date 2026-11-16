"""Pinned revisions for the Reflection-70B lineage checks (scripts/reflection_check/, scripts/fp32scan/lineage70b*.py).
A call that passes no revision resolves to the pin below, never to a moving `main`."""
REFLECTION = "mattshumer/Reflection-Llama-3.1-70B"
# three weight uploads: 'dup' (2024-09-05 18:29, "Duplicate from sahil2801/reflect_v5_70", 123 shards),
# 'first' (2024-09-05 23:53, 162 shards), and the re-upload of 2024-09-06 that is still on main.
REFL_DUP = "276a4a0a0a"
REFL_FIRST = "e7bfef9796"
REFL_MAIN = "a376762159d10b8077c6a162ebd2f72267fe8a2f"  # main on 2026-11-16 (last commit 2024-09-24: model card note)
PIN = {
    REFLECTION: REFL_MAIN,
    "NousResearch/Meta-Llama-3-70B-Instruct": "84cbdcd4bcccad50126c29ec7f7a476dec014fcf",
    "NousResearch/Meta-Llama-3-70B": "fdbf133d993d2b8157e518bfc8bc4570b2d31c45",
    "unsloth/Meta-Llama-3.1-70B": "1b7306651142d0cc65d993076a250a6a82cf046c",
    "unsloth/Meta-Llama-3.1-70B-Instruct": "1fdd0a465a29664d155aee8a9f77c55a65cc8d5f",
    "unsloth/Llama-3.3-70B-Instruct": "99cd0d2c829e92a67c844f9144c2509632e5c87f",
}
