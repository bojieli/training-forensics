# Snapshot of checkpoint-forensics inputs

Copied from https://github.com/bojieli/checkpoint-forensics (MIT, same author) at the commit in SOURCE_COMMIT:
`results/summary.json` and the fingerprint cards `ckptforensics/cards/*.json`.
Used by `scripts/roster/{coverage,pretrain_decay}.py` and `scripts/tokens/mine.py` when `CKPTF_REPO` is not a checkout,
so that `make all-offline` works from a clean clone.
