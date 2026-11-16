import json
d = [r for r in json.load(open("results/roster/p_intervals.json")) if "P_lo" in r]
lines = ["", "## Weight decay on the token vectors: exact intervals", "",
         "From `scripts/roster/p_interval.py`. A token vector that never received a gradient only shrinks by the factor P. "
         "Each number in it is published in 16 bits, so it pins P to the range of values that round to the published "
         "number; intersecting those ranges over every number of every untouched vector gives the range of P the weights "
         "allow. The total weight decay applied is sum(lr*wd) = -ln P. When every number shrank by exactly one smallest "
         "16-bit step, P is only bounded to ~0.2% (five pairs below share that range; their identical fitted P is the "
         "range's edge, not a shared recipe). Identical ranges elsewhere mean 'agree within the range', not 'same recipe'.", "",
         "| fine-tune | parent | untouched tokens | P range | total weight decay sum(lr*wd) | 16-bit steps shrunk (steps:count) |", "|---|---|---|---|---|---|"]
for r in sorted(d, key=lambda r: r["P_lo"]):
    st = ", ".join(f"{k}:{v:,}" for k, v in list(r["ulp_steps"].items())[:4])
    lines.append("| {} | {} | {:,} | [{:.6f}, {:.6f}] | [{:.5f}, {:.5f}] | {} |".format(
        r["child"], r["parent"], r["rows"], r["P_lo"], r["P_hi"], r["sum_lr_wd_lo"], r["sum_lr_wd_hi"], st))
open("results/roster/SWEEP.md", "a").write("\n".join(lines) + "\n")
print(len(d))
