import json
d = [r for r in json.load(open("results/roster/p_intervals.json")) if "P_lo" in r]
lines = ["", "## Weight decay on the embedding: exact intervals", "",
         "From `scripts/roster/p_interval.py`: every coordinate of a never-trained row confines P to the reals that round "
         "to the published value; the intersection is what the weights allow. sum(lr*wd) = -ln P. When every coordinate "
         "moved by exactly one 16-bit step, P is only bounded to ~0.2% (five pairs below share that interval; their "
         "identical fitted P is the interval edge, not a shared recipe).", "",
         "| child | parent | rows | P interval | sum lr*wd | 16-bit steps moved |", "|---|---|---|---|---|---|"]
for r in sorted(d, key=lambda r: r["P_lo"]):
    st = ", ".join(f"{k}:{v:,}" for k, v in list(r["ulp_steps"].items())[:4])
    lines.append("| {} | {} | {:,} | [{:.6f}, {:.6f}] | [{:.5f}, {:.5f}] | {} |".format(
        r["child"], r["parent"], r["rows"], r["P_lo"], r["P_hi"], r["sum_lr_wd_lo"], r["sum_lr_wd_hi"], st))
open("results/roster/SWEEP.md", "a").write("\n".join(lines) + "\n")
print(len(d))
