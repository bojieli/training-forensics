#!/usr/bin/env python3
"""Article figures (a), (c), (d) and the non-DeepSeek special-token grid, from results/frontier/*.json.
Palette: the reference categorical palette (blue, orange, aqua, yellow, magenta, green), light mode."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import OUT  # noqa: E402

FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5fc4"]
NEUTRAL = "#b9b7b0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False})


def fig_a():
    d = json.load(open(OUT / "special_tokens.json"))
    base = "deepseek-ai/DeepSeek-V3-Base -> deepseek-ai/"
    series = [("DeepSeek-V3-0324", CAT[0]), ("DeepSeek-R1", CAT[1]), ("DeepSeek-R1-Zero", CAT[2])]
    bins = ["bit-identical", "under 1% of\nnumbers changed", "1-10% of\nnumbers changed", "over 10% of\nnumbers changed"]
    keys = ["bit-identical", "barely changed", "partly changed", "updated"]
    data = {}
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    w = 0.26
    for k, (name, col) in enumerate(series):
        v = d[base + name]
        tot = sum(v["states"].values())
        sh = [100 * v["states"][s] / tot for s in keys]
        data[name] = dict(zip(keys, sh), vocab=tot)
        xs = [i + (k - 1) * w for i in range(4)]
        bars = ax.bar(xs, sh, width=w - 0.03, color=col, label=name, edgecolor=SURF, linewidth=1)
        for x, y in zip(xs, sh):
            ax.text(x, y + 1.2, f"{y:.0f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(4), bins)
    ax.set_ylabel("share of the 129,280 vocabulary tokens")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("How much each token vector changed from DeepSeek-V3-Base\n(R1-Zero: most changes are tiny, so 'bit-identical' is a detection limit, not proof of absence)",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "a_change_histogram.png", dpi=180)
    json.dump(data, open(FIG / "a_change_histogram.json", "w"), indent=1)


def fig_c():
    dr = json.load(open(OUT / "decay_ranges.json"))
    dr = [x for x in dr if x["mode"] == "full rows" and x["shrink_pct_hi"] < 8]
    dr.sort(key=lambda x: x["shrink_pct_lo"])
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(dr) + 1.4))
    for i, x in enumerate(dr):
        lo, hi = x["shrink_pct_lo"], x["shrink_pct_hi"]
        coarse = (hi - lo) > 0.05
        col = NEUTRAL if coarse else CAT[0]
        if coarse:
            ax.plot([lo, hi], [i, i], color=col, linewidth=5, solid_capstyle="round")
        else:
            ax.plot([(lo + hi) / 2], [i], "o", color=col, markersize=7)
        ax.text(hi + 0.08, i, f"{lo:.2f}-{hi:.2f}%" if coarse else f"{(lo + hi) / 2:.2f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(dr)), [x["child"].split("/")[-1] for x in dr], fontsize=8)
    ax.set_xlabel("shrink of never-updated token vectors during the fine-tune (%)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(x["shrink_pct_hi"] for x in dr) * 1.15)
    ax.legend(handles=[Patch(color=CAT[0], label="exact range (narrower than the dot)"),
                       Patch(color=NEUTRAL, label="only bounded: vectors moved by one 16-bit step")],
              frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Total weight decay of each fine-tune:\nhow much its never-updated token vectors shrank",
                 loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "c_decay_ranges.png", dpi=180)
    json.dump(dr, open(FIG / "c_decay_ranges.json", "w"), indent=1)


def fig_d():
    comp = [x for x in json.load(open(OUT / "comparability.json")) if x.get("lane", "roster") == "roster"]
    order = ["token record", "untrainable only", "record, poor fit", "changed throughout", "not comparable", "unchanged", "shared matrix"]
    label = {"token record": "token record (decay fit ≥99%)", "untrainable only": "only untrainable tokens bit-identical (shared start)", "record, poor fit": "some bit-identical tokens, decay fit <99%",
             "changed throughout": "every token changed, consistent with the base + decay",
             "not comparable": "not a decayed copy of that base", "unchanged": "embedding unchanged",
             "shared matrix": "shared input/output matrix (no record)"}
    nice = {"tiiuae": "Falcon", "huggingfacetb": "SmolLM", "lmsys": "Vicuna", "01-ai": "Yi", "togethercomputer": "RedPajama"}
    for x in comp:
        x["family"] = nice.get(x["family"], x["family"])
    fam = Counter(x["family"] for x in comp)
    top = [f for f, _ in fam.most_common(16)]
    by = defaultdict(Counter)
    for x in comp:
        by[x["family"] if x["family"] in top else "other"][x["cls"]] += 1
    rows = sorted(by, key=lambda f: -sum(by[f].values()))
    rows = [r for r in rows if r != "other"] + (["other"] if "other" in by else [])
    fig, ax = plt.subplots(figsize=(8.6, 0.36 * len(rows) + 3.0))
    for i, f in enumerate(rows):
        left = 0
        for k, c in enumerate(order):
            n = by[f][c]
            if n:
                ax.barh(i, n, left=left, color=CAT[k], edgecolor=SURF, linewidth=1.5, height=0.7)
                if n >= 3:
                    ax.text(left + n / 2, i, str(n), ha="center", va="center", fontsize=7, color="white" if k in (0, 1, 5, 6) else INK)
                left += n
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("base → release comparisons")
    tot = Counter(x["cls"] for x in comp)
    ax.legend(handles=[Patch(color=CAT[k], label=f"{label[c]}: {tot[c]}") for k, c in enumerate(order)],
              frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=2, fontsize=7.5)
    ax.set_title(f"Which of the {len(comp)} base → release comparisons leave a token record", loc="left", fontsize=10, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "d_comparability.png", dpi=180)
    json.dump(dict(totals=tot, by_family={f: dict(by[f]) for f in rows}), open(FIG / "d_comparability.json", "w"), indent=1)


def fig_grid():
    g = json.load(open(OUT / "special_grid_nondeepseek.json"))
    cats = ["chat markers", "tool calls", "thinking", "vision / audio", "code fill-in"]
    col = {"updated": CAT[0], "barely changed": "#a9c9ef", "bit-identical": NEUTRAL, "replaced": CAT[1]}
    g = [x for x in g if "cells" in x]
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(g) + 2.2))
    for i, x in enumerate(g):
        for j, c in enumerate(cats):
            cell = x["cells"].get(c, {})
            n = sum(cell.values())
            if not x["testable"]:
                ax.add_patch(plt.Rectangle((j - 0.45, i - 0.4), 0.9, 0.8, facecolor=SURF, edgecolor=NEUTRAL, hatch="///", linewidth=0.6))
                continue
            if not n:
                ax.text(j, i, "–", ha="center", va="center", color=INK2, fontsize=8)
                continue
            left = j - 0.45
            for s in ("updated", "replaced", "barely changed", "bit-identical"):
                k = cell.get(s, 0)
                if k:
                    wdt = 0.9 * k / n
                    ax.add_patch(plt.Rectangle((left, i - 0.4), wdt, 0.8, facecolor=col[s], edgecolor=SURF, linewidth=1))
                    left += wdt
            ax.text(j, i, f"{cell.get('updated', 0) + cell.get('replaced', 0)}/{n}", ha="center", va="center", fontsize=7,
                    color=INK, bbox=dict(boxstyle="round,pad=0.15", facecolor=SURF, edgecolor="none", alpha=0.85))
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(len(g) - 0.5, -0.5)
    ax.set_xticks(range(len(cats)), cats, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(g)), [x["release"].split("/")[-1] for x in g], fontsize=8)
    for s in ("left", "bottom", "top"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[Patch(color=col["updated"], label="updated"), Patch(color=col["barely changed"], label="barely changed (≤1% of numbers)"),
                       Patch(color=col["bit-identical"], label="bit-identical to the base (after decay)"), Patch(color=col["replaced"], label="replaced (unrelated to the base vector)"),
                       Patch(facecolor=SURF, edgecolor=NEUTRAL, hatch="///", label="not testable (decay fit <99%)")],
              frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.01), ncol=2, fontsize=7.5)
    ax.set_title("Named special tokens updated by each release\n(count updated / total in the category)", loc="left", fontsize=10, color=INK, pad=24)
    fig.tight_layout()
    fig.savefig(FIG / "b2_special_grid_nondeepseek.png", dpi=180)


if __name__ == "__main__":
    fig_a(); fig_c(); fig_d(); fig_grid()
    print("ok")
