#!/usr/bin/env python3
"""Figure (b): DeepSeek special-token grid. Reads results/frontier_ds/deepseek.json,
writes results/frontier_ds/fig_b_deepseek_special_tokens.{png,json}."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results" / "frontier_ds"
COL = {"updated": "#0072B2", "barely changed": "#E69F00", "bit-identical": "#D9D9D9", "not testable": "#FFFFFF"}
ORDER = list(COL)


def short(c):
    s = c.split("/")[-1].replace("DeepSeek-", "")
    return s + " (spans V3.1)" if s == "V3.1-Nex-N1" else s


def main():
    d = json.load(open(R / "deepseek.json"))
    comps = [c for c in d["comparisons"] if "error" not in c]
    panels = [("V3 line (18 named tokens)", [c for c in comps if "V3" in c["parent"]]),
              ("V4 line (named tokens)", [c for c in comps if "V4" in c["parent"]])]
    out = {}
    fig, axes = plt.subplots(2, 1, figsize=(16, 9.5), gridspec_kw=dict(height_ratios=[5, 7]))
    for ax, (title, cs) in zip(axes, panels):
        names = {}
        for c in cs:
            for t in c["named_detail"]:
                names.setdefault(t["id"], [])
                if t["token"] not in names[t["id"]]:
                    names[t["id"]].append(t["token"])
        toks = {i: " / ".join(v) for i, v in names.items()}
        ids = sorted(toks)
        rows = []
        labels = []
        for c in cs:
            m = {t["id"]: t for t in c["named_detail"]}
            row = []
            for i in ids:
                t = m.get(i)
                s = t["status"] if t else "not testable"
                row.append(ORDER.index(s))
            rows.append(row)
            labels.append(f"{short(c['parent'])} → {short(c['child'])}")
            out[labels[-1]] = {toks[i]: ORDER[v] for i, v in zip(ids, row)}
        ax.imshow(rows, cmap=ListedColormap([COL[k] for k in ORDER]), vmin=0, vmax=3, aspect="auto")
        for r, c in enumerate(cs):
            m = {t["id"]: t for t in c["named_detail"]}
            for j, i in enumerate(ids):
                if m.get(i) and any("unrelated" in f for f in m[i]["overwrite_flags"]):
                    ax.text(j, r, "×", ha="center", va="center", fontsize=9, color="black")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_xticks(range(len(ids)), [toks[i].replace("▁", " ").replace("｜", "|") for i in ids], rotation=90, fontsize=6.5)
        ax.set_xticks([x - 0.5 for x in range(len(ids) + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(len(labels) + 1)], minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        ax.set_title(title, loc="left", fontsize=10)
    fig.legend(handles=[Patch(facecolor=COL[k], edgecolor="#999", label=k) for k in ORDER] +
               [Patch(facecolor="white", edgecolor="white", label="× = start vector gone (replaced, or erased by heavy training)")],
               loc="upper right", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("DeepSeek named special tokens: did each release change them relative to its released base?\n"
                 f"(barely changed = under {int(100 * d['barely_threshold'])}% of the vector's numbers differ)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(R / "fig_b_deepseek_special_tokens.png", dpi=160)
    json.dump(out, open(R / "fig_b_deepseek_special_tokens.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
