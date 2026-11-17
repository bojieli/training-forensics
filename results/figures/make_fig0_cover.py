#!/usr/bin/env python3
"""Cover figure: published training data vs token vectors changed (exact-match table).

Run from the repository root:  python3 results/figures/make_fig0_cover.py
Writes fig0_cover_matches.{png,svg,json} next to this file. Same style as make_figures.py.
Every number is parsed from results/published/PUBLISHED.md and results/large/LARGE.md.
"""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
plt.rcParams["font.family"] = ["Lato", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "path"
BLUE, ORANGE = "#0072B2", "#E69F00"
INK, INK2, INK3, GRID, PALE = "#111111", "#444444", "#6B6B6B", "#D9D9D9", "#EFEFEF"


def num(s):
    return int(s.replace(",", ""))


pub = open(os.path.join(RES, "published", "PUBLISHED.md")).read()
rows = {}
for line in pub.splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    if len(c) >= 8 and "/" in c[0] and re.fullmatch(r"[\d,]+", c[2] or "x"):
        rows[c[0]] = dict(in_data=num(c[2]), changed=num(c[3]), both=num(c[4]),
                          in_data_identical=num(c[5]), changed_not_in_data=num(c[6]))
large = open(os.path.join(RES, "large", "LARGE.md")).read()
m = re.search(r"\*\*sft\*\* \(truncated as in training\): ([\d,]+) distinct tokens in the data; "
              r"([\d,]+) of them changed; ([\d,]+) in the data but bit-identical.*?; (\d+) regular tokens "
              r"updated but absent", large)
tulu = dict(in_data=num(m.group(1)), both=num(m.group(2)), in_data_identical=num(m.group(3)),
            changed_not_in_data=int(m.group(4)))

s7, s32 = rows["bespokelabs/Bespoke-Stratos-7B"], rows["bespokelabs/Bespoke-Stratos-32B"]
assert s7 == s32
table = [
    ("s1-32B", rows["simplescaling/s1-32B"]),
    ("s1.1-32B", rows["simplescaling/s1.1-32B"]),
    ("Sky-T1-32B-Preview", rows["NovaSky-AI/Sky-T1-32B-Preview"]),
    ("Bespoke-Stratos 7B and 32B", s7),
    ("OpenThinker-7B", rows["open-thoughts/OpenThinker-7B"]),
]
for _, r in table:
    assert r["in_data"] == r["changed"] == r["both"] and r["in_data_identical"] == 0 \
        and r["changed_not_in_data"] == 0, r

W, H = 1600, 640
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def T(x, y, s, size=26, color=INK, weight="normal", ha="left"):
    ax.text(x, y, s, fontsize=size * 0.72, color=color, fontweight=weight, ha=ha, va="center")


# Cover: five exact matches only, large enough to read as a thumbnail. Tulu 3 405B (close, not
# exact) is reported in the article text and kept in the sidecar, not drawn.
T(60, 62, "Published data matches the weights, token for token", 54, weight="bold")
T(60, 124, "Five fine-tunes with public training data. Counts are distinct vocabulary tokens.", 29, INK2)

X0, X1, XE, X2, XR = 60, 850, 1085, 1320, 1540
y = 190
T(X0, y, "Fine-tune", 27, INK3, "bold")
T(X1, y, "Tokens in its published data", 27, INK3, "bold", ha="center")
T(X2, y, "Token vectors that changed", 27, INK3, "bold", ha="center")
ax.plot([X0, XR], [y + 28, y + 28], color=INK3, lw=1.5)
RH = 70
y0 = y + 28
for i, (name, r) in enumerate(table):
    yy = y0 + RH * i + RH / 2
    if i % 2 == 0:
        ax.add_patch(Rectangle((X0 - 20, y0 + RH * i), XR - X0 + 40, RH, fc=PALE, ec="none", zorder=0))
    T(X0, yy, name, 34, weight="bold")
    T(X1, yy, f"{r['in_data']:,}", 42, ha="center")
    T(XE, yy - 2, "=", 46, BLUE, "bold", ha="center")
    T(X2, yy, f"{r['changed']:,}", 42, BLUE, "bold", ha="center")
T(60, H - 30, "Method: @alapanchaudhuri  ·  Replication and data: github.com/bojieli/training-forensics", 22, INK3)

for ext in ("png", "svg"):
    fig.savefig(os.path.join(HERE, f"fig0_cover_matches.{ext}"), dpi=100, facecolor="white")
side = {
    "figure": "fig0_cover_matches",
    "size_px": [W, H],
    "sources": ["results/published/PUBLISHED.md (main table)",
                "results/large/LARGE.md (Tulu 3 405B, sft, truncated as in training)"],
    "note": "Counts are distinct vocabulary tokens. The five exact matches are drawn. Tulu 3 405B's SFT stage "
            "(124,327 of 124,572 in-data tokens changed; 0 regular tokens outside the data were updated) is "
            "checked here and reported in the article text, but not drawn.",
    "rows": [{"fine_tune": n, **r} for n, r in table],
    "not_drawn": [{"fine_tune": "Tulu 3 405B (SFT stage)", **tulu}],
}
json.dump(side, open(os.path.join(HERE, "fig0_cover_matches.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(side["rows"] + side["not_drawn"]))
