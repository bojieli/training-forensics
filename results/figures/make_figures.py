#!/usr/bin/env python3
"""Draw every figure of the article in one consistent style.

Run from the repository root:  python3 results/figures/make_figures.py
Writes fig*.png / fig*.svg / fig*.json (sidecar with sources) next to this file.
Every number drawn is taken from a result file named in the sidecar. CPU only.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
ROOT = os.path.dirname(RES)

plt.rcParams["font.family"] = ["Lato", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "path"

# Okabe-Ito
BLUE, ORANGE, SKY, GREEN = "#0072B2", "#E69F00", "#56B4E9", "#009E73"
YELLOW, VERM, PURPLE, BLACK = "#F0E442", "#D55E00", "#CC79A7", "#000000"
INK, INK2, INK3 = "#111111", "#444444", "#6B6B6B"
GRID, PALE = "#D9D9D9", "#EFEFEF"

# one meaning per colour, in every figure
STATE = {
    "bit-identical": ORANGE,
    "barely changed": YELLOW,
    "partly changed": SKY,
    "updated": BLUE,
}
STATE_TEXT = {"bit-identical": INK, "barely changed": INK, "partly changed": INK, "updated": "white"}


def px(pt_px):
    """font size given in pixels -> points at dpi 100"""
    return pt_px * 0.72


class Canvas:
    def __init__(self, w=1600, h=900):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")

    def text(self, x, y, s, size=26, color=INK, weight="normal", ha="left", va="center", **kw):
        return self.ax.text(x, y, s, fontsize=px(size), color=color, fontweight=weight,
                            ha=ha, va=va, **kw)

    def rect(self, x, y, w, h, color, r=0, ec="none", lw=0, z=2, **kw):
        if r:
            p = FancyBboxPatch((x + r, y + r), max(w - 2 * r, 0.01), max(h - 2 * r, 0.01),
                               boxstyle=f"round,pad={r}", fc=color, ec=ec, lw=lw, zorder=z,
                               mutation_aspect=1, **kw)
        else:
            p = Rectangle((x, y), w, h, fc=color, ec=ec, lw=lw, zorder=z, **kw)
        self.ax.add_patch(p)
        return p

    def line(self, x0, y0, x1, y1, color=GRID, lw=1.5, z=1, **kw):
        self.ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=z, solid_capstyle="round", **kw)

    def arrow(self, x0, y0, x1, y1, color=INK2, lw=3, ms=22, z=3):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms,
                                          color=color, lw=lw, zorder=z, shrinkA=0, shrinkB=0))

    def title(self, title, sub=None, sub2=None):
        self.text(40, 58, title, size=44, weight="bold")
        if sub:
            self.text(40, 112, sub, size=27, color=INK2)
        if sub2:
            self.text(40, 148, sub2, size=27, color=INK2)

    def footer(self, s, size=19):
        self.text(40, self.h - 24, s, size=size, color=INK3)

    def swatch(self, x, y, color, label, sub=None, size=25, glyph=None, hatch=None):
        """legend entry; returns nothing. y is the centre line."""
        self.rect(x, y - 15, 30, 30, color, r=5, hatch=hatch, ec=INK3 if hatch else "none",
                  lw=1 if hatch else 0)
        if glyph:
            self.text(x + 15, y, glyph, size=24, weight="bold", ha="center")
        self.text(x + 42, y, label, size=size, weight="bold")
        if sub:
            self.text(x + 42, y + 30, sub, size=21, color=INK2)

    def save(self, name, sidecar):
        for ext in ("png", "svg"):
            self.fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=100, facecolor="white")
        plt.close(self.fig)
        with open(os.path.join(HERE, f"{name}.json"), "w") as f:
            json.dump(sidecar, f, indent=1, ensure_ascii=False)
        print("wrote", name)


def load(rel):
    with open(os.path.join(RES, rel)) as f:
        return json.load(f)


REPO = "github.com/bojieli/training-forensics"

# --------------------------------------------------------------------------- fig 1


def fig1():
    readme = open(os.path.join(ROOT, "README.md")).read()
    m = re.search(r"D/lr -> ([0-9]+\.[0-9]+)", readme)
    ratio = float(m.group(1))  # 31.77

    c = Canvas(1600, 900)
    c.title("How training leaves a record in each token's vector",
            "Every token in the vocabulary has its own vector of numbers. "
            "Compare base model and fine-tune, token by token.")

    X0, X1, CW, N = 400, 770, 37, 8          # strip left edges, cell width, cells
    SW = CW * N
    c.text(X0 + SW / 2, 172, "Base model", size=27, weight="bold", ha="center", color=INK2)
    c.text(X1 + SW / 2, 172, "Fine-tuned model", size=27, weight="bold", ha="center", color=INK2)

    vals = [0.62, -0.34, 0.80, -0.70, 0.24, 0.52, -0.50, 0.36]   # schematic shape, not data
    signs = [1, -1, -1, 1, 1, -1, 1, -1]
    H = 124                                   # strip height
    rows = [
        (262, "Token never in\nthe training data", "absent"),
        (418, "Token seen in\none training step", "once"),
        (574, "Token never seen,\nweight decay is on", "decay"),
    ]

    def strip(x, yc):
        c.rect(x - 6, yc - H / 2, SW + 12, H, "#F6F6F6", r=8)
        c.line(x, yc, x + SW, yc, color="#BDBDBD", lw=1.5, z=3)

    def bars(x, yc, scale=1.0, color="#8C8C8C", z=4, alpha=1.0):
        for i, v in enumerate(vals):
            hgt = v * 40 * scale
            c.rect(x + i * CW + 7, yc - max(hgt, 0), CW - 14, abs(hgt), color, z=z, alpha=alpha)

    def badge(x, y, w, label, color, tcolor):
        c.rect(x, y, w, 48, color, r=8)
        c.text(x + w / 2, y + 23, label, size=27, weight="bold", ha="center", color=tcolor)

    for yc, label, kind in rows:
        c.text(40, yc, label, size=30, weight="bold", linespacing=1.25)
        strip(X0, yc)
        bars(X0, yc)
        c.arrow(X0 + SW + 22, yc, X1 - 22, yc)
        strip(X1, yc)
        tx = X1 + SW + 36
        if kind == "absent":
            bars(X1, yc)
            badge(tx, yc - 56, 210, "bit-identical", ORANGE, INK)
            c.text(tx, yc + 20, "Training never touches it,", size=26, color=INK2)
            c.text(tx, yc + 52, "so no number changes at all.", size=26, color=INK2)
        elif kind == "once":
            bars(X1, yc)
            for i, (v, s) in enumerate(zip(vals, signs)):
                tip = yc - v * 40
                xm = X1 + i * CW + CW / 2
                c.ax.add_patch(FancyArrowPatch((xm, tip), (xm, tip - s * 26), arrowstyle="-|>",
                                               mutation_scale=17, color=BLUE, lw=4, zorder=6,
                                               shrinkA=0, shrinkB=0))
            badge(tx, yc - 73, 262, "one uniform shift", BLUE, "white")
            c.text(tx, yc - 6, "Every number moves by the same", size=26, color=INK2)
            c.text(tx, yc + 25, "small amount, up or down: about", size=26, color=INK2)
            c.text(tx, yc + 56, f"{ratio:.1f} × the learning rate.", size=26, color=INK2)
        else:
            bars(X1, yc, color="#D2D2D2")
            bars(X1, yc, scale=0.72, color=GREEN, z=5)
            badge(tx, yc - 56, 278, "one uniform shrink", GREEN, "white")
            c.text(tx, yc + 20, "Every such vector shrinks by", size=26, color=INK2)
            c.text(tx, yc + 52, "exactly the same factor.", size=26, color=INK2)
    for y in (340, 496):
        c.line(40, y, 1560, y, color=GRID, lw=1.2)

    # what this recovers
    y0 = 672
    c.line(40, y0 - 18, 1560, y0 - 18, color=INK3, lw=1.5)
    c.text(40, y0 + 14, "What the weights then tell you", size=30, weight="bold")
    cards = [
        ("The token list", "which vocabulary tokens the", "training data contained"),
        ("The schedule", "how the learning rate changed", "over time (32-bit releases only)"),
        ("The family", "which base model, or family of", "models, a release started from"),
    ]
    cw = 496
    for i, (head, l1, l2) in enumerate(cards):
        x = 40 + i * (cw + 16)
        c.rect(x, y0 + 44, cw, 132, "#F6F6F6", r=10)
        c.rect(x, y0 + 44, 10, 132, INK3, r=4)
        c.text(x + 30, y0 + 74, head, size=29, weight="bold")
        c.text(x + 30, y0 + 114, l1, size=26, color=INK2)
        c.text(x + 30, y0 + 146, l2, size=26, color=INK2)
    c.footer(f"Schematic, not to scale. Method: Alapan Chaudhuri. Replication: {REPO}", size=22)
    c.save("fig1_mechanism", {
        "figure": "fig1_mechanism",
        "kind": "schematic diagram; bar heights and arrow directions are illustrative, not data",
        "numbers": {"31.8 (step size / learning rate)": {
            "value": ratio, "source": "README.md, 'The mechanism, in five lines', item 4 (D/lr -> 31.77)"}},
        "sources": ["README.md 'The mechanism, in five lines'"],
    })


# --------------------------------------------------------------------------- fig 2


def short_tok(t):
    t = t.replace("<｜", "").replace("｜>", "").replace("<|", "").replace("|>", "")
    return t.replace("▁", " ").strip()


def fig2():
    src = "frontier_ds/fig_b_deepseek_special_tokens.json"
    d = load(src)
    rows = [("V3", "V3-Base → V3"), ("V3-0324", "V3-Base → V3-0324"),
            ("R1", "V3-Base → R1"), ("R1-Zero", "V3-Base → R1-Zero")]
    toks = list(d[rows[0][1]].keys())
    by = {short_tok(t): t for t in toks}
    groups = [
        ("sentence\nmarkers", ["begin of sentence", "end of sentence", "pad"]),
        ("code\nfill-in", ["fim hole", "fim begin", "fim end"]),
        ("chat", ["User", "Assistant", "EOT"]),
        ("tool calls", ["tool calls begin", "tool calls end", "tool call begin", "tool call end", "tool sep"]),
        ("tool outputs", ["tool outputs begin", "tool outputs end", "tool output begin", "tool output end"]),
    ]
    # plain column labels for the abbreviated token names
    shown = {"pad": "padding", "fim hole": "fill-in hole", "fim begin": "fill-in begin",
             "fim end": "fill-in end", "EOT": "end of turn", "tool sep": "tool separator"}
    assert sorted(sum((g[1] for g in groups), [])) == sorted(by), sorted(by)

    c = Canvas(1600, 940)
    c.title("DeepSeek: which special tokens each release updated",
            "Each square is one special token's vector, compared with the base model, DeepSeek-V3-Base.",
            "R1 and R1-Zero left all nine tool tokens bit-identical.")
    GX, GY, CWd, RH, GAP = 215, 388, 66, 82, 22
    x = GX
    cols = []
    for gname, names in groups:
        gx0 = x
        for n in names:
            cols.append((x, n))
            c.text(x + CWd / 2 - 4, GY - 12, shown.get(n, n), size=25, rotation=52, ha="left",
                   va="bottom", rotation_mode="anchor")
            x += CWd
        c.line(gx0 + 3, GY + 4 * RH + 16, x - 5, GY + 4 * RH + 16, color=INK3, lw=2.5)
        c.text((gx0 + x) / 2, GY + 4 * RH + 30, gname, size=27, weight="bold", ha="center",
               va="top", linespacing=1.1)
        x += GAP
    grid = {}
    for r, (label, key) in enumerate(rows):
        y = GY + r * RH
        c.text(40, y + RH / 2, label, size=31, weight="bold")
        for cx, n in cols:
            st = d[key][by[n]]
            grid.setdefault(label, {})[n] = st
            c.rect(cx + 3, y + 3, CWd - 6, RH - 6, STATE[st], r=7)
            if st == "bit-identical":
                c.text(cx + CWd / 2, y + RH / 2 - 2, "=", size=36, weight="bold", ha="center")
    ly = GY + 4 * RH + 140
    c.swatch(GX, ly, ORANGE, "bit-identical to V3-Base", glyph="=", size=28)
    c.swatch(GX + 430, ly, BLUE, "updated", size=28)
    c.footer(REPO, size=22)
    c.save("fig2_deepseek_special_tokens", {
        "figure": "fig2_deepseek_special_tokens",
        "sources": ["results/" + src],
        "note": "Token names shortened from the tokenizer strings and written out (fim = fill-in, "
                "EOT = end of turn, pad = padding, sep = separator); grouping and column order are ours "
                "(tool sep is drawn with the tool-call tokens, as in the article's '5 tool-call tokens').",
        "rows": {label: key for label, key in rows},
        "column_labels": {n: shown.get(n, n) for _, n in cols},
        "values": grid,
    })


# --------------------------------------------------------------------------- fig 3


def fig3():
    src = "frontier/special_grid_nondeepseek.json"
    g = {r["release"]: r for r in load(src)}
    pick = [
        ("Qwen3-8B", "Qwen/Qwen3-8B"),
        ("Qwen3-14B", "Qwen/Qwen3-14B"),
        ("Qwen3-30B-A3B", "Qwen/Qwen3-30B-A3B"),
        ("Qwen3.5-35B-A3B", "Qwen/Qwen3.5-35B-A3B"),
        ("Llama-3.1-70B-Instruct", "unsloth/Meta-Llama-3.1-70B-Instruct"),
        ("Llama-4-Scout Instruct", "unsloth/Llama-4-Scout-17B-16E-Instruct"),
        ("Llama-4-Maverick Instruct", "unsloth/Llama-4-Maverick-17B-128E-Instruct"),
        ("Mistral-Small-24B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501"),
        ("Nemotron-3-Ultra", "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"),
        ("gpt-oss-safeguard-120b", "openai/gpt-oss-safeguard-120b"),
        ("Phi-4-reasoning", "microsoft/Phi-4-reasoning"),
        ("Olmo-3-7B-Instruct-SFT", "allenai/Olmo-3-7B-Instruct-SFT"),
        ("Kimi-K2-Thinking", "moonshotai/Kimi-K2-Thinking"),
    ]
    # the source's "vision / audio" class holds image, video and grounding tokens everywhere, and
    # audio tokens in one drawn release (Qwen3.5-35B-A3B)
    cats = [("chat markers", "chat"), ("tool calls", "tool\ncalls"), ("thinking", "thinking"),
            ("code fill-in", "code\nfill-in"), ("vision / audio", "image,\nvideo, audio"),
            ("other", "other and\nreserved")]
    order = ["updated", "barely changed", "bit-identical", "replaced"]
    RH, GY, GX, CWd = 66, 280, 480, 180
    Hh = GY + RH * len(pick) + 210
    c = Canvas(1600, Hh)
    c.title("Other labs: which kinds of special token each release updated",
            "Number of special tokens in each state, compared with the release's own base model.")
    for j, (_, lab) in enumerate(cats):
        c.text(GX + j * CWd + CWd / 2, GY - 50, lab, size=27, weight="bold", ha="center",
               linespacing=1.1)
    vals = {}
    for i, (label, rel) in enumerate(pick):
        y = GY + i * RH
        c.line(40, y, 1560, y, color=GRID, lw=1.2)
        c.text(40, y + RH / 2, label, size=28, weight="bold")
        cells = g[rel]["cells"]
        vals[label] = {"release": rel, "parent": g[rel]["parent"], "cells": cells}
        for j, (cat, _) in enumerate(cats):
            cell = cells.get(cat)
            cx = GX + j * CWd + CWd / 2
            if not cell:
                c.text(cx, y + RH / 2, "–", size=26, color="#B0B0B0", ha="center")
                continue
            items = [(s, cell[s]) for s in order if s in cell]
            assert len(items) == len(cell), cell
            PW, PG = 62, 5
            tot = len(items) * PW + (len(items) - 1) * PG
            x = cx - tot / 2
            for s, n in items:
                if s == "replaced":
                    c.rect(x, y + 11, PW, RH - 22, "#E4E4E4", r=7, hatch="///", ec="#9A9A9A", lw=0.8)
                    c.text(x + PW / 2, y + RH / 2, str(n), size=27, weight="bold", ha="center",
                           bbox=dict(fc="#E4E4E4", ec="none", pad=1.5))
                else:
                    c.rect(x, y + 11, PW, RH - 22, STATE[s], r=7)
                    c.text(x + PW / 2, y + RH / 2, str(n), size=27, weight="bold", ha="center",
                           color=STATE_TEXT[s])
                x += PW + PG
    yb = GY + RH * len(pick)
    c.line(40, yb, 1560, yb, color=GRID, lw=1.2)
    ly = yb + 50
    c.swatch(40, ly, BLUE, "updated", size=27)
    c.swatch(760, ly, ORANGE, "bit-identical to the base", size=27)
    c.swatch(40, ly + 56, YELLOW, "barely changed (1% of its numbers or fewer)", size=27)
    c.swatch(760, ly + 56, "#E4E4E4", "not testable (vectors were replaced)", hatch="///", size=27)
    c.footer(REPO, size=22)
    omitted = sorted(set(g) - {rel for _, rel in pick})
    c.save("fig3_special_tokens_other_labs", {
        "figure": "fig3_special_tokens_other_labs",
        "sources": ["results/" + src],
        "note": "13 of the 22 comparisons in the source are drawn. Omitted: sibling releases with the same "
                "pattern, and three the source marks testable=false (Kimi-K2-Instruct, GLM-4.5, GLM-4.5-Air). "
                "'replaced' in the source is drawn as 'not testable'. The source class 'vision / audio' is "
                "headed 'image, video, audio': it holds image, video and grounding tokens, plus audio "
                "tokens in Qwen3.5-35B-A3B only.",
        "omitted_releases": omitted,
        "values": vals,
    })


# --------------------------------------------------------------------------- fig 4


def fig4():
    src = "frontier/fig/a_change_histogram.json"
    d = load(src)
    order = ["DeepSeek-R1-Zero", "DeepSeek-R1", "DeepSeek-V3-0324"]
    states = ["bit-identical", "barely changed", "partly changed", "updated"]
    defs = {"bit-identical": "no number differs", "barely changed": "1% of its numbers or fewer",
            "partly changed": "1% to 10%", "updated": "more than 10%"}
    vocab = {d[k]["vocab"] for k in order}
    assert len(vocab) == 1
    vocab = vocab.pop()
    c = Canvas(1600, 900)
    c.title("DeepSeek: how much of each token vector changed",
            f"Each bar is all {vocab:,} vocabulary tokens of one release, compared with DeepSeek-V3-Base,",
            "grouped by the share of a token vector's numbers that differ.")
    lx = [40, 400, 860, 1230]
    for x, s in zip(lx, states):
        c.rect(x, 203, 30, 30, STATE[s], r=5)
        c.text(x + 42, 218, s, size=28, weight="bold")
        c.text(x + 42, 252, defs[s], size=25, color=INK2)
    BX, BW, BH = 250, 1310, 92
    ys = [366, 530, 690]
    for k, y in zip(order, ys):
        c.text(40, y + BH / 2, k.replace("DeepSeek-", ""), size=34, weight="bold")
        x = BX
        small = []
        for s in states:
            v = d[k][s]
            w = BW * v / 100
            c.rect(x + 1.5, y, max(w - 3, 1.5), BH, STATE[s])
            if v >= 5.5:
                c.text(x + w / 2, y + BH / 2, f"{v:.1f}%", size=31, weight="bold", ha="center",
                       color=STATE_TEXT[s])
            else:
                small.append(f"{s} {v:.1f}%")
            x += w
        if small:
            c.text(BX, y + BH + 30, "Too thin to label:  " + "   ·   ".join(small), size=25, color=INK2)
    # bracket: R1-Zero lightly changed
    k = "DeepSeek-R1-Zero"
    a = BX + BW * d[k]["bit-identical"] / 100
    b = a + BW * (d[k]["barely changed"] + d[k]["partly changed"]) / 100
    light = d[k]["barely changed"] + d[k]["partly changed"]
    yb = ys[0] - 16
    c.line(a + 3, yb, b - 3, yb, color=INK, lw=2.5)
    c.line(a + 3, yb, a + 3, yb + 9, color=INK, lw=2.5)
    c.line(b - 3, yb, b - 3, yb + 9, color=INK, lw=2.5)
    c.text((a + b) / 2, yb - 26, f"{light:.1f}% changed in under 10% of their numbers", size=28,
           weight="bold", ha="center")
    c.footer(REPO, size=22)
    c.save("fig4_change_share", {
        "figure": "fig4_change_share",
        "sources": ["results/" + src, "results/frontier/FRONTIER.md (state definitions; Figures note a)"],
        "values_percent_of_vocabulary": {k: {s: d[k][s] for s in states} for k in order},
        "vocab": vocab,
        "derived": {"R1-Zero barely + partly changed (%)": light},
    })


# --------------------------------------------------------------------------- fig 5


def fig6():
    import math
    src = "frontier/fig/c_decay_ranges.json"
    d = load(src)
    fams = ["Qwen", "Llama", "Mistral", "NVIDIA Nemotron"]
    groups = [(f, [r for r in d if r["family"] == f]) for f in fams]
    groups.append(("Other labs", [r for r in d if r["family"] not in fams]))
    for _, rs in groups:
        rs.sort(key=lambda r: r["shrink_pct_lo"])
    assert sum(len(rs) for _, rs in groups) == len(d)

    def name(r):
        n = r["child"].split("/")[1]
        for a in ("NVIDIA-", "Meta-", "-BF16"):
            n = n.replace(a, "")
        n = re.sub(r"-550B-A55B", "", n)
        return n

    RH, GY = 43, 250
    nrow = len(d)
    Hh = GY + nrow * RH + (len(groups) - 1) * 14 + 150
    c = Canvas(1600, Hh)
    c.title("How much weight decay shrank the never-updated token vectors",
            "Shrink of the vectors that post-training left untouched, release against its base model.",
            "A 16-bit release gives a range, not one value.")
    PX0, PX1, lo, hi = 640, 1330, 0.2, 8.0

    def X(v):
        return PX0 + (PX1 - PX0) * (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))

    ticks = [0.25, 0.5, 1, 2, 4, 8]
    ybot = GY + nrow * RH + (len(groups) - 1) * 14
    for t in ticks:
        c.line(X(t), GY - 8, X(t), ybot + 4, color=GRID, lw=1.2)
        c.text(X(t), GY - 28, f"{t:g}%", size=22, color=INK2, ha="center")
        c.text(X(t), ybot + 26, f"{t:g}%", size=22, color=INK2, ha="center")
    y = GY
    vals = []
    qwen3_rows = []
    for gi, (fam, rs) in enumerate(groups):
        c.text(40, y + RH / 2, fam.replace("NVIDIA Nemotron", "NVIDIA"), size=25, weight="bold")
        for r in rs:
            a, b = r["shrink_pct_lo"], r["shrink_pct_hi"]
            wide = (b - a) > 0.05
            col = ORANGE if r.get("one_16bit_step") else BLUE
            yc = y + RH / 2
            c.text(200, yc, name(r), size=23)
            if wide:
                c.rect(X(a), yc - 9, X(b) - X(a), 18, col, r=6, z=4)
                lab = f"{a:.2f}–{b:.2f}%"
            else:
                c.ax.plot([X((a + b) / 2)], [yc], "o", ms=13, color=col, zorder=4,
                          mec="white", mew=2)
                lab = f"{a:.2f}%" if f"{a:.2f}" == f"{b:.2f}" else f"{a:.2f}–{b:.2f}%"
            c.text(1548, yc, lab, size=23, ha="right", color=INK)
            vals.append({"child": r["child"], "parent": r["parent"], "family": r["family"],
                         "shrink_pct_lo": a, "shrink_pct_hi": b, "label": lab, "drawn_as_range": wide})
            if r["child"] in ("Qwen/Qwen3-8B", "Qwen/Qwen3-14B", "Qwen/Qwen3-30B-A3B"):
                qwen3_rows.append(yc)
            y += RH
        if gi < len(groups) - 1:
            c.line(40, y + 7, 1560, y + 7, color=GRID, lw=1.2)
            y += 14
    # Qwen3 note
    q = next(r for r in d if r["child"] == "Qwen/Qwen3-8B")
    xa = X(q["shrink_pct_hi"]) + 20
    c.line(xa, min(qwen3_rows) - 12, xa, max(qwen3_rows) + 12, color=INK, lw=2.5)
    c.text(xa + 12, sum(qwen3_rows) / 3, "same shrink\nat three sizes", size=22, weight="bold",
           linespacing=1.15)
    ly = ybot + 78
    c.swatch(40, ly, BLUE, "range from the 16-bit grid, widened for rounding drift")
    c.swatch(760, ly, ORANGE, "shrank by one smallest 16-bit step: a wide bound only")
    c.footer(f"Source: results/frontier/fig/c_decay_ranges.json · {REPO}")
    c.save("fig6_weight_decay_ranges", {
        "figure": "fig6_weight_decay_ranges",
        "sources": ["results/" + src, "results/roster/NOTES.md 'CORRECTED' (one smallest 16-bit step)"],
        "note": "x axis is logarithmic. Each bar is the range of shrink that reproduces the same vectors, "
                "widened by the largest simulated fp32 rounding drift (results/frontier/drift_sim.json). "
                "Orange: the vectors shrank by exactly one smallest 16-bit step (one_16bit_step in the source). "
                "Labels are the range ends rounded to two decimals.",
        "values": vals,
    })


# --------------------------------------------------------------------------- fig 6


def fig5():
    src = "frontier/fig/d_comparability.json"
    d = load(src)
    tot = d["totals"]
    # one highlight colour for the usable record; the other six groups are grey. Blue and orange are
    # not used here, because in figures 1 to 4 they mean "updated" and "bit-identical".
    GREY = "#A6A6A6"
    classes = [
        ("token record", GREEN, "Usable token record",
         "enough ordinary token vectors are bit-identical to the base to list the updated ones"),
        ("untrainable only", GREY, "Only unused vocabulary slots are bit-identical",
         "those tokens cannot occur in training text, so they say nothing about the data"),
        ("record, poor fit", GREY, "Partial record",
         "some vectors are bit-identical, but the weight-decay shrink fits poorly; treat with care"),
        ("changed throughout", GREY, "Every token vector changed",
         "the release follows its base, but no vector is left bit-identical to compare"),
        ("not comparable", GREY, "No match with that base",
         "the token vectors are not a shrunken copy of that base's (e.g. Kimi-K2.5)"),
        ("unchanged", GREY, "No token vector changed",
         "all are bit-identical to the base, so no token stands out"),
        ("shared matrix", GREY, "One matrix for input and output",
         "tied embeddings: every vector changes and no record is kept (e.g. Gemma)"),
    ]
    assert set(tot) == {k for k, *_ in classes}
    n = sum(tot.values())
    c = Canvas(1600, 920)
    c.title("Which releases carry a usable token record",
            f"{n} comparisons of a release with its base model, sorted into seven groups.")
    y0, RH = 160, 100
    NX, BX, BMAX, TX = 122, 144, 300, 476
    mx = max(tot.values())
    for i, (key, col, head, sub) in enumerate(classes):
        y = y0 + i * RH
        if i:
            c.line(40, y, 1560, y, color=GRID, lw=1.2)
        w = max(BMAX * tot[key] / mx, 6)
        c.text(NX, y + RH / 2, str(tot[key]), size=44, weight="bold", ha="right",
               color=INK if i else GREEN)
        c.rect(BX, y + RH / 2 - 22, w, 44, col, r=5)
        c.text(TX, y + 33, head, size=30, weight="bold")
        c.text(TX, y + 70, sub, size=25, color=INK2)
    c.footer(REPO, size=22)
    c.save("fig5_comparability", {
        "figure": "fig5_comparability",
        "sources": ["results/" + src, "results/frontier/FRONTIER.md §7 (class meanings, examples)"],
        "note": "Totals only. The by-family breakdown in the source is not drawn.",
        "totals": tot, "total": n,
        "labels": {k: h for k, _, h, _ in classes},
        "explanations": {k: s for k, _, _, s in classes},
    })


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for w in which:
        globals()["fig" + w]()
