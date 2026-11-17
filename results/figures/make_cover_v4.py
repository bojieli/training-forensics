#!/usr/bin/env python3
"""Article cover: the 11 neighbouring special tokens in DeepSeek V4's task-token block.

Run from the repository root:  python3 results/figures/make_cover_v4.py
Writes fig_cover_v4.{png,svg,json} and the 5:2 X-cover variant fig_cover_v4_wide.{png,svg,json}
next to this file. Same style as make_figures.py.

Every value drawn is read from results/frontier_ds/v4_task_tokens.json (share of numbers changed,
documented / undocumented grouping), results/frontier_ds/DEEPSEEK.md section 4 (the per-token table:
status, sizes, the third-party fine-tune) and the V4 tokenizer (token ids). The script asserts that
the JSON and the table agree before drawing.
"""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
DS = os.path.join(RES, "frontier_ds")
plt.rcParams["font.family"] = ["Lato", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "path"
BLUE, ORANGE = "#0072B2", "#E69F00"
INK, INK2, INK3, GRID, PALE = "#111111", "#444444", "#6B6B6B", "#D9D9D9", "#EFEFEF"
TINT = "#DCDCDC"

# ---------------------------------------------------------------- data
tk = json.load(open(os.path.join(DS, "tok", "DeepSeek-V4-Pro-Base__tokenizer.json")))
tkf = json.load(open(os.path.join(DS, "tok", "DeepSeek-V4-Flash-Base__tokenizer.json")))
d = json.load(open(os.path.join(DS, "v4_task_tokens.json")))
doc, undoc = set(d["documentation"]["documented"]), set(d["documentation"]["undocumented"])


def ids_of(t):
    out = {}
    for a in t["added_tokens"]:
        m = re.fullmatch("<｜([a-z_]+)｜>", a["content"])
        if m:
            out[m.group(1)] = a["id"]
    return out


ids = ids_of(tk)
assert ids == ids_of(tkf)
# the block: the contiguous id range from <|action|> to <|search|>
lo, hi = ids["action"], ids["search"]
block = sorted((n for n, i in ids.items() if lo <= i <= hi), key=lambda n: ids[n])
assert [ids[n] for n in block] == list(range(lo, hi + 1)) and len(block) == 11, block
assert set(block) <= doc | undoc
assert undoc <= set(block) and len(undoc) == 6
assert sorted(set(block) & doc) == ["action", "authority", "domain", "query", "title"]

cmpn = d["comparisons"]
flash = cmpn["DeepSeek-V4-Flash-Base -> DeepSeek-V4-Flash"]["tokens"]
pro = cmpn["DeepSeek-V4-Pro-Base -> DeepSeek-V4-Pro"]["tokens"]

# DEEPSEEK.md section 4 table
md = open(os.path.join(DS, "DEEPSEEK.md")).read()
sec4 = md[md.index("## 4. DeepSeek V4's named special tokens"):]
table = {}
for line in sec4.splitlines():
    c = [x.strip() for x in line.strip().strip("|").split("|")]
    m = re.fullmatch("`<｜([a-z_]+)｜>`", c[0]) if c else None
    if m and len(c) == 7:
        table[m.group(1)] = dict(group=c[1], flash=c[2], pro=c[3], nex=c[5])
CELL = re.compile(r"updated \((\d+)%, size ([\d.]+)→([\d.]+)\)( ×)?")
rows = []
for n in block:
    t = table[n]
    assert t["group"] == flash[n]["group"] == pro[n]["group"] == ("documented" if n in doc else "undocumented")
    mf, mp = CELL.fullmatch(t["flash"]), CELL.fullmatch(t["pro"])
    assert mf and mp, (n, t)
    pf, pp = round(flash[n]["frac_changed"] * 100), round(pro[n]["frac_changed"] * 100)
    assert pf == int(mf.group(1)) and pp == int(mp.group(1)), n
    assert t["nex"] == "bit-identical", n
    rows.append(dict(token=n, id=ids[n], group=t["group"],
                     flash_pct=pf, pro_pct=pp,
                     flash_size=[float(mf.group(2)), float(mf.group(3))],
                     pro_size=[float(mp.group(2)), float(mp.group(3))],
                     start_erased=bool(mf.group(4)) and bool(mp.group(4)),
                     flash_sim_to_base=flash[n]["sim_to_base"], pro_sim_to_base=pro[n]["sim_to_base"],
                     nex="bit-identical"))
erased = [r["token"] for r in rows if r["start_erased"]]
assert erased == ["action"], erased
rest = [r for r in rows if not r["start_erased"]]
sims = [r[k] for r in rest for k in ("flash_sim_to_base", "pro_sim_to_base")]
assert 0.98 <= min(sims) and max(sims) <= 1.0, sims          # "0.98-1.00 similarity to the base vector"
n_doc = sum(r["group"] == "documented" for r in rows)
n_undoc = sum(r["group"] == "undocumented" for r in rows)
assert (n_doc, n_undoc) == (5, 6)
assert all(r["flash_pct"] >= 93 and r["pro_pct"] >= 93 for r in rows)
act = rows[0]
assert act["token"] == "action"

# ---------------------------------------------------------------- drawing
W, H = 1600, 880
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")


def T(x, y, s, size=26, color=INK, weight="normal", ha="left"):
    return ax.text(x, y, s, fontsize=size * 0.72, color=color, fontweight=weight, ha=ha, va="center")


def R(x, y, w, h, fc, r=8, ec="none", lw=0, z=2):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r, boxstyle=f"round,pad={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=1))


M = 40
T(M, 58, f"DeepSeek V4 updated {n_undoc} special tokens its README doesn't describe", 46, weight="bold")
T(M, 116, f"The {len(rows)} neighbouring special tokens in the V4 tokenizer (ids {lo:,}–{hi:,}), "
          "in tokenizer order.", 28, INK2)

# legend for the two groups
y = 172
R(M, y - 17, 34, 34, "white", r=6, ec=INK3, lw=1.5)
T(M + 46, y, f"documented in DeepSeek's README ({n_doc})", 27, weight="bold")
R(700, y - 17, 34, 34, INK, r=6)
T(746, y, f"not in the README, changed the same way ({n_undoc})", 27, weight="bold")

GAP = 10
BW = (W - 2 * M - GAP * (len(rows) - 1)) / len(rows)
Y_NAME, H_NAME = 212, 72
BANDS = [  # (caption bold, caption rest, y of caption, key)
    ("V4-Flash", "compared with V4-Flash-Base: share of the vector's numbers that changed", 322, "flash_pct"),
    ("V4-Pro", "compared with V4-Pro-Base", 456, "pro_pct"),
    ("Nex-N2.5-Max", "a third-party fine-tune of the same V4-Pro-Base", 590, None),
]
H_CELL = 76
for i, r in enumerate(rows):
    x = M + i * (BW + GAP)
    und = r["group"] == "undocumented"
    if und:   # tinted band behind the whole column
        R(x - 5, Y_NAME - 5, BW + 10, (BANDS[-1][2] + 24 + H_CELL) - Y_NAME + 10, TINT, r=10, z=1)
    R(x, Y_NAME, BW, H_NAME, INK if und else "white", ec="none" if und else INK3, lw=0 if und else 1.5)
    T(x + BW / 2, Y_NAME + H_NAME / 2, r["token"], 27, "white" if und else INK, "bold", ha="center")
    for _, _, yc, key in BANDS:
        yb = yc + 24
        if key:
            R(x, yb, BW, H_CELL, BLUE)
            star = "*" if r["start_erased"] else ""
            T(x + BW / 2, yb + H_CELL / 2, f"{r[key]}%{star}", 36, "white", "bold", ha="center")
        else:
            R(x, yb, BW, H_CELL, ORANGE)
            T(x + BW / 2, yb + H_CELL / 2 - 2, "=", 44, INK, "bold", ha="center")
for bold, rest_, yc, _ in BANDS:
    t = T(M, yc, bold, 26, weight="bold")
    fig.canvas.draw()
    x1 = t.get_window_extent().x1
    T(x1 + 12, yc, rest_, 26, INK2)

y = 728
R(M, y - 17, 34, 34, BLUE, r=6)
T(M + 46, y, "updated from the base", 26, weight="bold")
R(420, y - 17, 34, 34, ORANGE, r=6)
T(437, y - 1, "=", 26, INK, "bold", ha="center")
T(466, y, "bit-identical to the base", 26, weight="bold")
T(M, 778, f"* action also grew in size ({act['flash_size'][0]:.2f} to {act['flash_size'][1]:.2f} in V4-Flash, "
          f"{act['pro_size'][0]:.2f} to {act['pro_size'][1]:.2f} in V4-Pro). The other ten kept 0.98–1.00 "
          "similarity to their base vectors.", 23, INK2)
T(M, 812, "The weights show that these vectors changed. They do not say what the tokens are for.", 23, INK2)
T(M, H - 26, "Method: @alapanchaudhuri  ·  github.com/bojieli/training-forensics", 22, INK3)

for ext in ("png", "svg"):
    fig.savefig(os.path.join(HERE, f"fig_cover_v4.{ext}"), dpi=100, facecolor="white")
side = {
    "figure": "fig_cover_v4",
    "size_px": [W, H],
    "sources": ["results/frontier_ds/v4_task_tokens.json (frac_changed, group, sim_to_base)",
                "results/frontier_ds/DEEPSEEK.md section 4 (per-token table: status, sizes, Nex-N2.5-Max column)",
                "results/frontier_ds/tok/DeepSeek-V4-{Pro,Flash}-Base__tokenizer.json (token ids)"],
    "note": "Percentages are the share of a vector's numbers that differ from the base, rounded as in DEEPSEEK.md. "
            "'documented' = described in DeepSeek's encoding README; read_url and extracted_url are documented "
            "too but sit outside this id range and are not drawn. action's start vector is gone from the release "
            "vector (the × in DEEPSEEK.md).",
    "tokens": rows,
}
json.dump(side, open(os.path.join(HERE, "fig_cover_v4.json"), "w"), indent=1, ensure_ascii=False)
# ---------------------------------------------------------------- wide variant (5:2, X article cover)
# Same data and assertions as above. X shows article covers at 5:2 and may crop a little more, so all
# text stays inside a 6% margin at the top and bottom. Dropped relative to the figure above: the
# subtitle, the colour key (the row captions carry it), and the action footnote with its asterisk.
WW, WH, WDPI = 1600, 640, 125          # layout units; saved at 2000x800 px
SAFE = 0.06 * WH
fig = plt.figure(figsize=(WW / 100, WH / 100), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WW); ax.set_ylim(WH, 0); ax.axis("off")
texts = []
texts.append(T(M, 70, f"DeepSeek V4 updated {n_undoc} special tokens its README doesn't describe", 46, weight="bold"))

y = 126
R(M, y - 17, 34, 34, "white", r=6, ec=INK3, lw=1.5)
texts.append(T(M + 46, y, f"documented in DeepSeek's README ({n_doc})", 27, weight="bold"))
R(700, y - 17, 34, 34, INK, r=6)
texts.append(T(746, y, f"not in the README, changed the same way ({n_undoc})", 27, weight="bold"))

BW = (WW - 2 * M - GAP * (len(rows) - 1)) / len(rows)
Y_NAME, H_NAME, H_CELL, CAP = 154, 64, 66, 18
assert all(r["nex"] == "bit-identical" for r in rows)
WBANDS = [  # (caption bold, caption rest, y of caption, key)
    ("V4-Flash", "compared with V4-Flash-Base: share of the vector's numbers that changed", 246, "flash_pct"),
    ("V4-Pro", "compared with V4-Pro-Base", 358, "pro_pct"),
    ("Nex-N2.5-Max", "a third-party fine-tune of the same V4-Pro-Base: bit-identical to the base", 470, None),
]
Y_END = WBANDS[-1][2] + CAP + H_CELL
for i, r in enumerate(rows):
    x = M + i * (BW + GAP)
    und = r["group"] == "undocumented"
    if und:
        R(x - 5, Y_NAME - 5, BW + 10, Y_END - Y_NAME + 10, TINT, r=10, z=1)
    R(x, Y_NAME, BW, H_NAME, INK if und else "white", ec="none" if und else INK3, lw=0 if und else 1.5)
    texts.append(T(x + BW / 2, Y_NAME + H_NAME / 2, r["token"], 27, "white" if und else INK, "bold", ha="center"))
    for _, _, yc, key in WBANDS:
        yb = yc + CAP
        if key:
            R(x, yb, BW, H_CELL, BLUE)
            texts.append(T(x + BW / 2, yb + H_CELL / 2, f"{r[key]}%", 36, "white", "bold", ha="center"))
        else:
            R(x, yb, BW, H_CELL, ORANGE)
            texts.append(T(x + BW / 2, yb + H_CELL / 2 - 2, "=", 44, INK, "bold", ha="center"))
for bold, rest_, yc, _ in WBANDS:
    t = T(M, yc, bold, 26, weight="bold")
    fig.canvas.draw()
    x1 = t.get_window_extent().x1
    texts += [t, T(x1 + 12, yc, rest_, 26, INK2)]

texts.append(T(M, 584, "The weights show that these vectors changed, not what the tokens are for.", 23, INK2))
texts.append(T(WW - M, 584, "Method: @alapanchaudhuri  ·  github.com/bojieli/training-forensics", 22, INK3, ha="right"))

fig.canvas.draw()
boxes = [t.get_window_extent() for t in texts]
assert min(b.y0 for b in boxes) >= SAFE and max(b.y1 for b in boxes) <= WH - SAFE, "text outside the 6% safe margin"
assert min(b.x0 for b in boxes) >= M - 1 and max(b.x1 for b in boxes) <= WW - M + 1, "text outside the side margin"
assert boxes[-2].x1 + 30 < boxes[-1].x0, "bottom line: caveat and credit collide"
for ext in ("png", "svg"):
    fig.savefig(os.path.join(HERE, f"fig_cover_v4_wide.{ext}"), dpi=WDPI if ext == "png" else 100, facecolor="white")
assert (WW * WDPI // 100) * 2 == (WH * WDPI // 100) * 5      # exactly 5:2
wide = dict(side, figure="fig_cover_v4_wide", size_px=[WW * WDPI // 100, WH * WDPI // 100],
            note=side["note"] + " Wide 5:2 variant for the X article cover: same tokens, percentages and "
                 "Nex-N2.5-Max row; the subtitle, colour key and action footnote (and its asterisk) are not drawn.")
json.dump(wide, open(os.path.join(HERE, "fig_cover_v4_wide.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps([(r["token"], r["id"], r["group"], r["flash_pct"], r["pro_pct"], r["nex"]) for r in rows]))
