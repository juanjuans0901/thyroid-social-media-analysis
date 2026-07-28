"""
Figure 3. Cross-platform topic distribution heatmap (N = 1,916 posts).
DHJ-26-0922 (Digital Health) — figure generation script.

Row percentages of the platform x topic cross-tabulation (Supplemental Table
S8). The plotted values are the published values; they are regenerated from the
analytic corpus by 01_full_pipeline.py, which writes
output/table_s8_platform_topic.csv on every run.

Fonts: this figure contains no Chinese glyphs. Latin text is set in a
metric-compatible Arial substitute so that the figure matches Figure 1 and the
manuscript body text. pdf.fonttype = 42 embeds every face in the exported PDF;
verify with `pdffonts` (every listed font must show emb = yes).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("./output", exist_ok=True)

plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "axes.unicode_minus": False,
})

# ── Data: counts from 01_full_pipeline.py (output/table_s8_platform_topic.csv) ──
PLATFORMS = ["Xiaohongshu", "Weibo", "Zhihu", "Douyin"]
TOPICS = ["T1", "T2", "T3", "T4", "T5"]
COUNTS = np.array([[119, 75, 186, 161, 171],    # Xiaohongshu, n = 712
                   [28, 43, 87, 150, 71],       # Weibo,       n = 379
                   [91, 54, 121, 185, 193],     # Zhihu,       n = 644
                   [25, 34, 40, 56, 26]])       # Douyin,      n = 181

TOTALS = COUNTS.sum(axis=1)
assert list(TOTALS) == [712, 379, 644, 181], TOTALS
assert list(COUNTS.sum(axis=0)) == [263, 206, 434, 552, 461], COUNTS.sum(axis=0)
assert COUNTS.sum() == 1916
PCT = COUNTS / TOTALS[:, None] * 100

# ── Unified palette, anchored to Figures 2, 4, S4 and S6 ──
TCOL = ["#1F77B4", "#C67A2E", "#2CA02C", "#9467BD", "#17BECF"]
TNAME = ["T1: Postoperative\nRecovery", "T2: Medication &\nSurveillance",
         "T3: Living With\nThyroid Cancer", "T4: Treatment\nDecision",
         "T5: Healthcare\nNavigation"]

VMIN, VCENTER, VMAX = 0, 20, 45   # centred on the 20% expected under k = 5

fig = plt.figure(figsize=(11.5, 5.1), facecolor="white")
ax = fig.add_axes([0.115, 0.235, 0.775, 0.630])
cax = fig.add_axes([0.905, 0.235, 0.013, 0.630])

norm = matplotlib.colors.TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX)
cmap = plt.get_cmap("RdBu_r")
im = ax.imshow(PCT, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks([])
ax.set_yticks([])
ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=3.0)
ax.tick_params(which="both", length=0)
for s in ax.spines.values():
    s.set_visible(False)

for r in range(4):
    for c in range(5):
        rgb = np.array(cmap(norm(PCT[r, c]))[:3])
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        ax.text(c, r, f"{PCT[r, c]:.1f}%", ha="center", va="center",
                fontsize=12.5, fontweight="bold",
                color="white" if lum < 0.52 else "#1a1a1a")

# Row labels: platform name over an italic n
for r, (plat, n) in enumerate(zip(PLATFORMS, TOTALS)):
    ax.text(-0.62, r - 0.13, plat, ha="right", va="center",
            fontsize=11.5, fontweight="bold", color="#1a1a1a")
    ax.text(-0.62, r + 0.17, f"($\\it{{n}}$ = {n:,})", ha="right", va="center",
            fontsize=10.5, color="#1a1a1a")

# Column labels below the map, coloured to match Figures 2 and 4
for c, (name, col) in enumerate(zip(TNAME, TCOL)):
    ax.text(c, 3.62, name, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=col, linespacing=1.35)
ax.set_ylim(3.5, -0.5)

cb = fig.colorbar(im, cax=cax, ticks=np.arange(0, 46, 5))
cb.outline.set_linewidth(0.6)
cb.outline.set_edgecolor("#888888")
cb.set_label("% of platform posts", fontsize=10.5, labelpad=8)
cb.ax.tick_params(labelsize=9.5, length=3)
cb.ax.axhline(VCENTER, color="#3a3a3a", linestyle="--", linewidth=1.1)

fig.text(0.5, 0.945,
         "Figure 3. Cross-Platform Topic Distribution Heatmap "
         "($\\it{N}$ = 1,916)",
         ha="center", va="center", fontsize=14, fontweight="bold")

# ── Note ─────────────────────────────────────────────────────────────────────
# Split by hand into two lines at a sentence boundary, so that no test
# statistic, effect size or P value can be broken across a line end. The
# submitted version of this figure wrapped "Cramer\'s V = 0.121" after the
# decimal point. Statistical symbols are set in italics via mathtext.
NOTE_LINES = [
    "Note. Row percentages sum to 100% within each platform. Color scale "
    "centered at 20%, the uniform distribution expected for $\\it{k}$ = 5 topics.",
    "Overall $\\chi^{2}$(12) = 84.16, $\\it{P}$ < .001, Cramér\'s $\\it{V}$ = 0.121. "
    "See Supplemental Table S8 for pairwise comparisons. Topic label colors "
    "match Figure 4.",
]
for i, line in enumerate(NOTE_LINES):
    fig.text(0.075, 0.078 - i * 0.040, line, ha="left", va="top",
             fontsize=9.2, color="#1a1a1a")

fig.savefig("./output/Fig3.pdf", facecolor="white")
fig.savefig("./output/Fig3.png", dpi=600, facecolor="white")
plt.close()
print("Figure 3 saved (PDF + PNG).")
print("  row totals:", list(TOTALS), " column totals:", list(COUNTS.sum(axis=0)))
print("Reminder: verify the exported PDF with `pdffonts` — every font listed "
      "must show emb = yes.")
