"""
Supplementary Figure S4. Comparison of k = 5 and k = 8 topic size distributions.
DHJ-26-0922 (Digital Health) — figure generation script.

k = 8 is shown because it was the previously reported coherence maximum. On the
recomputed values (Supplemental Table S3) k = 8 has LOWER coherence than k = 5;
this panel is retained to document the two micro-topics it produces.

Fonts: Latin text is set in a metric-compatible Arial substitute so that the
figure matches Figure 1 and the manuscript body text; Chinese glyphs fall back
to a Simplified-Chinese face. Place a CJK font named NotoSansSC-Regular.otf next to this script
(or edit CJK_FONT below).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

os.makedirs("./output", exist_ok=True)

def draw_note(fig, x0, y0, text, width_frac, fontsize, color="#1a1a1a",
              linespacing=1.2, ha="left"):
    """Word-wrap a NOTE to `width_frac` of the figure width, measured with the
    renderer so mathtext spans are sized correctly, and draw it as a full-width
    left-justified block at the given line spacing. y0 is the block top."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    max_px = width_frac * fig.get_size_inches()[0] * fig.dpi
    def _w(s):
        t = fig.text(0, 0, s, fontsize=fontsize)
        wpx = t.get_window_extent(rend).width
        t.remove()
        return wpx
    lines, cur = [], ""
    for word in text.split():
        trial = word if not cur else cur + " " + word
        if _w(trial) <= max_px or not cur:
            cur = trial
        else:
            lines.append(cur); cur = word
    if cur:
        lines.append(cur)
    dy = fontsize * linespacing / (72 * fig.get_size_inches()[1])
    for i, ln in enumerate(lines):
        fig.text(x0, y0 - i * dy, ln, ha=ha, va="top", fontsize=fontsize,
                 color=color, linespacing=linespacing)
    return len(lines)


CJK_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "NotoSansSC-Regular.otf")
if os.path.exists(CJK_FONT):
    fm.fontManager.addfont(CJK_FONT)
plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", "Noto Sans CJK SC", "Noto Sans SC"],
    "pdf.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "axes.unicode_minus": False,
})

import numpy as np
from matplotlib.gridspec import GridSpec

# ══════════════════════════════════════════════════════════════
# UNIFIED COLOR PALETTE — Anchored to Figure 4
# ══════════════════════════════════════════════════════════════
TOPIC_COLORS_K5 = {
    'T1': '#1F77B4',   # Blue
    'T2': '#C67A2E',   # Orange
    'T3': '#2CA02C',   # Green
    'T4': '#9467BD',   # Purple  (was pink, now matches Fig 2/4)
    'T5': '#17BECF',   # Teal
}

# k=8 uses a muted palette to differentiate from k=5
TOPIC_COLORS_K8 = {
    'T1': '#D55E00',   # Vermillion
    'T2': '#56B4E9',   # Sky blue
    'T3': '#009E73',   # Bluish green
    'T4': '#E69F00',   # Amber
    'T5': '#0072B2',   # Dark blue
    'T6': '#CC79A7',   # Muted pink
    'T7*': '#882255',  # Wine (micro-topic)
    'T8*': '#AA4499',  # Plum (micro-topic)
}

# ── Topic sizes (Table S11) ──
k5_labels = ['T1', 'T2', 'T3', 'T4', 'T5']
k5_sizes  = [263, 206, 434, 552, 461]
k5_names  = ['Postoperative\nRecovery',
             'Postoperative Medication\nand Surveillance',
             'Living With\nThyroid Cancer',
             'Treatment Decision\nand Debate',
             'Healthcare\nNavigation']

k8_labels = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7*', 'T8*']
# k = 8 sizes recomputed from the analytic corpus with the same estimator
# (sklearn LDA, batch, max_iter=20, random_state=42); see 01_full_pipeline.py.
k8_sizes  = [501, 418, 336, 245, 184, 169, 42, 21]

# Verify totals
assert sum(k5_sizes) == 1916, f"k5 sum = {sum(k5_sizes)}, expected 1916"
assert sum(k8_sizes) == 1916, f"k8 sum = {sum(k8_sizes)}, expected 1916"

# ── Figure setup ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9.4), dpi=600,
                                gridspec_kw={'height_ratios': [1, 1.4],
                                             'hspace': 0.42})
# Fixed margins: reserve a bottom band for the wrapped note so it never
# collides with the 'Number of Posts' axis label, and enough left room for the
# k = 5 topic labels. The figure is saved on this fixed box (no bbox_inches),
# so the note coordinates below are exact.
fig.subplots_adjust(left=0.235, right=0.965, top=0.905, bottom=0.205)
fig.patch.set_facecolor('white')

fig.suptitle('Supplementary Figure S4. Comparison of $k$ = 5 and $k$ = 8\n'
             'Topic Size Distributions ($N$ = 1,916)',
             fontsize=11.5, fontweight='bold', y=0.98, va='top')

# ═══════════════════════════════════════
# Panel A: k = 5 (selected)
# ═══════════════════════════════════════
ax1.set_title('$k$ = 5 (selected)', fontsize=10.5, fontweight='bold',
              loc='left', pad=8, color='#1F77B4')

y_pos = np.arange(len(k5_labels))
bar_h = 0.55
colors_k5 = [TOPIC_COLORS_K5[t] for t in k5_labels]

bars1 = ax1.barh(y_pos, k5_sizes, height=bar_h, color=colors_k5,
                  edgecolor='white', linewidth=0.8, zorder=3)

# Y-axis: topic label + name
ytick_labels = [f'{lbl}: {nm}' for lbl, nm in zip(k5_labels, k5_names)]
ax1.set_yticks(y_pos)
ax1.set_yticklabels(ytick_labels, fontsize=8.5, fontweight='500', linespacing=1.1)
ax1.invert_yaxis()

# Value labels
for i, v in enumerate(k5_sizes):
    pct = v / 1916 * 100
    ax1.text(v + 12, i, f'{v}  ({pct:.1f}%)', va='center', fontsize=8,
             fontweight='bold', color='#333333')

ax1.set_xlim(0, 700)
ax1.set_xlabel('Number of Posts', fontsize=9, labelpad=6)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(0.6)
ax1.spines['bottom'].set_linewidth(0.6)
ax1.tick_params(axis='x', labelsize=8)
ax1.tick_params(axis='y', length=0)

# Light grid
ax1.xaxis.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
ax1.set_axisbelow(True)

# ═══════════════════════════════════════
# Panel B: k = 8 (rejected)
# ═══════════════════════════════════════
ax2.set_title('$k$ = 8 (rejected: lower coherence than $k$ = 5; two micro-topics)',
              fontsize=10.5, fontweight='bold', loc='left', pad=8, color='#D55E00')

y_pos8 = np.arange(len(k8_labels))
colors_k8 = [TOPIC_COLORS_K8[t] for t in k8_labels]

bars2 = ax2.barh(y_pos8, k8_sizes, height=bar_h, color=colors_k8,
                  edgecolor='white', linewidth=0.8, zorder=3)

# Hatch micro-topics
for i, lbl in enumerate(k8_labels):
    if '*' in lbl:
        bars2[i].set_hatch('///')
        bars2[i].set_edgecolor(TOPIC_COLORS_K8[lbl])
        bars2[i].set_linewidth(0.5)

ax2.set_yticks(y_pos8)
ax2.set_yticklabels(k8_labels, fontsize=8.5, fontweight='bold')
ax2.invert_yaxis()

for i, v in enumerate(k8_sizes):
    pct = v / 1916 * 100
    ax2.text(v + 12, i, f'{v}  ({pct:.1f}%)', va='center', fontsize=8,
             fontweight='bold', color='#333333')

ax2.set_xlim(0, 700)
ax2.set_xlabel('Number of Posts', fontsize=9, labelpad=6)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.6)
ax2.spines['bottom'].set_linewidth(0.6)
ax2.tick_params(axis='x', labelsize=8)
ax2.tick_params(axis='y', length=0)

ax2.xaxis.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)

# Dashed line separating micro-topics
micro_start = k8_labels.index('T7*') - 0.5
ax2.axhline(y=micro_start, color='#D55E00', linestyle='--', linewidth=1.2, alpha=0.7, zorder=4)

# Micro-topic annotation
ax2.annotate('Micro-topics ($n$ < 50)',
             xy=(250, micro_start + 1.0),
             fontsize=8.5, color='#D55E00', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF3E8',
                       edgecolor='#D55E00', alpha=0.9, linewidth=0.8))

# ── Note — auto-wrapped to the figure width, line spacing 1.2 ─────────────────
NOTE = ('Note. Coherence is essentially flat between $k$ = 5 and $k$ = 7 ($c_v$ = 0.4995, 0.5069 and 0.5079; Supplemental Table S3), so coherence does not discriminate among those solutions. $k$ = 5 was selected because $k$ = 6 and $k$ = 7 subdivide topics already present at $k$ = 5 rather than identifying additional content, and because $k$ = 5 yields no micro-topics. $k$ = 8 has lower coherence than $k$ = 5 ($c_v$ = 0.4797) and fragments into two micro-topics ($n$ = 42 and $n$ = 21; lower two bars). Both solutions via sklearn LDA (batch, max_iter = 20, random_state = 42). Topic label colors in the $k$ = 5 panel match Figures 2–4. See Methods.')
draw_note(fig, 0.045, 0.140, NOTE, 0.92, 8.0, color="#555555")

plt.savefig('./output/FigS4.png', dpi=600, facecolor='white', edgecolor='none')
plt.savefig('./output/FigS4.pdf', facecolor='white', edgecolor='none')
plt.close()
print("Figure S4 saved (PNG + PDF).")
print(f"  k=5 sizes: {k5_sizes} (sum={sum(k5_sizes)})")
print(f"  k=8 largest: {max(k8_sizes)} ({max(k8_sizes)/1916*100:.1f}%)")
