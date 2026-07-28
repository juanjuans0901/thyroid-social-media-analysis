"""
Figure 2. Topic distribution across 1,916 patient-authored posts.
DHJ-26-0922 (Digital Health) — figure generation script.

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


# ── Data (from manuscript) ──
topics = [
    'T1: Postoperative Recovery',
    'T2: Postoperative Medication\n      and Surveillance',
    'T3: Living With Thyroid Cancer',
    'T4: Treatment Decision and Debate',
    'T5: Healthcare Navigation',
]
pcts = [13.7, 10.8, 22.7, 28.8, 24.1]
ns = [263, 206, 434, 552, 461]

# Colors matching the unified palette
colors = ['#1F77B4', '#C67A2E', '#2CA02C', '#9467BD', '#17BECF']

# ── Figure ──
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=600, facecolor='white')

y_pos = np.arange(len(topics))

# 20% uniform baseline
ax.axvline(x=20, color='#CCCCCC', linewidth=1.2, linestyle='--', zorder=1)
ax.text(21, 0.5, '20% uniform', ha='left', va='center',
        fontsize=8.5, color='#AAAAAA', fontstyle='italic')

# Lollipop chart: horizontal lines + dots
for i, (topic, pct, n, color) in enumerate(zip(topics, pcts, ns, colors)):
    # Line
    ax.plot([0, pct], [i, i], color=color, linewidth=3.5, solid_capstyle='round', zorder=3)
    # Dot
    ax.plot(pct, i, 'o', color=color, markersize=12, zorder=4)
    # Label: percentage and n
    ax.text(pct + 0.8, i, f'{pct}%  (n = {n})', va='center', ha='left',
            fontsize=11.5, fontweight='bold', color=color)

# Y-axis
ax.set_yticks(y_pos)
ax.set_yticklabels(topics, fontsize=12, fontweight='bold')
for i, tick in enumerate(ax.get_yticklabels()):
    tick.set_color(colors[i])

# X-axis
ax.set_xlim(0, 35)
ax.set_xlabel('Percentage of Posts (%)', fontsize=11, labelpad=8)
ax.set_xticks(np.arange(0, 36, 5))
ax.tick_params(axis='x', labelsize=10)

# Spine styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.8)
ax.tick_params(axis='y', length=0)
ax.invert_yaxis()

# Title
ax.set_title('Figure 2. Topic Distribution (N = 1,916 Posts)',
             fontsize=14, fontweight='bold', pad=14)

plt.tight_layout()

plt.savefig('./output/Fig2.png', dpi=600, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.savefig('./output/Fig2.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.15)
plt.close()
print('Figure 2 saved (PNG + PDF).')
