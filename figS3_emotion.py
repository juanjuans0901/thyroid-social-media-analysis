"""
Supplementary Figure S3. Emotion category intensity by topic (posts, N = 1,916).
DHJ-26-0922 (Digital Health) — figure generation script.

This is the script that produced the figure submitted with the revision. The
plotted values are the published values; they are regenerated from the analytic
corpus by 01_full_pipeline.py.

Fonts: Latin text is set in a metric-compatible Arial substitute so that the
figure matches Figure 1 and the manuscript body text; Chinese glyphs are set in
a Simplified-Chinese face. A font named NotoSansSC-Regular.otf MUST be present
next to this script — the script stops with an error if it is not, because a
silent fallback produces a figure with the Chinese characters missing. All
fonts are embedded (pdf.fonttype = 42); verify with `pdffonts`.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

# ── Fonts ─────────────────────────────────────────────────────────────────────
# This figure contains Chinese glyphs. The submitted version of it lost those
# glyphs because the export fell back to a face with no CJK coverage and no
# fonts were embedded, so the failure is made LOUD here rather than silent.
# Per-glyph font fallback across a family list requires matplotlib >= 3.6
# (pinned in requirements.txt); pdf.fonttype = 42 embeds the faces in the PDF.
CJK_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "NotoSansSC-Regular.otf")
if not os.path.exists(CJK_FONT):
    raise SystemExit(
        "Missing Simplified-Chinese font.\n"
        "  Expected: " + CJK_FONT + "\n"
        "  Place a font file named NotoSansSC-Regular.otf next to this script.\n"
        "  Noto Sans SC (SIL Open Font License) works and is available from\n"
        "  https://fonts.google.com/noto/specimen/Noto+Sans+SC\n"
        "  The script stops rather than falling back, because a silent fallback\n"
        "  produces a figure with the Chinese characters missing.")
if tuple(int(x) for x in matplotlib.__version__.split(".")[:2]) < (3, 6):
    raise SystemExit("matplotlib >= 3.6 is required for per-glyph font fallback; "
                     "found " + matplotlib.__version__)
fm.fontManager.addfont(CJK_FONT)
_CJK_NAME = fm.FontProperties(fname=CJK_FONT).get_name()
plt.rcParams.update({
    "font.family": ["Liberation Sans", "Arial", _CJK_NAME,
                    "Noto Sans CJK SC", "Noto Sans SC"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "axes.unicode_minus": False,
})
os.makedirs("./output", exist_ok=True)

topics=['T1: Postoperative Recovery','T2: Postoperative Medication and Surveillance',
        'T3: Living With Thyroid Cancer','T4: Treatment Decision and Debate',
        'T5: Healthcare Navigation']
tcol=['#1f77b4','#c67a2e','#2ca02c','#9467bd','#17becf']
en=['Fear','Sadness','Anger','Disgust','Surprise','Joy','Like']
cn=['惧','哀','怒','恶','惊','乐','好']
V=np.array([[2.52,3.02,0.35,0.00,0.08,2.86,1.44],
            [1.10,0.30,0.12,0.00,0.00,0.78,0.57],
            [1.53,1.40,0.42,0.02,0.05,1.29,1.05],
            [0.93,0.22,0.07,0.00,0.02,0.44,0.63],
            [1.68,0.79,0.22,0.00,0.04,1.59,1.47]])
VMAX=3.1
cmap=plt.get_cmap('Blues')

fig=plt.figure(figsize=(14.09,6.91))
ax=fig.add_axes([0.310,0.300,0.565,0.560])
im=ax.imshow(V,cmap=cmap,vmin=0,vmax=VMAX,aspect='auto')
ax.set_xticks([]); ax.set_yticks([])
ax.set_xticks(np.arange(-.5,7,1),minor=True); ax.set_yticks(np.arange(-.5,5,1),minor=True)
ax.grid(which='minor',color='white',linewidth=1.6)
ax.tick_params(which='both',length=0)
for s in ax.spines.values(): s.set_visible(False)
for r in range(5):
    for c in range(7):
        rgb=np.array(cmap(V[r,c]/VMAX)[:3])
        lum=0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]
        ax.text(c,r,f'{V[r,c]:.2f}',ha='center',va='center',fontsize=14.5,
                color='white' if lum<0.55 else '#1a1a1a')
for r,(t,c) in enumerate(zip(topics,tcol)):
    ax.text(-0.60,r,t,ha='right',va='center',fontsize=12.5,color=c)
for c,(e,z) in enumerate(zip(en,cn)):
    ax.text(c,4.60,e,ha='center',va='top',fontsize=14,color='#1a1a1a')
    ax.text(c,4.86,z,ha='center',va='top',fontsize=14,color='#1a1a1a')
ax.axvline(4.5,color='#333333',linewidth=2.4)
ax.set_ylim(4.5,-0.5)

# group brackets under the columns
def bracket(x0,x1,label,colr):
    y=5.30
    ax.plot([x0,x1],[y,y],color=colr,lw=1.5,clip_on=False)
    for xx in (x0,x1): ax.plot([xx,xx],[y-0.07,y],color=colr,lw=1.5,clip_on=False)
    ax.text((x0+x1)/2,y+0.14,label,ha='center',va='top',fontsize=13,color=colr,clip_on=False)
bracket(-0.42,4.42,'Negative emotions','#b5502f')
bracket(4.58,6.42,'Positive emotions','#2b6ca3')

cax=fig.add_axes([0.892,0.300,0.014,0.560])
cb=fig.colorbar(im,cax=cax,ticks=np.arange(0,3.1,0.5))
cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor('#888888')
cb.set_label('Mean weighted emotion score per post',fontsize=12,labelpad=9)
cb.ax.tick_params(labelsize=11.5,length=3)

fig.text(0.5,0.945,'Supplementary Figure S3. Emotion Category Intensity by Topic (Posts, $N$ = 1,916)',
         ha='center',va='center',fontsize=15.5)
note=('Note. Cells show the mean weighted emotion score per post within each topic. Weights are from the domain-specific sentiment lexicon (Table S4, 80 terms). Negative\n'
      'emotions: Fear (惧), Sadness (哀), Anger (怒), Disgust (恶), Surprise (惊). Positive emotions: Joy (乐), Like (好). See Methods. Distributions are right-skewed; full descriptive\n'
      'statistics including medians and IQRs are reported in Table S12.')
fig.text(0.085,0.093,note,ha='left',va='center',fontsize=11,linespacing=1.75,color='#1a1a1a')
fig.savefig('./output/FigS3.pdf')
fig.savefig('./output/FigS3.png', dpi=600)
print('Figure S3 saved (PDF + PNG).')
print('S3 built')
print("Reminder: verify the exported PDF with `pdffonts` — every font listed "
      "must show emb = yes.")
