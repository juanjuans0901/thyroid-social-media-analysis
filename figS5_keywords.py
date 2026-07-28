"""
Supplementary Figure S5. Top 10 keyword weights per LDA topic (N = 1,916).
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

def draw_note(fig, x0, y0, text, width_frac, fontsize, color="#1a1a1a",
              linespacing=1.2, ha="left"):
    """Word-wrap a NOTE to `width_frac` of the figure width, measured with the
    renderer so mathtext spans are sized correctly, and draw it as a full-width
    left-justified block at the given line spacing. Lines are packed as full as
    they fit, so there are no short/long alternating lines. y0 is the block top."""
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


from matplotlib.gridspec import GridSpec

# Topic-word distributions from the fitted model (k = 5, batch variational
# inference, max_iter = 20, random_state = 42). These values are written to
# output/topic_keywords_top20.csv by 01_full_pipeline.py; the top ten per topic
# are reproduced here so that this script runs standalone.
D={
'T1: Postoperative Recovery':('#1f77b4',[('医生 (Doctor)',0.0209),('术后 (Postop.)',0.0127),('感觉 (Feeling)',0.0107),
 ('脖子 (Neck)',0.0094),('护士 (Nurse)',0.0078),('开始 (Start)',0.0078),('小时 (Hour)',0.0076),
 ('出院 (Discharge)',0.0075),('不能 (Cannot)',0.0071),('伤口 (Wound)',0.0066)]),
'T2: Postoperative Medication and Surveillance':('#c67a2e',[('术后 (Postop.)',0.0246),('身体 (Body)',0.0079),
 ('恢复 (Recovery)',0.0076),('甲状腺癌 (Thyroid ca.)',0.0072),('饮食 (Diet)',0.0071),
 ('复查 (Follow-up)',0.0068),('TSH',0.0067),('需要 (Need)',0.0061),('影响 (Effect)',0.0061),
 ('神经 (Nerve)',0.0057)]),
'T3: Living With Thyroid Cancer':('#2ca02c',[('术后 (Postop.)',0.0086),('甲状腺癌 (Thyroid ca.)',0.0080),
 ('甲癌 (Thyroid ca., colloq.)',0.0078),('生活 (Life)',0.0077),('身体 (Body)',0.0069),
 ('觉得 (Feel)',0.0060),('大家 (Everyone)',0.0059),('复查 (Follow-up)',0.0059),
 ('医生 (Doctor)',0.0055),('很多 (Many)',0.0053)]),
'T4: Treatment Decision and Debate':('#9467bd',[('结节 (Nodule)',0.0231),('治疗 (Treatment)',0.0142),
 ('甲状腺癌 (Thyroid ca.)',0.0132),('甲状腺结节 (Thyroid nodule)',0.0114),
 ('患者 (Patient)',0.0102),('恶性 (Malignant)',0.0096),('观察 (Observation)',0.0092),
 ('消融 (Ablation)',0.0089),('肿瘤 (Tumor)',0.0087),('需要 (Need)',0.0084)]),
'T5: Healthcare Navigation':('#17becf',[('医生 (Doctor)',0.0345),('医院 (Hospital)',0.0199),
 ('穿刺 (FNA)',0.0181),('消融 (Ablation)',0.0174),('结节 (Nodule)',0.0146),
 ('复查 (Follow-up)',0.0128),('结果 (Result)',0.0123),('检查 (Exam)',0.0123),
 ('建议 (Advice)',0.0103),('体检 (Health check)',0.0078)]),
}
fig=plt.figure(figsize=(15.90,11.07))
gs=GridSpec(2,6,figure=fig,left=0.130,right=0.980,top=0.885,bottom=0.165,hspace=0.42,wspace=2.9)
slots=[gs[0,0:2],gs[0,2:4],gs[0,4:6],gs[1,1:3],gs[1,3:5]]
for slot,(title,(col,items)) in zip(slots,D.items()):
    ax=fig.add_subplot(slot)
    labs=[i[0] for i in items][::-1]; vals=[i[1] for i in items][::-1]
    y=np.arange(len(vals))
    ax.barh(y,vals,color=col,height=0.68)
    ax.set_yticks(y); ax.set_yticklabels(labs,fontsize=11.5)
    ax.set_xlim(0,max(vals)*1.30)
    ax.set_xlabel('Weight',fontsize=12)
    ax.set_title(title,fontsize=11.5,color=col,pad=10)
    ax.tick_params(axis='x',labelsize=10.5)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#999999'); ax.spines['bottom'].set_color('#999999')
    for yi,v in zip(y,vals):
        ax.text(v+max(vals)*0.025,yi,f'{v:.4f}',va='center',ha='left',fontsize=10.5,color=col)
fig.text(0.5,0.945,'Supplementary Figure S5. Top 10 Keyword Weights per LDA Topic ($N$ = 1,916)',
         ha='center',va='center',fontsize=16,fontweight='bold')
NOTE = ('Note. Keywords are shown as Chinese term (English translation). Weights are from the LDA topic–word distribution ($k$ = 5, batch variational inference, random_state = 42). Color scheme is consistent with Figures 2–4. See Methods.')

draw_note(fig, 0.085, 0.090, NOTE, 0.86, 11.5, color="#1a1a1a")
fig.savefig('./output/FigS5.pdf')
fig.savefig('./output/FigS5.png', dpi=600)
print('Figure S5 saved (PDF + PNG).')
print('S5 built')
print("Reminder: verify the exported PDF with `pdffonts` — every font listed "
      "must show emb = yes.")
