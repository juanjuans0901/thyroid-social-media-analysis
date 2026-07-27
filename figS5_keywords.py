"""
Supplementary Figure S5. Top 10 keyword weights per LDA topic (N = 1,916).
DHJ-26-0922 (Digital Health) — figure generation script.

This is the script that produced the figure submitted with the revision. The
plotted values are the published values; they are regenerated from the analytic
corpus by 01_full_pipeline.py.

Fonts: Latin text is set in a metric-compatible Arial substitute so that the
figure matches Figure 1 and the manuscript body text; Chinese glyphs fall back
to a Simplified-Chinese face. Place a CJK font named NotoSansSC-Regular.otf next to this script
(or edit CJK_FONT below).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

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
os.makedirs("./output", exist_ok=True)

from matplotlib.gridspec import GridSpec

D={
'T1: Postoperative Recovery':('#1f77b4',[('术后 (Postop.)',0.0283),('身体 (Body)',0.0119),('甲状腺癌 (Thyroid ca.)',0.0104),
 ('甲状腺 (Thyroid)',0.0097),('复查 (Follow-up)',0.0093),('生活 (Life)',0.0082),('饮食 (Diet)',0.0080),
 ('大家 (Everyone)',0.0074),('恢复 (Recovery)',0.0070),('健康 (Health)',0.0067)]),
'T2: Cancer Progression':('#c67a2e',[('甲状腺癌 (Thyroid ca.)',0.0201),('治疗 (Treatment)',0.0165),('转移 (Metastasis)',0.0152),
 ('肿瘤 (Tumor)',0.0147),('观察 (Observation)',0.0131),('患者 (Patient)',0.0131),('淋巴结 (Lymph node)',0.0101),
 ('发现 (Discovery)',0.0081),('复发 (Recurrence)',0.0081),('乳头状癌 (Papillary ca.)',0.0077)]),
'T3: Diagnostic Evaluation':('#2ca02c',[('结节 (Nodule)',0.0358),('甲状腺结节 (Thyroid nodule)',0.0205),('恶性 (Malignant)',0.0115),
 ('治疗 (Treatment)',0.0113),('需要 (Need)',0.0112),('良性 (Benign)',0.0097),('出现 (Occurrence)',0.0072),
 ('甲状腺癌 (Thyroid ca.)',0.0068),('症状 (Symptom)',0.0068),('患者 (Patient)',0.0067)]),
'T4: Treatment Decision':('#9467bd',[('医生 (Doctor)',0.0210),('感觉 (Feeling)',0.0084),('术后 (Postop.)',0.0083),
 ('医院 (Hospital)',0.0078),('结果 (Result)',0.0063),('开始 (Start)',0.0062),('脖子 (Neck)',0.0060),
 ('检查 (Exam)',0.0058),('不能 (Cannot)',0.0055),('一直 (Always)',0.0055)]),
'T5: Healthcare Navigation':('#17becf',[('消融 (Ablation)',0.0297),('医生 (Doctor)',0.0273),('结节 (Nodule)',0.0185),
 ('穿刺 (FNA)',0.0164),('医院 (Hospital)',0.0156),('建议 (Advice)',0.0115),('复查 (Follow-up)',0.0115),
 ('结果 (Result)',0.0085),('检查 (Exam)',0.0080),('观察 (Observation)',0.0075)]),
}
fig=plt.figure(figsize=(15.90,11.07))
gs=GridSpec(2,6,figure=fig,left=0.130,right=0.980,top=0.885,bottom=0.115,hspace=0.42,wspace=2.9)
slots=[gs[0,0:2],gs[0,2:4],gs[0,4:6],gs[1,1:3],gs[1,3:5]]
for slot,(title,(col,items)) in zip(slots,D.items()):
    ax=fig.add_subplot(slot)
    labs=[i[0] for i in items][::-1]; vals=[i[1] for i in items][::-1]
    y=np.arange(len(vals))
    ax.barh(y,vals,color=col,height=0.68)
    ax.set_yticks(y); ax.set_yticklabels(labs,fontsize=11.5)
    ax.set_xlim(0,max(vals)*1.30)
    ax.set_xlabel('Weight',fontsize=12)
    ax.set_title(title,fontsize=13.5,color=col,pad=10)
    ax.tick_params(axis='x',labelsize=10.5)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#999999'); ax.spines['bottom'].set_color('#999999')
    for yi,v in zip(y,vals):
        ax.text(v+max(vals)*0.025,yi,f'{v:.4f}',va='center',ha='left',fontsize=10.5,color=col)
fig.text(0.5,0.945,'Supplementary Figure S5. Top 10 Keyword Weights per LDA Topic ($N$ = 1,916)',
         ha='center',va='center',fontsize=16)
note=('Note. Keywords are shown as Chinese term (English translation). Weights are from the LDA topic–word distribution ($k$ = 5, $c_v$ = 0.4802).\n'
      'Color scheme is consistent with Figures 2–4. See Methods.')
fig.text(0.085,0.045,note,ha='left',va='center',fontsize=11.5,linespacing=1.75,color='#1a1a1a')
fig.savefig('./output/FigS5.pdf')
fig.savefig('./output/FigS5.png', dpi=600)
print('Figure S5 saved (PDF + PNG).')
print('S5 built')
