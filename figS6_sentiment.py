"""
Supplementary Figure S6. Sentiment distribution by platform and topic.
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

plats=['Xiaohongshu','Zhihu','Weibo','Douyin']
posts={'Xiaohongshu':(21.3,58.1,20.5),'Zhihu':(15.7,55.6,28.7),'Weibo':(23.2,58.0,18.7),'Douyin':(11.6,59.7,28.7)}
cmts ={'Xiaohongshu':(10.8,83.3,5.9),'Zhihu':(9.5,79.9,10.6),'Weibo':(3.9,81.1,15.0),'Douyin':(8.9,82.3,8.8)}
cn   ={'Xiaohongshu':4332,'Zhihu':10425,'Weibo':2161,'Douyin':9325}
NEG,NEU,POS='#d95f02','#bdbdbd','#3a8fc7'

tp=['T1\nPostoperative\nRecovery','T2\nMedication &\nSurveillance','T3\nLiving With\nThyroid Cancer','T4\nTreatment\nDecision','T5\nHealthcare\nNavigation']
tcol=['#1f77b4','#c67a2e','#2ca02c','#9467bd','#17becf']
pB=np.array([[18.5,17.3,32.3,11.2,22.8],[28.6,11.1,16.5,9.7,16.1],[28.6,30.2,27.6,18.7,21.1],[20.0,8.8,15.0,3.6,19.2]])
pN=np.array([[119,75,186,161,171],[91,54,121,185,193],[28,43,87,150,71],[25,34,40,56,26]])
cB=np.array([[11.6,20.8,10.7,8.5,12.0],[12.3,7.9,10.5,7.8,10.6],[0.0,2.1,3.6,5.8,6.9],[10.7,11.7,8.1,7.6,12.3]])
cN=np.array([[215,106,614,632,607],[1783,302,1430,1325,2686],[49,94,253,291,175],[215,307,577,459,260]])

fig=plt.figure(figsize=(14.31,12.14))
fig.text(0.5,0.968,'Supplementary Figure S6. Sentiment Distribution by Platform and Topic',
         ha='center',va='center',fontsize=16.5)
fig.text(0.055,0.925,'(A)   Overall Sentiment Distribution by Platform',ha='left',va='center',fontsize=14.5,weight='bold')

def stacked(ax,data,title,labels):
    y=np.arange(4)[::-1]
    left=np.zeros(4)
    for vals,c,nm in [([data[p][0] for p in plats],NEG,'Negative'),
                      ([data[p][1] for p in plats],NEU,'Neutral'),
                      ([data[p][2] for p in plats],POS,'Positive')]:
        vals=np.array(vals)
        ax.barh(y,vals,left=left,color=c,height=0.62,label=nm)
        for yi,v,l in zip(y,vals,left):
            if v>=7.0:
                ax.text(l+v/2,yi,f'{v:.1f}%',ha='center',va='center',fontsize=11.5,
                        color='white' if c!=NEU else '#333333')
            elif nm=='Positive':
                ax.text(l-1.5,yi,f'{v:.1f}%',ha='right',va='center',fontsize=10.5,color='#1f6ea3')
            else:
                ax.text(l+v+1.5,yi,f'{v:.1f}%',ha='left',va='center',fontsize=10.5,color=c)
        left=left+vals
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=12)
    ax.set_xlim(0,100); ax.set_xlabel('Percentage (%)',fontsize=12)
    ax.set_title(title,fontsize=13,pad=8)
    ax.tick_params(axis='x',labelsize=11)
    for s in ('top','right','left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#999999')

axA1=fig.add_axes([0.075,0.700,0.385,0.185])
axA2=fig.add_axes([0.565,0.700,0.385,0.185])
stacked(axA1,posts,'Posts ($N$ = 1,916)',plats)
stacked(axA2,cmts,'Comments ($N$ = 26,243)',[f'{p}\n($n$ = {cn[p]:,})' for p in plats])
h,l=axA1.get_legend_handles_labels()
fig.legend(h,l,loc='center',bbox_to_anchor=(0.5,0.648),ncol=3,frameon=False,fontsize=12.5)
noteA=('Note. Posts: $N$ = 1,916 patient-authored posts included in topic modeling. Comments: $N$ = 26,243 included comments across all posts (including posts excluded from\n'
       'topic modeling). Sentiment was classified using the optimized domain-specific lexicon (80 terms; Table S4); see Methods. Topic assignments via LDA ($k$ = 5, batch\n'
       'variational inference, random_state = 42).')
fig.text(0.075,0.600,noteA,ha='left',va='center',fontsize=10.5,linespacing=1.7,color='#1a1a1a')

fig.text(0.055,0.545,'(B)   Negative Sentiment by Topic × Platform',ha='left',va='center',fontsize=14.5,weight='bold')
def heat(ax,M,N,cmap,vmax,title,ylab):
    im=ax.imshow(M,cmap=cmap,vmin=0,vmax=vmax,aspect='auto')
    ax.set_xticks(range(5)); ax.set_yticks(range(4))
    ax.set_xticklabels(tp,fontsize=11); ax.set_yticklabels(plats if ylab else ['']*4,fontsize=12)
    for t,c in zip(ax.get_xticklabels(),tcol): t.set_color(c)
    ax.set_xticks(np.arange(-.5,5,1),minor=True); ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.5); ax.tick_params(which='both',length=0)
    for s in ax.spines.values(): s.set_visible(False)
    for r in range(4):
        for c in range(5):
            rgb=np.array(plt.get_cmap(cmap)(M[r,c]/vmax)[:3])
            lum=0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]
            col='white' if lum<0.55 else '#1a1a1a'
            ax.text(c,r-0.13,f'{M[r,c]:.1f}%',ha='center',va='center',fontsize=11.5,color=col)
            ax.text(c,r+0.19,f'($n$ = {N[r,c]:,})',ha='center',va='center',fontsize=9.5,color=col)
    ax.set_title(title,fontsize=13,pad=8)
    return im
axB1=fig.add_axes([0.085,0.250,0.360,0.250])
axB2=fig.add_axes([0.560,0.250,0.360,0.250])
i1=heat(axB1,pB,pN,'YlOrRd',35,'Posts',True)
i2=heat(axB2,cB,cN,'YlGnBu',22,'Comments',True)
for ax,im,pos in [(axB1,i1,[0.455,0.250,0.011,0.250]),(axB2,i2,[0.930,0.250,0.011,0.250])]:
    cax=fig.add_axes(pos); cb=fig.colorbar(im,cax=cax)
    cb.set_label('Negative %',fontsize=11.5,labelpad=8); cb.ax.tick_params(labelsize=10.5,length=3)
    cb.outline.set_linewidth(0.6); cb.outline.set_edgecolor('#888888')
noteB=('Note. Topic assignments via LDA ($k$ = 5). Cell values are the percentage of texts classified as negative, with the number of texts in parentheses.\n'
       'The two panels use separate color scales — posts shaded pale yellow to deep red (0–35%) and comments pale yellow to deep blue (0–22%) — each chosen to\n'
       'maximize contrast within its own panel; color intensity should therefore not be compared across the two panels. Topic label colors match Figure 4.')
fig.text(0.075,0.150,noteB,ha='left',va='center',fontsize=10.5,linespacing=1.7,color='#1a1a1a')
fig.savefig('./output/FigS6.pdf')
fig.savefig('./output/FigS6.png', dpi=600)
print('Figure S6 saved (PDF + PNG).')
print('S6 built')
