"""
Figure 4. Thematic landscape of thyroid nodule and cancer patient discourse
across Chinese social media.
DHJ-26-0922 (Digital Health) — figure generation script.

Four panels, matching the submitted layout:
  A  The five LDA-derived topics (N = 1,916 posts)
  B  The 25 categories of patient concern in five content domains (D1-D5),
     from the directed content analysis of 1,284 coded posts. Presented as a
     separate, single-row panel because the coding frame was developed
     alongside rather than after the topic solution: D1-D5 are NOT nested
     within T1-T5 and the domain identifiers are independent of the topic
     identifiers.
  C  Sentiment profile by topic, posts versus comments
  D  Mapping to Leventhal's Common Sense Model, including the parallel
     emotional-representation pathway

Category percentages are the proportion of coded posts within each domain to
which the category was applied, taken from the two-coder consensus coding
(Supplemental Table S7). Topic sizes and sentiment profiles are regenerated
from the analytic corpus by 01_full_pipeline.py.

Fonts: this figure contains no Chinese glyphs. It is drawn with hand-placed
coordinates, so it is laid out against the same face that produced the
submitted PDF (Noto Sans SC / Noto Sans CJK SC, which also carries the Latin
set). If that font is not present the script falls back to a metric-compatible
Arial substitute and prints a warning; label positions may shift slightly.
"""

import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs("./output", exist_ok=True)

# ── Font resolution ───────────────────────────────────────────────────────────
FAMILY = ["Liberation Sans", "Arial", "DejaVu Sans"]
_cands = [os.path.join(HERE, n) for n in
          ("NotoSansSC-Regular.otf", "NotoSansCJKsc-Regular.otf")]
_cands += sorted(glob.glob(os.path.join(HERE, "NotoSans*SC*.[ot]tf")))
_font = next((f for f in _cands if os.path.exists(f)), None)
if _font:
    fm.fontManager.addfont(_font)
    FAMILY = ["Noto Sans SC", "Noto Sans CJK SC"] + FAMILY
else:
    print("WARNING: NotoSansSC-Regular.otf not found next to this script. "
          "Falling back to a metric-compatible Arial substitute; label "
          "positions in this hand-laid-out figure may shift slightly.")

plt.rcParams.update({
    "font.family": FAMILY,
    "pdf.fonttype": 42,     # embed as TrueType so all glyphs survive export
    "ps.fonttype": 42,
    "mathtext.fontset": "custom",
    "mathtext.rm": FAMILY[0],
    "mathtext.it": FAMILY[0] + ":italic",
    "mathtext.bf": FAMILY[0] + ":bold",
    "axes.unicode_minus": False,
})

# ── Unified palette, anchored to Figures 2, S3, S4 and S6 ─────────────────────
C = {"T1": "#1f77b4", "T2": "#d2691e", "T3": "#2ca02c",
     "T4": "#9467bd", "T5": "#17becf"}
GREY = "#3d4852"

# ── Panel A: the five topics (sizes from 01_full_pipeline.py) ────────────────
TOP = [
    ("T1", "Postoperative Recovery", "n = 263 (13.7%)", "CSM: CONSEQUENCES"),
    ("T2", "Postoperative Medication\nand Surveillance", "n = 206 (10.8%)",
     "CSM: CURE/CONTROL (ACTION)"),
    ("T3", "Living With Thyroid Cancer", "n = 434 (22.7%)", "CSM: CONSEQUENCES"),
    ("T4", "Treatment Decision\nand Debate", "n = 552 (28.8%)",
     "CSM: TIMELINE + CURE/CONTROL"),
    ("T5", "Healthcare Navigation", "n = 461 (24.1%)", "CSM: ILLNESS IDENTITY"),
]

# ── Panel B: 25 categories in five content domains ───────────────────────────
# Category names and percentages are the locked values from the two-coder
# consensus coding (Supplemental Table S7). Percentages are of the coded posts
# within that domain, so they do not sum to 100.
DOM = [
    ("D1  Postoperative daily life  ($\\it{n}$ = 502)",
     ["Diet, daily life & return to work 55.8%",
      "Medication & surveillance mgmt 44.6%",
      "Psychological adaptation 37.3%",
      "Symptom burden & functioning 33.3%",
      "Peer support & reprod. planning 16.3%"]),
    ("D2  Treatment decision-making  ($\\it{n}$ = 215)",
     ["Prognosis & risk perception 73.5%",
      "Surgical/ablation/AS decision 62.3%",
      "Adjuvant therapy decision 34.4%",
      "Diagnostic appraisal & evidence 23.7%",
      "Decisional distress 8.8%"]),
    ("D3  Diagnostic appraisal  ($\\it{n}$ = 164)",
     ["Nodule characterisation & dx 88.4%",
      "Intervention timing & pathway 39.6%",
      "Aetiology & risk factors 30.5%",
      "Imaging & comorbidity 22.0%",
      "Diagnostic anxiety & coping 21.3%"]),
    ("D4  Evidence appraisal  ($\\it{n}$ = 126)",
     ["Microcarcinoma AS advocacy 50.8%",
      "Overtreatment & commerc. critique 42.1%",
      "Ablation appraisal & provider 40.5%",
      "Evidence-based info & KOL 38.9%",
      "Radiation risk & screening 10.3%"]),
    ("D5  Care-seeking experience  ($\\it{n}$ = 277)",
     ["Surgical & ablation experience 95.3%",
      "Navigation & logistics 65.7%",
      "Provider interaction & coord. 54.5%",
      "Service experience & costs 27.1%",
      "Emotional response & adj. 4.7%"]),
]

# ── Panel C: sentiment profile — (negative, neutral, positive) ───────────────
SENT = {"T1": ((23.2, 48.3, 28.5), (11.8, 75.2, 13.0)),
        "T2": ((17.0, 65.0, 18.0), (10.4, 81.2, 8.4)),
        "T3": ((25.3, 46.5, 28.1), (9.5, 78.2, 12.4)),
        "T4": ((12.0, 74.1, 13.9), (7.7, 83.9, 8.3)),
        "T5": ((19.5, 49.5, 31.0), (10.8, 79.4, 9.8))}

# ── Panel D: Common Sense Model ──────────────────────────────────────────────
CSM = [
    ("Identity", '"What is my condition?"', "Labelling & symptom recognition",
     "→  T5", "#17becf"),
    ("Cause", '"Why did I get this?"', "Causal attribution",
     "→  no topic; category level only (D3-3)", "#8a8f98"),
    ("Timeline", '"How long will it last?"', "Trajectory & recurrence beliefs",
     "→  T4", "#9467bd"),
    ("Consequences", '"How will it affect my life?"', "Functional & social impact",
     "→  T1, T3", "#1f77b4"),
    ("Cure / Control", '"Can it be managed?"', "Beliefs → T4    Action → T2",
     "", "#d2691e"),
]

fig = plt.figure(figsize=(16.4, 13.6), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def T(x, y, s, **k):
    ax.text(x, y, s, transform=ax.transData, **k)


T(50, 97.8, "Figure 4. Thematic Landscape of Thyroid Nodule and Cancer Patient "
            "Discourse Across Chinese Social Media",
  ha="center", fontsize=15.5, fontweight="bold")
T(50, 96.2, "Five LDA-derived topics, the categories of patient concern from "
            "directed content analysis, sentiment profile (posts vs comments), "
            "and mapping to Leventhal’s Common Sense Model.",
  ha="center", fontsize=10.2, color=GREY)

# ══════════════════════════════════════════════════════════ Panel A
T(3, 93.6, "A.  Five LDA-derived topics ($\\it{N}$ = 1,916 posts)",
  fontsize=12.5, fontweight="bold")
x0, wid, gap = 3.0, 18.2, 0.5
for i, (k, name, n, csm) in enumerate(TOP):
    x = x0 + i * (wid + gap)
    ax.add_line(plt.Line2D([x, x + wid], [92.3, 92.3], color=C[k], lw=2.6))
    T(x + wid / 2, 90.9, k, ha="center", fontsize=13, fontweight="bold", color=C[k])
    T(x + wid / 2, 89.6, name, ha="center", va="top", fontsize=9.8, fontweight="bold")
    yy = 87.3 if "\n" in name else 88.4
    T(x + wid / 2, yy, n, ha="center", va="top", fontsize=9.4, style="italic", color=GREY)
    ax.add_patch(Rectangle((x, 85.0), wid, 1.5, fc=C[k], alpha=0.13, ec="none"))
    T(x + wid / 2, 85.75, csm, ha="center", va="center", fontsize=8.4,
      fontweight="bold", color=C[k])

# ══════════════════════════════════════════════════════════ Panel B
ax.add_line(plt.Line2D([2, 98], [83.4, 83.4], color="#c9ced6", lw=1.1))
ax.add_patch(FancyBboxPatch((2.2, 68.6), 95.6, 14.1,
                            boxstyle="round,pad=0.25", fc="#fbfcfd",
                            ec="#c9ced6", lw=1.0))
T(3.4, 81.6, "B.  Categories of patient concern — 25 categories in five content "
             "domains (1,284 coded posts; mean Cohen’s κ = 0.722)",
  fontsize=12.5, fontweight="bold")
T(3.4, 80.5, "The coding frame was developed alongside rather than after the final "
             "topic solution. The five content domains (D1–D5) are therefore mapped "
             "in parallel with the five topics and are NOT nested within them;",
  fontsize=8.9, style="italic", color="#4a5058")
T(3.4, 79.5, "domain identifiers D1–D5 are independent of topic identifiers T1–T5. "
             "Categories are listed in order of prominence within each domain, with the "
             "percentage of coded posts in that domain to which each was applied. "
             "Full definitions in Supplemental Table S7.",
  fontsize=8.9, style="italic", color="#4a5058")

# The five domains sit in a single row, one column each, so that no domain is
# visually aligned beneath a topic column of Panel A.
pos = [(4.6, 77.0), (23.6, 77.0), (42.6, 77.0), (61.6, 77.0), (80.6, 77.0)]
for (dx, dy), (dname, cats) in zip(pos, DOM):
    T(dx, dy, dname, fontsize=8.7, fontweight="bold", color=GREY)
    ax.add_line(plt.Line2D([dx, dx + 17.4], [dy - 0.55, dy - 0.55],
                           color="#aeb5bf", lw=0.9))
    for j, c in enumerate(cats):
        T(dx + 0.35, dy - 1.95 - j * 1.25, "•  " + c, fontsize=7.6)

# ══════════════════════════════════════════════════════════ Panel C
T(3, 66.3, "C.  Sentiment profile by topic (posts vs comments)",
  fontsize=12.5, fontweight="bold")
leg = [("Negative (posts)", "#c0522a"), ("Negative (comments)", "#f0a882"),
       ("Neutral", "#dcdcdc"), ("Positive (posts)", "#2874a6"),
       ("Positive (comments)", "#a9cce3")]
lx = 3
for lab, col in leg:
    ax.add_patch(Rectangle((lx, 64.5), 2.4, 1.0, fc=col, ec="none"))
    T(lx + 2.9, 65.0, lab, va="center", fontsize=9)
    lx += len(lab) * 0.55 + 6.0

BL, BW = 13.0, 80.0
y = 62.7
for k in ["T1", "T2", "T3", "T4", "T5"]:
    T(4.2, y - 1.15, k, fontsize=11, fontweight="bold", color=C[k], va="center")
    for r, (row, lbl) in enumerate(zip(SENT[k], ["Posts", "Comments"])):
        yy = y - r * 2.3
        T(12.3, yy - 0.5, lbl, ha="right", va="center", fontsize=8.8, color=GREY)
        cx = BL
        cols = [("#c0522a" if r == 0 else "#f0a882"), "#dcdcdc",
                ("#2874a6" if r == 0 else "#a9cce3")]
        for v, col in zip(row, cols):
            w = BW * v / 100.0
            ax.add_patch(Rectangle((cx, yy - 1.0), w, 1.0, fc=col, ec="none"))
            if w > 2.6:
                T(cx + w / 2, yy - 0.5, f"{v:.1f}", ha="center", va="center",
                  fontsize=8.5, fontweight="bold",
                  color="white" if col in ("#c0522a", "#2874a6") else "#33393f")
            cx += w
    y -= 5.35
T(50, 36.1, "Solid bars = posts ($\\it{N}$ = 1,916); lighter bars = comments "
            "($\\it{N}$ = 12,380 topic-linked). Each bar sums to 100%.",
  ha="center", fontsize=9, color=GREY)

# ══════════════════════════════════════════════════════════ Panel D
ax.add_line(plt.Line2D([2, 98], [34.4, 34.4], color="#c9ced6", lw=1.1))
T(3, 32.8, "D.  Mapping to Leventhal’s Common Sense Model (CSM) of Self-Regulation",
  fontsize=12.5, fontweight="bold")
T(3, 31.6, "Concurrent cognitive dimensions activated in parallel; topic numbering "
           "is algorithm-assigned and does not imply sequence.",
  fontsize=9, color=GREY)
T(50, 30.4, "C O G N I T I V E   R E P R E S E N T A T I O N S",
  ha="center", fontsize=10, fontweight="bold", color=GREY)
cw = 18.2
for i, (nm, q, desc, arrow, col) in enumerate(CSM):
    x = 3.0 + i * (cw + 0.5)
    ax.add_line(plt.Line2D([x, x + cw], [29.1, 29.1], color=col, lw=2.4))
    T(x, 27.6, nm, fontsize=11.4, fontweight="bold", color=col)
    T(x, 26.2, q, fontsize=9, style="italic", color=GREY)
    T(x, 24.8, desc, fontsize=8.6, va="top")
    if arrow:
        T(x, 23.5, arrow, fontsize=8.6, va="top", fontweight="bold")

ax.add_line(plt.Line2D([2, 98], [21.8, 21.8], color="#c9ced6", lw=1.0))
T(3, 20.4, "E M O T I O N A L   R E P R E S E N T A T I O N  —  "
           "P A R A L L E L   A F F E C T I V E   P A T H W A Y",
  fontsize=10, fontweight="bold", color="#4a5058")
T(3, 18.5, "Emotional Representation", fontsize=11.4, fontweight="bold")
T(3, 17.2, '"How do I feel about it?"', fontsize=9, style="italic", color=GREY)
T(3, 16.0, "Emotional processing  →  no topic", fontsize=8.8,
  fontweight="bold", va="top")
T(26, 18.4, "Emotional processing did not emerge as an independent topic. Emotional "
            "content is embedded across content domains, in psychological adaptation "
            "(D1-3),", fontsize=9.2, va="top")
T(26, 17.1, "decisional distress (D2-5), diagnostic anxiety and coping (D3-5), ablation "
            "appraisal and provider choice (D4-5), and emotional response and adjustment "
            "(D5-5),", fontsize=9.2, va="top")
T(26, 15.8, "consistent with the CSM dual-pathway hypothesis.", fontsize=9.2, va="top")

# ══════════════════════════════════════════════════════════ note
note = (
    "Note. Topics derived from LDA ($\\it{k}$ = 5, batch variational inference, "
    "random_state = 42) on 1,916 patient-authored posts across Xiaohongshu, Weibo, "
    "Zhihu and Douyin; the model is fully deterministic and reproduces from the "
    "deposited scripts.\n"
    "Topic labels were assigned by two researchers who independently inspected the "
    "topic–word distributions and representative posts and reconciled the result "
    "through discussion. Categories of patient concern come from directed content "
    "analysis of high-loading\n"
    "posts (dominant-topic probability > 0.60; 16 institutional posts excluded) and "
    "are reported as an ordinal ranking within each content domain. Comment sentiment "
    "is based on 12,380 topic-linked comments, each inheriting its parent post’s "
    "dominant topic.\n"
    "Sentiment was classified with the optimised domain-specific lexicon (80 terms; "
    "Supplemental Table S4) and is valid at group level only. CSM mapping was "
    "performed independently by two reviewers (100% agreement). Cause did not "
    "correspond to any topic and\n"
    "appears only at category level (D3-3 aetiology and risk factor awareness). "
    "Category percentages give the proportion of coded posts within each domain to "
    "which a category was applied and derive from the two-coder consensus coding.")
T(3, 12.6, note, fontsize=8.0, va="top", color="#4a5058", linespacing=1.45)

fig.savefig("./output/Fig4.pdf", facecolor="white")
fig.savefig("./output/Fig4.png", dpi=600, facecolor="white")
plt.close()
print("Figure 4 saved (PDF + PNG).")
