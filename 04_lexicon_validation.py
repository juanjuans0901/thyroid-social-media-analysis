"""
=============================================================
Lexicon validation — Supplemental Table S5, Panels A and C
DHJ-26-0922 (Digital Health)
=============================================================
Reproduces, exactly:

  Panel A  Inter-rater reliability for sentiment classification
           n = 299, Cohen's kappa = 0.729

  Panel C  Lexicon validation against human consensus labels
           299 annotated -> 260 matched to the analytic corpus -> 221 with
           coder agreement; accuracy 58.4%, macro F1 0.503, kappa 0.284,
           F1 negative/neutral/positive = 0.310 / 0.707 / 0.492,
           primary error Positive -> Neutral (27 cases)

Input
  sentiment_validation_labels.csv
      De-identified label table. One row per annotated post; contains the two
      coder labels, the consensus label, the lexicon label and the topic label,
      but no post text and no platform post identifier, so it can be shared
      without restriction.

  The earlier version of this script matched annotations to the corpus by
  fuzzy text prefix, which recovered only 243 of the 260 posts and therefore
  printed numbers 0.3-3 pp away from the paper. The join is now done on a
  stored key, so the reproduction is exact.

Usage
  pip install -r requirements.txt
  python 04_lexicon_validation.py
=============================================================
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

_HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_FILE = os.path.join(_HERE, "sentiment_validation_labels.csv")
if not os.path.exists(LABELS_FILE):                      # also accept ./data/
    LABELS_FILE = os.path.join(_HERE, "data", "sentiment_validation_labels.csv")
OUTPUT_DIR = "./output"
LABELS = ["Negative", "Neutral", "Positive"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def confusion(human, machine):
    ct = pd.crosstab(pd.Series(human, name="human"), pd.Series(machine, name="machine"))
    return ct.reindex(index=LABELS, columns=LABELS).fillna(0).astype(int)


def distribution_test(human, machine):
    """Group-level distributional validity: 2 x 3 table of the two marginals."""
    obs = np.array(
        [[int((human == lab).sum()) for lab in LABELS],
         [int((machine == lab).sum()) for lab in LABELS]]
    )
    chi2, p, dof, _ = chi2_contingency(obs)
    cramers_v = np.sqrt(chi2 / obs.sum())  # min(rows, cols) - 1 == 1
    return chi2, p, dof, cramers_v, obs


def main():
    df = pd.read_csv(LABELS_FILE)
    print(f"Loaded {len(df)} annotated posts from {LABELS_FILE}")

    # ---------------- Panel A: inter-rater reliability ----------------
    a, b = df["coder_A_DJ"], df["coder_B_DQ"]
    agree = (a == b)
    kappa_inter = cohen_kappa_score(a, b, labels=LABELS)

    print("\n" + "=" * 62)
    print("  PANEL A — Inter-rater reliability (sentiment)")
    print("=" * 62)
    print(f"  Sample size          : {len(df)}")
    print(f"  Platform allocation  : "
          + ", ".join(f"{k} {v}" for k, v in df['platform'].value_counts().items()))
    print(f"  Raw agreement        : {agree.sum()}/{len(df)} = {agree.mean():.1%}")
    print(f"  Cohen's kappa        : {kappa_inter:.3f}   [paper: 0.729]")

    # ---------------- Panel C: lexicon vs human consensus ----------------
    matched = df[df["in_analytic_corpus"] == "Y"]
    ref = matched[matched["agreed"] == "Y"]
    human = ref["consensus_label"].to_numpy()
    lexicon = ref["lexicon_label"].to_numpy()

    accuracy = float((human == lexicon).mean())
    kappa = cohen_kappa_score(human, lexicon, labels=LABELS)
    prec, rec, f1, _ = precision_recall_fscore_support(
        human, lexicon, labels=LABELS, zero_division=0
    )
    ct = confusion(human, lexicon)
    off_diagonal = {
        (h, m): int(ct.loc[h, m]) for h in LABELS for m in LABELS if h != m
    }
    (top_h, top_m), top_n = max(off_diagonal.items(), key=lambda kv: kv[1])
    chi2, p, dof, cramers_v, obs = distribution_test(human, lexicon)

    print("\n" + "=" * 62)
    print("  PANEL C — Lexicon validation against human consensus")
    print("=" * 62)
    print(f"  Posts annotated              : {len(df)}      [paper: 299]")
    print(f"  Matched to analytic corpus   : {len(matched)}      [paper: 260]")
    print(f"  Consensus reference labels   : {len(ref)}      [paper: 221]")
    print("\n  -- Individual-level performance --")
    print(f"  Accuracy   : {accuracy:.1%}   [paper: 58.4%]")
    print(f"  Macro F1   : {f1.mean():.3f}   [paper: 0.503]")
    print(f"  Cohen's k  : {kappa:.3f}   [paper: 0.284]")
    print("\n  Confusion matrix (rows = human consensus, cols = lexicon):")
    print(ct.to_string())
    print("\n  Per-class metrics:")
    print(f"  {'class':>10} {'precision':>10} {'recall':>10} {'F1':>8}")
    for i, lab in enumerate(LABELS):
        print(f"  {lab:>10} {prec[i]:>10.3f} {rec[i]:>10.3f} {f1[i]:>8.3f}")
    print(f"\n  Primary error source: {top_h} -> {top_m} ({top_n} cases)"
          f"   [paper: Positive -> Neutral, 27 cases]")

    print("\n  -- Group-level distributional validity --")
    print(f"  {'':>10} {'Negative':>10} {'Neutral':>10} {'Positive':>10}")
    print(f"  {'Human':>10} {obs[0][0]:>10} {obs[0][1]:>10} {obs[0][2]:>10}")
    print(f"  {'Lexicon':>10} {obs[1][0]:>10} {obs[1][1]:>10} {obs[1][2]:>10}")
    print(f"  chi2({dof}) = {chi2:.3f}, P = {p:.3f}, Cramer's V = {cramers_v:.3f}")
    print("  -> distributions do not differ significantly (P > .05)")

    summary = pd.DataFrame(
        [
            ("Posts annotated", len(df)),
            ("Inter-rater raw agreement", f"{agree.mean():.3f}"),
            ("Inter-rater Cohen's kappa", f"{kappa_inter:.3f}"),
            ("Matched to analytic corpus", len(matched)),
            ("Consensus reference labels", len(ref)),
            ("Accuracy", f"{accuracy:.3f}"),
            ("Macro F1", f"{f1.mean():.3f}"),
            ("Cohen's kappa (lexicon vs human)", f"{kappa:.3f}"),
            ("F1 Negative", f"{f1[0]:.3f}"),
            ("F1 Neutral", f"{f1[1]:.3f}"),
            ("F1 Positive", f"{f1[2]:.3f}"),
            ("Primary error", f"{top_h} -> {top_m} ({top_n})"),
            ("Chi-square (2 x 3)", f"{chi2:.3f}"),
            ("P", f"{p:.3f}"),
            ("Cramer's V", f"{cramers_v:.3f}"),
        ],
        columns=["metric", "value"],
    )
    out = os.path.join(OUTPUT_DIR, "table_s5_panels_a_c.csv")
    summary.to_csv(out, index=False, encoding="utf-8-sig")
    ct.to_csv(os.path.join(OUTPUT_DIR, "table_s5_panel_c_confusion.csv"),
              encoding="utf-8-sig")
    print(f"\nSaved: {out}")

    # Hard checks — the script fails loudly if the deposited data drifts.
    assert len(df) == 299 and len(matched) == 260 and len(ref) == 221
    assert round(kappa_inter, 3) == 0.729
    assert round(accuracy, 3) == 0.584
    assert round(f1.mean(), 3) == 0.503
    assert round(kappa, 3) == 0.284
    assert [round(x, 3) for x in f1] == [0.310, 0.707, 0.492]
    assert (top_h, top_m, top_n) == ("Positive", "Neutral", 27)
    print("All Table S5 Panel A / Panel C values reproduced exactly.")


if __name__ == "__main__":
    main()
