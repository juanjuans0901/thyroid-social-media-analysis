"""
Monte Carlo Sensitivity Analysis for Sentiment Misclassification
Supplemental Table S6 (DHJ-26-0922)
================================================================
Simulates the impact of individual-level sentiment classification errors
on group-level cross-tabulation statistics (chi-square, Cramér's V).

Uses asymmetric perturbation based on the known error pattern:
- Primary error: Positive → Neutral (positive posts misclassified as neutral)
- Secondary error: Negative → Neutral
- Tertiary: other misclassifications

Perturbation rates: 20%, 30%, 40%; 1,000 iterations are run at each rate.
The largest rate approximates the observed 41.6% error rate (i.e. 100% - 58.4%
individual-level accuracy; Table S5, Panel C).
"""

import os

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import json

np.random.seed(42)
os.makedirs('./output', exist_ok=True)

# ============================================================
# 1. Load data
# ============================================================
# posts_with_topics_FINAL.csv is written to ./output/ by 01_full_pipeline.py.
# It is also accepted from ./data/ if it was placed there by hand.
CANDIDATES = ['./output/posts_with_topics_FINAL.csv',
              './data/posts_with_topics_FINAL.csv']
INPUT = next((f for f in CANDIDATES if os.path.exists(f)), None)
if INPUT is None:
    raise SystemExit(
        "posts_with_topics_FINAL.csv not found. Run 01_full_pipeline.py first; "
        "it writes the file to ./output/. Looked in: " + ", ".join(CANDIDATES))

df = pd.read_csv(INPUT, encoding='utf-8-sig')
print(f"Loaded {len(df)} posts from {INPUT}")
missing = [c for c in ('sent_opt', 'topic_label', 'platform') if c not in df.columns]
if missing:
    raise SystemExit(
        f"{INPUT} is missing required column(s): {missing}. The sentiment column "
        "must be named 'sent_opt' (the optimised 80-term lexicon label).")
print(f"Sentiment distribution: {df['sent_opt'].value_counts().to_dict()}")
print(f"Topic distribution: {df['topic_label'].value_counts().sort_index().to_dict()}")
print(f"Platform distribution: {df['platform'].value_counts().to_dict()}")

# ============================================================
# 2. Compute baseline statistics (original data)
# ============================================================
def cramers_v(contingency_table):
    """Compute Cramér's V from a contingency table."""
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.values.sum()
    k = min(contingency_table.shape) - 1
    v = np.sqrt(chi2 / (n * k))
    return chi2, p, v

# Baseline: Topic × Sentiment
ct_topic = pd.crosstab(df['topic_label'], df['sent_opt'])
chi2_topic_base, p_topic_base, v_topic_base = cramers_v(ct_topic)

# Baseline: Platform × Sentiment
ct_platform = pd.crosstab(df['platform'], df['sent_opt'])
chi2_platform_base, p_platform_base, v_platform_base = cramers_v(ct_platform)

print(f"\n=== BASELINE STATISTICS ===")
print(f"Topic × Sentiment:    χ²={chi2_topic_base:.3f}, p={p_topic_base:.6f}, V={v_topic_base:.3f}")
print(f"Platform × Sentiment: χ²={chi2_platform_base:.3f}, p={p_platform_base:.6f}, V={v_platform_base:.3f}")
print(f"V_topic - V_platform = {v_topic_base - v_platform_base:.3f}")
print(f"V_topic > V_platform: {v_topic_base > v_platform_base}")

# ============================================================
# 3. Define asymmetric confusion/transition matrix
# ============================================================
# Based on paper's reported metrics:
# - Individual accuracy: 58.4%, Macro F1 = 0.503  (Table S5, Panel C)
# - Neutral F1 = 0.707, Positive F1 = 0.492, Negative F1 = 0.310
# - Primary error: Positive misclassified as Neutral
#
# Estimated transition probabilities (when a label IS perturbed):
# Row = original label, Col = what it gets changed to
# These reflect the asymmetric error pattern

labels = ['Negative', 'Neutral', 'Positive']

# When a label is selected for perturbation, it transitions to another label
# with these probabilities (rows must sum to 1, diagonal = 0 since we're perturbing)
transition_matrix = {
    'Negative': {'Negative': 0.0, 'Neutral': 0.70, 'Positive': 0.30},   # Neg mainly → Neutral
    'Neutral':  {'Negative': 0.40, 'Neutral': 0.0, 'Positive': 0.60},   # Neutral → either, slight positive bias
    'Positive': {'Negative': 0.15, 'Neutral': 0.85, 'Positive': 0.0},   # Pos mainly → Neutral (key error)
}

print(f"\n=== TRANSITION MATRIX (when perturbed) ===")
for orig, targets in transition_matrix.items():
    print(f"  {orig:>8s} → {targets}")

# ============================================================
# 4. Monte Carlo simulation
# ============================================================
N_SIMULATIONS = 1000
PERTURBATION_RATES = [0.20, 0.30, 0.40]

results = {}

for rate in PERTURBATION_RATES:
    v_topics = []
    v_platforms = []
    v_diffs = []
    p_topics = []
    p_platforms = []
    topic_sig_count = 0
    platform_sig_count = 0
    ordering_preserved = 0
    
    for sim in range(N_SIMULATIONS):
        # Create perturbed labels
        perturbed = df['sent_opt'].copy()
        
        for idx in range(len(df)):
            if np.random.random() < rate:
                orig_label = perturbed.iloc[idx]
                # Sample new label from transition matrix
                probs = transition_matrix[orig_label]
                new_label = np.random.choice(labels, p=[probs[l] for l in labels])
                perturbed.iloc[idx] = new_label
        
        # Recompute contingency tables
        ct_t = pd.crosstab(df['topic_label'], perturbed)
        ct_p = pd.crosstab(df['platform'], perturbed)
        
        # Ensure all columns present
        for l in labels:
            if l not in ct_t.columns:
                ct_t[l] = 0
            if l not in ct_p.columns:
                ct_p[l] = 0
        ct_t = ct_t[labels]
        ct_p = ct_p[labels]
        
        chi2_t, p_t, v_t = cramers_v(ct_t)
        chi2_p, p_p, v_p = cramers_v(ct_p)
        
        v_topics.append(v_t)
        v_platforms.append(v_p)
        v_diffs.append(v_t - v_p)
        p_topics.append(p_t)
        p_platforms.append(p_p)
        
        if p_t < 0.001:
            topic_sig_count += 1
        if p_p < 0.001:
            platform_sig_count += 1
        if v_t > v_p:
            ordering_preserved += 1
    
    v_topics = np.array(v_topics)
    v_platforms = np.array(v_platforms)
    v_diffs = np.array(v_diffs)
    p_topics = np.array(p_topics)
    p_platforms = np.array(p_platforms)
    
    results[rate] = {
        'v_topic_median': np.median(v_topics),
        'v_topic_mean': np.mean(v_topics),
        'v_topic_95ci': (np.percentile(v_topics, 2.5), np.percentile(v_topics, 97.5)),
        'v_platform_median': np.median(v_platforms),
        'v_platform_mean': np.mean(v_platforms),
        'v_platform_95ci': (np.percentile(v_platforms, 2.5), np.percentile(v_platforms, 97.5)),
        'v_diff_median': np.median(v_diffs),
        'v_diff_95ci': (np.percentile(v_diffs, 2.5), np.percentile(v_diffs, 97.5)),
        'ordering_preserved_pct': ordering_preserved / N_SIMULATIONS * 100,
        'topic_sig_pct': topic_sig_count / N_SIMULATIONS * 100,
        'platform_sig_pct': platform_sig_count / N_SIMULATIONS * 100,
    }

# ============================================================
# 5. Print results
# ============================================================
print(f"\n{'='*80}")
print(f"MONTE CARLO SENSITIVITY ANALYSIS RESULTS ({N_SIMULATIONS} simulations)")
print(f"{'='*80}")

for rate in PERTURBATION_RATES:
    r = results[rate]
    print(f"\n--- Perturbation Rate: {rate*100:.0f}% ---")
    print(f"  Topic × Sentiment V:")
    print(f"    Median = {r['v_topic_median']:.3f}  (95% SI: {r['v_topic_95ci'][0]:.3f}–{r['v_topic_95ci'][1]:.3f})")
    print(f"  Platform × Sentiment V:")
    print(f"    Median = {r['v_platform_median']:.3f}  (95% SI: {r['v_platform_95ci'][0]:.3f}–{r['v_platform_95ci'][1]:.3f})")
    print(f"  V_topic − V_platform:")
    print(f"    Median = {r['v_diff_median']:.3f}  (95% SI: {r['v_diff_95ci'][0]:.3f}–{r['v_diff_95ci'][1]:.3f})")
    print(f"  Ordering V_topic > V_platform preserved: {r['ordering_preserved_pct']:.1f}%")
    print(f"  Topic association p < .001: {r['topic_sig_pct']:.1f}% of iterations")
    print(f"  Platform association p < .001: {r['platform_sig_pct']:.1f}% of iterations")

# ============================================================
# 6. Save detailed results
# ============================================================
output = {
    'baseline': {
        'chi2_topic': round(chi2_topic_base, 3),
        'p_topic': p_topic_base,
        'v_topic': round(v_topic_base, 3),
        'chi2_platform': round(chi2_platform_base, 3),
        'p_platform': p_platform_base,
        'v_platform': round(v_platform_base, 3),
    },
    'simulations': {}
}
for rate in PERTURBATION_RATES:
    r = results[rate]
    output['simulations'][f'{rate*100:.0f}pct'] = {
        'v_topic_median': round(r['v_topic_median'], 3),
        'v_topic_95ci': [round(r['v_topic_95ci'][0], 3), round(r['v_topic_95ci'][1], 3)],
        'v_platform_median': round(r['v_platform_median'], 3),
        'v_platform_95ci': [round(r['v_platform_95ci'][0], 3), round(r['v_platform_95ci'][1], 3)],
        'v_diff_median': round(r['v_diff_median'], 3),
        'v_diff_95ci': [round(r['v_diff_95ci'][0], 3), round(r['v_diff_95ci'][1], 3)],
        'ordering_preserved_pct': round(r['ordering_preserved_pct'], 1),
        'topic_sig_pct': round(r['topic_sig_pct'], 1),
        'platform_sig_pct': round(r['platform_sig_pct'], 1),
    }

with open('./output/sensitivity_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n\nResults saved to ./output/sensitivity_results.json")
