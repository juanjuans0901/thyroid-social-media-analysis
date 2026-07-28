"""
=============================================================
Thyroid nodule / thyroid cancer social media discourse — full pipeline
甲状腺结节与甲状腺癌社交媒体话语分析 — 完整流水线
DHJ-26-0922 (Digital Health)
=============================================================
Input : ./data/posts_FINAL_CLEANED_v3.csv      (N = 1,916 patient-authored posts)
        ./data/comments_CONSENSUS_FINAL.csv    (N = 26,243 included comments)
Output: ./output/  — Supplemental Tables S8, S9, S10, S11, the data behind
        Supplemental Figures S3, S5 and S6, and
        posts_with_topics_FINAL.csv (the file consumed by script 03)

Domain word lists are imported from resources.py (single source of truth):
  MEDICAL_TERMS      n = 65   Table S1
  STOPWORDS          n = 172  Table S2
  SENTIMENT_LEXICON  n = 80   Table S4 (optimized lexicon)
  EMOTION_CATEGORIES n = 80   7 DUTIR categories (Figure S3, Table S12)

Expected topic sizes: T1 = 263, T2 = 206, T3 = 434, T4 = 552, T5 = 461.
The run asserts these at the end of STEP 6.

Multiple comparisons: Bonferroni is applied ONCE. Each unadjusted p is compared
with the adjusted alpha (.05/6 for the six platform pairs, .05/10 for the ten
topic pairs). The Bonferroni-adjusted p (p x m) is exported alongside it for
transparency only; it must be compared with .05, never with the adjusted alpha.

Install:
  pip install -r requirements.txt
=============================================================
"""

import pandas as pd
import numpy as np
import jieba
import re
import os
import sys
import warnings
from collections import defaultdict
from itertools import combinations

warnings.filterwarnings('ignore')

# ==============================
# 配置区 — 请修改文件路径
# ==============================
POSTS_FILE = r"./data/posts_FINAL_CLEANED_v3.csv"
COMMENTS_FILE = r"./data/comments_CONSENSUS_FINAL.csv"
OUTPUT_DIR = r"./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Domain resources (Tables S1, S2, S4) — see resources.py
# ==============================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resources import (
    MEDICAL_TERMS,          # n = 65,  Table S1
    STOPWORDS,              # n = 172, Table S2
    SENTIMENT_LEXICON,      # n = 80,  Table S4 (optimized)
    EMOTION_CATEGORIES,     # 7 DUTIR categories
    NEGATION_WORDS,         # n = 9
    NEGATION_WINDOW_CHARS,  # = 4
    net_threshold,          # length-adaptive net-score threshold
)

assert len(STOPWORDS) == 172, "STOPWORDS must be the 172-word list in Table S2"
assert len(SENTIMENT_LEXICON) == 80, "SENTIMENT_LEXICON must be the 80-term optimized lexicon"


# ==============================
# 辅助函数
# ==============================

def classify_sentiment(text):
    """Three-class polarity for one text.

    Implements the Methods exactly: lexicon terms are matched against the
    *cleaned text before segmentation*, so negation cues survive; a term is
    reversed when a negation word occurs in the NEGATION_WINDOW_CHARS (= 4)
    characters immediately before it; and the net score is compared with a
    length-adaptive threshold (>2 for texts >500 characters, >1 for 100-500,
    >0 for shorter texts).

    Returns (label, neg_score, pos_score, emotion_counts, n_lexicon_hits).

    Emotion counts are raw intensity-weighted term frequencies with NO
    negation handling, per the Note on Emotion Category Scoring in the
    Methods; they feed Figure S3 and Table S12.
    """
    text = str(text)
    neg_score = 0
    pos_score = 0
    hits = 0
    emotion_counts = defaultdict(float)

    for term, weight in SENTIMENT_LEXICON.items():
        start = 0
        while True:
            k = text.find(term, start)
            if k < 0:
                break
            hits += 1
            preceding = text[max(0, k - NEGATION_WINDOW_CHARS):k]
            negated = any(neg in preceding for neg in NEGATION_WORDS)
            score = -weight if negated else weight
            if score < 0:
                neg_score += abs(score)
            else:
                pos_score += score
            if term in EMOTION_CATEGORIES:
                emotion_counts[EMOTION_CATEGORIES[term]] += abs(weight)
            start = k + len(term)

    net = pos_score - neg_score
    threshold = net_threshold(len(text))
    if net > threshold:
        label = "Positive"
    elif net < -threshold:
        label = "Negative"
    else:
        label = "Neutral"

    return label, neg_score, pos_score, dict(emotion_counts), hits


def tokenize_for_lda(text):
    """对单条文本进行分词（用于LDA）"""
    text = str(text)
    words = jieba.lcut(text)
    cleaned = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if w in STOPWORDS:
            continue
        if len(w) == 1:
            continue
        if re.match(r'^[\d.]+$', w):
            continue
        if re.match(r'^[a-zA-Z]$', w):
            continue
        if re.match(r'^[\W]+$', w):
            continue
        cleaned.append(w)
    return cleaned


# ==============================
# STEP 1: 加载数据
# ==============================
print("=" * 60)
print("STEP 1: 加载数据")
print("=" * 60)

posts = pd.read_csv(POSTS_FILE)
print(f"  帖子: {len(posts)} 条")
print(f"  列: {list(posts.columns)}")
print(f"  平台分布:")
print(posts['platform'].value_counts().to_string(header=False))

# 加载评论
try:
    comments = pd.read_csv(COMMENTS_FILE, encoding='utf-8')
except:
    comments = pd.read_csv(COMMENTS_FILE, encoding='iso-8859-1')
print(f"\n  评论: {len(comments)} 条")

# 筛选已纳入的评论
if 'comment_decision' in comments.columns:
    included_comments = comments[
        comments['comment_decision'].str.lower().str.strip().isin(
            ['include', 'included', '1', '1.0', 'yes']
        )
    ].copy()
    if len(included_comments) == 0:
        print("  警告: comment_decision筛选后为0条，尝试其他值...")
        print(f"  comment_decision唯一值: {comments['comment_decision'].unique()[:20]}")
        included_comments = comments.copy()
    print(f"  纳入评论: {len(included_comments)} 条")
else:
    included_comments = comments.copy()
    print(f"  无comment_decision列，使用全部评论: {len(included_comments)} 条")


# ==============================
# STEP 2: 注册医学词典 + 分词
# ==============================
print("\n" + "=" * 60)
print("STEP 2: 分词 (jieba + 65个医学词)")
print("=" * 60)

# jieba segmentation is used for LDA only; sentiment scoring works on the
# raw cleaned text (see classify_sentiment).
for term in MEDICAL_TERMS:
    jieba.add_word(term)

docs_tokenized = [tokenize_for_lda(text) for text in posts['clean_text']]
print(f"  分词完成: {len(docs_tokenized)} 篇文档")
avg_len = np.mean([len(d) for d in docs_tokenized])
print(f"  平均词数: {avg_len:.1f}")


# ==============================
# STEP 3: 情感分类 (80词词典)
# ==============================
print("\n" + "=" * 60)
print("STEP 3: 情感分类 (80词优化词典)")
print("=" * 60)

# 帖子情感
post_results = []
for idx, row in posts.iterrows():
    label, neg, pos, emotions, hits = classify_sentiment(row['clean_text'])
    post_results.append({
        'sentiment': label,
        'lexicon_hits': hits,
        'neg_score': neg,
        'pos_score': pos,
        **{f'emo_{k}': v for k, v in emotions.items()}
    })

sent_df = pd.DataFrame(post_results)
posts = pd.concat([posts.reset_index(drop=True), sent_df], axis=1)

print(f"  帖子情感分布:")
print(posts['sentiment'].value_counts().to_string(header=False))
n_hit = int((posts['lexicon_hits'] > 0).sum())
print(f"  匹配到至少一个词典词的帖子 / posts matching >=1 lexicon term: "
      f"{n_hit} ({n_hit / len(posts) * 100:.1f}%)  [paper: 1,297 = 67.7%]")
for plat in posts['platform'].unique():
    sub = posts[posts['platform'] == plat]
    n = len(sub)
    neg_pct = (sub['sentiment'] == 'Negative').sum() / n * 100
    neu_pct = (sub['sentiment'] == 'Neutral').sum() / n * 100
    pos_pct = (sub['sentiment'] == 'Positive').sum() / n * 100
    print(f"  {plat}: Neg={neg_pct:.1f}% Neu={neu_pct:.1f}% Pos={pos_pct:.1f}% (n={n})")

# 评论情感
print("\n  评论情感分类...")
comment_results = []
for idx, row in included_comments.iterrows():
    label, neg, pos, emotions, hits = classify_sentiment(row['clean_text'])
    comment_results.append({'comment_sentiment': label})

csent_df = pd.DataFrame(comment_results)
included_comments = pd.concat([
    included_comments.reset_index(drop=True),
    csent_df
], axis=1)

print(f"  评论情感分布:")
print(included_comments['comment_sentiment'].value_counts().to_string(header=False))


# ==============================
# STEP 4: sklearn LDA 主题建模
# ==============================
print("\n" + "=" * 60)
print("STEP 4: sklearn LDA 主题建模")
print("=" * 60)

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# 将分词结果拼回字符串给CountVectorizer
docs_joined = [' '.join(doc) for doc in docs_tokenized]

vectorizer = CountVectorizer(
    min_df=5,
    max_df=0.5,
    token_pattern=r'(?u)\b\w+\b'
)
dtm = vectorizer.fit_transform(docs_joined)
vocab = vectorizer.get_feature_names_out()
print(f"  词典大小: {len(vocab)} 个词")
print(f"  语料大小: {dtm.shape[0]} 篇文档")

# Fit the final k = 5 model. Coherence (c_v) is computed separately, by
# 02_coherence_gensim.py, on the topic-word distributions of these same
# scikit-learn models (Supplemental Table S3, Supplemental Figure S1).
print("\n  Coherence (c_v) is computed by 02_coherence_gensim.py on the")
print("  topic-word distributions of the same sklearn models fitted here.")
print("  (Supplemental Table S3: k=5 c_v=0.4995; k=7 c_v=0.5079 is the maximum;")
print("   k=8 c_v=0.4797, i.e. LOWER than k=5.)")
print("  This run performs the topic assignment; parameters follow the Methods.")
# ==============================
# STEP 5: 训练k=5最终模型
# ==============================
print("\n" + "=" * 60)
print("STEP 5: k=5 最终模型 (sklearn)")
print("=" * 60)

lda5 = LatentDirichletAllocation(
    n_components=5,
    max_iter=20,
    learning_method='batch',
    random_state=42
)
doc_topics_matrix = lda5.fit_transform(dtm)

# 打印每个主题的关键词
print("\n  原始主题关键词:")
for i in range(5):
    top_idx = lda5.components_[i].argsort()[-15:][::-1]
    words_str = ', '.join([f"{vocab[j]}({lda5.components_[i][j]:.1f})" for j in top_idx])
    print(f"  Raw Topic {i}: {words_str}")

posts['dominant_topic_raw'] = doc_topics_matrix.argmax(axis=1)
posts['topic_prob'] = doc_topics_matrix.max(axis=1)

print("\n  原始主题分布:")
for i in range(5):
    n = (posts['dominant_topic_raw'] == i).sum()
    print(f"    Raw Topic {i}: n={n} ({n/len(posts)*100:.1f}%)")


# ==============================
# STEP 6: 主题映射到T1-T5
# ==============================
print("\n" + "=" * 60)
print("STEP 6: 主题映射")
print("=" * 60)

# Mapping of the five raw LDA topic indices onto the published labels T1-T5.
#
# The labels were assigned by two researchers who independently inspected the
# topic-word distributions and representative posts for each raw topic, and were
# reconciled through discussion (Methods, "Topic modeling"). The mapping is fixed
# here rather than inferred from a keyword heuristic, so that this script
# reproduces the published labelling exactly and the deposited code matches the
# procedure described in the manuscript.
TOPIC_MAP = {1: 'T1', 2: 'T2', 3: 'T3', 0: 'T4', 4: 'T5'}

TOPIC_NAMES = {
    'T1': 'Postoperative Recovery',
    'T2': 'Postoperative Medication and Surveillance',
    'T3': 'Living With Thyroid Cancer',
    'T4': 'Treatment Decision and Debate',
    'T5': 'Healthcare Navigation',
}

print("  Topic labelling (fixed mapping, assigned by manual inspection):")
for raw in sorted(TOPIC_MAP):
    lab = TOPIC_MAP[raw]
    n = (posts['dominant_topic_raw'] == raw).sum()
    top_idx = lda5.components_[raw].argsort()[-10:][::-1]
    top_words = ", ".join(vocab[t] for t in top_idx)
    print(f"    Raw topic {raw} -> {lab}  {TOPIC_NAMES[lab]}  (n = {n})")
    print(f"      top 10: {top_words}")

posts['topic_label'] = posts['dominant_topic_raw'].map(TOPIC_MAP)
posts['topic_name'] = posts['topic_label'].map(TOPIC_NAMES)

# Export the topic-word distributions that Figure S5 plots, so that the figure
# is always regenerated from the fitted model rather than from stored values.
_kw = []
for raw in sorted(TOPIC_MAP):
    comp = lda5.components_[raw]
    dist = comp / comp.sum()
    for rank, t in enumerate(comp.argsort()[-20:][::-1], start=1):
        _kw.append({'raw_topic': raw, 'topic_label': TOPIC_MAP[raw],
                    'topic_name': TOPIC_NAMES[TOPIC_MAP[raw]],
                    'rank': rank, 'term': vocab[t],
                    'weight': round(float(dist[t]), 6)})
pd.DataFrame(_kw).to_csv(os.path.join(OUTPUT_DIR, 'topic_keywords_top20.csv'),
                         index=False, encoding='utf-8-sig')

print("\n  最终主题分布:")
print("  " + "-" * 50)
for t in ['T1', 'T2', 'T3', 'T4', 'T5']:
    n = (posts['topic_label'] == t).sum()
    pct = n / len(posts) * 100
    print(f"  {t}: n={n} ({pct:.1f}%)")

# 平台x主题交叉表
print("\n  平台 x 主题交叉表:")
ct = pd.crosstab(posts['platform'], posts['topic_label'])
ct = ct.reindex(columns=['T1', 'T2', 'T3', 'T4', 'T5'])
print(ct.to_string())

# 与论文对比
print("\n  Compare with the published solution (Supplemental Table S11):")
paper_n = {'T1': 263, 'T2': 206, 'T3': 434, 'T4': 552, 'T5': 461}
for t in ['T1', 'T2', 'T3', 'T4', 'T5']:
    actual = (posts['topic_label'] == t).sum()
    expected = paper_n[t]
    diff = actual - expected
    pct_diff = abs(diff) / expected * 100
    status = "OK" if diff == 0 else "MISMATCH"
    print(f"  {t}: actual={actual}, paper={expected}, diff={diff:+d} [{status}]")
assert all((posts["topic_label"] == t).sum() == paper_n[t] for t in paper_n), \
    "Topic sizes do not match the published solution — check resources.STOPWORDS (must be the 172-word Table S2 list)"


# ==============================
# STEP 7: 所有统计输出
# ==============================
print("\n" + "=" * 60)
print("STEP 7: 统计结果")
print("=" * 60)

from scipy import stats

# --- 7a: Supplemental Table S11 — topic structure and sentiment profiles ---
print("\n--- Table S11: topic sentiment profiles ---")
topic_profiles = []
for t in ['T1', 'T2', 'T3', 'T4', 'T5']:
    sub = posts[posts['topic_label'] == t]
    n = len(sub)
    neg_pct = (sub['sentiment'] == 'Negative').sum() / n * 100 if n > 0 else 0
    neu_pct = (sub['sentiment'] == 'Neutral').sum() / n * 100 if n > 0 else 0
    pos_pct = (sub['sentiment'] == 'Positive').sum() / n * 100 if n > 0 else 0
    topic_profiles.append({
        'topic': t, 'n': n, 'pct': round(n / len(posts) * 100, 1),
        'neg_pct': round(neg_pct, 1), 'neu_pct': round(neu_pct, 1),
        'pos_pct': round(pos_pct, 1)
    })
    print(f"  {t}: n={n} ({n/len(posts)*100:.1f}%) | Neg={neg_pct:.1f}% Neu={neu_pct:.1f}% Pos={pos_pct:.1f}%")

pd.DataFrame(topic_profiles).to_csv(
    os.path.join(OUTPUT_DIR, 'table_s11_topic_profiles.csv'),
    index=False, encoding='utf-8-sig'
)

# --- 7b: Supplemental Table S8 — topic distribution across platforms ---
print("\n--- Table S8: platform x topic ---")
plat_map = {'小红书': 'Xiaohongshu', '微博': 'Weibo', '知乎': 'Zhihu', '抖音': 'Douyin'}

s8_data = []
for plat_cn, plat_en in plat_map.items():
    for t in ['T1', 'T2', 'T3', 'T4', 'T5']:
        n = len(posts[(posts['platform'] == plat_cn) & (posts['topic_label'] == t)])
        s8_data.append({'platform': plat_en, 'topic': t, 'n': n})

s8_df = pd.DataFrame(s8_data)
s8_pivot = s8_df.pivot(index='platform', columns='topic', values='n')
print(s8_pivot.to_string())
s8_df.to_csv(os.path.join(OUTPUT_DIR, 'table_s8_platform_topic.csv'),
             index=False, encoding='utf-8-sig')

# --- 7c: Supplemental Table S9 — sentiment chi-square tests ---
print("\n--- Table S9: chi-square tests ---")
chi2_results = []

# 帖子 vs 评论
posts_sent = posts['sentiment'].value_counts()
comm_sent = included_comments['comment_sentiment'].value_counts()
ct_content = pd.DataFrame({
    'Posts': [posts_sent.get(s, 0) for s in ['Negative', 'Neutral', 'Positive']],
    'Comments': [comm_sent.get(s, 0) for s in ['Negative', 'Neutral', 'Positive']]
}, index=['Negative', 'Neutral', 'Positive'])
chi2, p, dof, exp = stats.chi2_contingency(ct_content.T)
n_tot = ct_content.sum().sum()
v = np.sqrt(chi2 / (n_tot * (min(ct_content.T.shape) - 1)))
chi2_results.append({'comparison': 'Posts vs Comments', 'N': int(n_tot), 'df': dof,
                     'chi2': round(chi2, 3), 'p': f'{p:.6f}', 'V': round(v, 3),
                     'min_exp': round(exp.min(), 1)})
print(f"  Posts vs Comments: chi2={chi2:.3f}, p={p:.6f}, V={v:.3f}")

# 帖子: 平台x情感
ct_ps = pd.crosstab(posts['platform'], posts['sentiment'])
chi2, p, dof, exp = stats.chi2_contingency(ct_ps)
n_tot = ct_ps.sum().sum()
v = np.sqrt(chi2 / (n_tot * (min(ct_ps.shape) - 1)))
chi2_results.append({'comparison': 'Posts: Platform x Sentiment', 'N': int(n_tot), 'df': dof,
                     'chi2': round(chi2, 3), 'p': f'{p:.6f}', 'V': round(v, 3),
                     'min_exp': round(exp.min(), 1)})
print(f"  Posts Platform x Sent: chi2={chi2:.3f}, p={p:.6f}, V={v:.3f}")

# 帖子: 主题x情感
ct_ts = pd.crosstab(posts['topic_label'], posts['sentiment'])
chi2, p, dof, exp = stats.chi2_contingency(ct_ts)
n_tot = ct_ts.sum().sum()
v = np.sqrt(chi2 / (n_tot * (min(ct_ts.shape) - 1)))
chi2_results.append({'comparison': 'Posts: Topic x Sentiment', 'N': int(n_tot), 'df': dof,
                     'chi2': round(chi2, 3), 'p': f'{p:.6f}', 'V': round(v, 3),
                     'min_exp': round(exp.min(), 1)})
print(f"  Posts Topic x Sent: chi2={chi2:.3f}, p={p:.6f}, V={v:.3f}")

# 评论: 平台x情感
ct_cp = pd.crosstab(included_comments['platform'], included_comments['comment_sentiment'])
chi2, p, dof, exp = stats.chi2_contingency(ct_cp)
n_tot = ct_cp.sum().sum()
v = np.sqrt(chi2 / (n_tot * (min(ct_cp.shape) - 1)))
chi2_results.append({'comparison': 'Comments: Platform x Sentiment', 'N': int(n_tot), 'df': dof,
                     'chi2': round(chi2, 3), 'p': f'{p:.6f}', 'V': round(v, 3),
                     'min_exp': round(exp.min(), 1)})
print(f"  Comments Platform x Sent: chi2={chi2:.3f}, p={p:.6f}, V={v:.3f}")

# 评论: 主题x情感 (需要继承帖子主题)
post_topic_map = posts.set_index('post_id')['topic_label'].to_dict()
included_comments['topic_label'] = included_comments['post_id'].map(post_topic_map)
comments_with_topic = included_comments.dropna(subset=['topic_label'])

if len(comments_with_topic) > 0:
    ct_ct = pd.crosstab(comments_with_topic['topic_label'], comments_with_topic['comment_sentiment'])
    chi2, p, dof, exp = stats.chi2_contingency(ct_ct)
    n_tot = ct_ct.sum().sum()
    v = np.sqrt(chi2 / (n_tot * (min(ct_ct.shape) - 1)))
    chi2_results.append({'comparison': 'Comments: Topic x Sentiment', 'N': int(n_tot), 'df': dof,
                         'chi2': round(chi2, 3), 'p': f'{p:.6f}', 'V': round(v, 3),
                         'min_exp': round(exp.min(), 1)})
    print(f"  Comments Topic x Sent: chi2={chi2:.3f}, p={p:.6f}, V={v:.3f}")
    print(f"  (topic-linked comments: {len(comments_with_topic)})")

pd.DataFrame(chi2_results).to_csv(
    os.path.join(OUTPUT_DIR, 'table_s9_chi_square.csv'),
    index=False, encoding='utf-8-sig'
)

# --- 7d: Supplemental Table S10 — Bonferroni-corrected pairwise comparisons ---
print("\n--- Table S10: Bonferroni-corrected pairwise comparisons ---")
bonf_results = []
platform_list = ['小红书', '微博', '知乎', '抖音']

# Bonferroni is applied ONCE. The unadjusted p is compared with the adjusted
# alpha (0.05 / 6 = .00833); the Bonferroni-adjusted p (p x 6) is reported for
# transparency and is compared with .05 if used at all. Comparing an ADJUSTED p
# with an ADJUSTED alpha applies the correction twice and is wrong.
ALPHA_A = 0.05 / 6
print(f"  Panel A: platform pairs (6 comparisons, Bonferroni alpha = {ALPHA_A:.5f})")
for p1, p2 in combinations(platform_list, 2):
    sub = posts[posts['platform'].isin([p1, p2])]
    ct_pair = pd.crosstab(sub['platform'], sub['sentiment'])
    chi2, p, dof, exp = stats.chi2_contingency(ct_pair)
    n = ct_pair.sum().sum()
    v = np.sqrt(chi2 / (n * (min(ct_pair.shape) - 1)))
    bonf_p = min(p * 6, 1.0)
    sig = "Yes" if p < ALPHA_A else "n.s."   # unadjusted p vs adjusted alpha
    bonf_results.append({
        'panel': 'A: Platform', 'pair': f'{plat_map[p1]} vs {plat_map[p2]}',
        'chi2': round(chi2, 2), 'df': dof, 'p_raw': round(p, 6),
        'p_bonf': round(bonf_p, 6), 'alpha_bonf': round(ALPHA_A, 5), 'V': round(v, 3),
        'effect': 'Negligible' if v < 0.1 else 'Small' if v < 0.3 else 'Medium',
        'sig': sig, 'min_exp': round(exp.min(), 1)
    })
    print(f"    {plat_map[p1]} vs {plat_map[p2]}: chi2={chi2:.2f}, V={v:.3f}, sig={sig}")

ALPHA_B = 0.05 / 10
print(f"\n  Panel B: topic pairs (10 comparisons, Bonferroni alpha = {ALPHA_B:.5f})")
for t1, t2 in combinations(['T1', 'T2', 'T3', 'T4', 'T5'], 2):
    sub = posts[posts['topic_label'].isin([t1, t2])]
    ct_pair = pd.crosstab(sub['topic_label'], sub['sentiment'])
    chi2, p, dof, exp = stats.chi2_contingency(ct_pair)
    n = ct_pair.sum().sum()
    v = np.sqrt(chi2 / (n * (min(ct_pair.shape) - 1)))
    bonf_p = min(p * 10, 1.0)
    sig = "Yes" if p < ALPHA_B else "n.s."   # unadjusted p vs adjusted alpha
    bonf_results.append({
        'panel': 'B: Topic', 'pair': f'{t1} vs {t2}',
        'chi2': round(chi2, 2), 'df': dof, 'p_raw': round(p, 6),
        'p_bonf': round(bonf_p, 6), 'alpha_bonf': round(ALPHA_B, 5), 'V': round(v, 3),
        'effect': 'Negligible' if v < 0.1 else 'Small' if v < 0.3 else 'Medium',
        'sig': sig, 'min_exp': round(exp.min(), 1)
    })
    print(f"    {t1} vs {t2}: chi2={chi2:.2f}, V={v:.3f}, sig={sig}")

pd.DataFrame(bonf_results).to_csv(
    os.path.join(OUTPUT_DIR, 'table_s10_bonferroni.csv'),
    index=False, encoding='utf-8-sig'
)

# --- 7e: Supplemental Figure S6, Panel B — platform x topic x sentiment ---
print("\n--- Figure S6 Panel B: platform x topic x sentiment ---")
pb_data = []
for plat_cn, plat_en in plat_map.items():
    for t in ['T1', 'T2', 'T3', 'T4', 'T5']:
        sub = posts[(posts['platform'] == plat_cn) & (posts['topic_label'] == t)]
        n = len(sub)
        neg_n = (sub['sentiment'] == 'Negative').sum()
        neu_n = (sub['sentiment'] == 'Neutral').sum()
        pos_n = (sub['sentiment'] == 'Positive').sum()
        pb_data.append({
            'platform': plat_en, 'topic': t, 'n': n,
            'neg_n': neg_n, 'neg_pct': round(neg_n/n*100, 1) if n > 0 else 0,
            'neu_n': neu_n, 'neu_pct': round(neu_n/n*100, 1) if n > 0 else 0,
            'pos_n': pos_n, 'pos_pct': round(pos_n/n*100, 1) if n > 0 else 0,
        })

pd.DataFrame(pb_data).to_csv(
    os.path.join(OUTPUT_DIR, 'panel_b_posts.csv'),
    index=False, encoding='utf-8-sig'
)

# --- 7f: Supplemental Figure S3 / Table S12 — emotion category intensity ---
print("\n--- Figure S3 / Table S12: emotion category intensity ---")
emo_cats = ['惧', '哀', '怒', '恶', '惊', '乐', '好']
emo_en = {'惧': 'Fear', '哀': 'Sadness', '怒': 'Anger', '恶': 'Disgust',
          '惊': 'Surprise', '乐': 'Joy', '好': 'Like'}

heatmap_data = []
for t in ['T1', 'T2', 'T3', 'T4', 'T5']:
    sub = posts[posts['topic_label'] == t]
    n = len(sub)
    row = {'topic': t, 'n': n}
    for cat in emo_cats:
        col = f'emo_{cat}'
        if col in sub.columns:
            mean_val = sub[col].fillna(0).mean()
        else:
            mean_val = 0
        row[f'{emo_en[cat]}'] = round(mean_val, 2)
    heatmap_data.append(row)
    vals = " | ".join(f"{emo_en[c]}={row[emo_en[c]]}" for c in emo_cats)
    print(f"  {t}: {vals}")

pd.DataFrame(heatmap_data).to_csv(
    os.path.join(OUTPUT_DIR, 'figure_s3_emotion_intensity.csv'),
    index=False, encoding='utf-8-sig'
)


# ==============================
# STEP 8: Coherence数据 (来自之前的Gensim分析)
# ==============================
print("\n" + "=" * 60)
print("STEP 8: Coherence数据 (已有值，直接保存)")
print("=" * 60)
# Supplemental Table S3 / Supplemental Figure S1. Computed by
# 02_coherence_gensim.py with the gensim CoherenceModel (v4.3.2, topn = 10) on
# the topic-word distributions of the scikit-learn models that produced the
# reported assignments, so that the selection curve and the reported solution
# come from the same estimator. Coherence is flat between k = 5 and k = 7
# (range 0.008) and FALLS at k = 8; k = 5 was selected because k = 6 and k = 7
# only subdivide topics already present at k = 5 and because k = 5 yields no
# micro-topics (n < 50). k = 8 is not the maximum.
coherence_scores = {2: 0.4818, 3: 0.4892, 4: 0.4458, 5: 0.4995,
                    6: 0.5069, 7: 0.5079, 8: 0.4797, 9: 0.4996, 10: 0.4703}
coh_df = pd.DataFrame([{'k': k, 'c_v': v} for k, v in coherence_scores.items()])
coh_df.to_csv(os.path.join(OUTPUT_DIR, 'coherence_scores.csv'), index=False, encoding='utf-8-sig')
print(coh_df.to_string(index=False))


# ==============================
# STEP 9: 保存带标签的完整帖子数据
# ==============================
posts.to_csv(os.path.join(OUTPUT_DIR, 'posts_with_all_labels.csv'),
             index=False, encoding='utf-8-sig')
print(f"\n  Saved full post-level output: {len(posts)} rows "
      f"-> output/posts_with_all_labels.csv")

# posts_with_topics_FINAL.csv is the interchange file consumed by
# 03_sentiment_sensitivity.py. Its sentiment column is named `sent_opt`
# (the optimised 80-term lexicon label) to match the schema described in
# DATA.md and used by the deposited de-identified results table.
_final = posts[['platform', 'post_id', 'clean_text', 'sentiment',
                'dominant_topic_raw', 'topic_prob', 'topic_label']].copy()
_final = _final.rename(columns={'sentiment': 'sent_opt'})
_final.to_csv(os.path.join(OUTPUT_DIR, 'posts_with_topics_FINAL.csv'),
              index=False, encoding='utf-8-sig')
print(f"  Saved interchange file for script 03: {len(_final)} rows "
      f"-> output/posts_with_topics_FINAL.csv")

# De-identified post-level results table (deposited with this repository).
# No post text and no platform post identifier, so it carries no
# re-identification risk. Regenerating it here keeps the deposited copy and
# the pipeline in step.
_deid = pd.DataFrame({
    'anon_id': [f'P{i:04d}' for i in range(1, len(posts) + 1)],
    'platform': posts['platform'].map(plat_map).fillna(posts['platform']),
    'topic': posts['topic_label'],
    'topic_probability': posts['topic_prob'].round(4),
    'sentiment': posts['sentiment'],
    'char_count': posts['clean_text'].astype(str).str.len(),
})
_deid.to_csv(os.path.join(OUTPUT_DIR, 'post_level_results_deidentified.csv'),
             index=False, encoding='utf-8-sig')
print(f"  Rebuilt de-identified results table: {len(_deid)} rows "
      f"-> output/post_level_results_deidentified.csv")

# 保存评论数据
included_comments.to_csv(os.path.join(OUTPUT_DIR, 'comments_with_sentiment.csv'),
                         index=False, encoding='utf-8-sig')
print(f"  Saved comment-level output: {len(included_comments)} rows "
      f"-> output/comments_with_sentiment.csv")


# ==============================
# 完成
# ==============================
print("\n" + "=" * 60)
print("全部完成！输出文件在:", OUTPUT_DIR)
print("=" * 60)
print("""
Output files
------------
  table_s11_topic_profiles.csv          Supplemental Table S11  topic structure and sentiment profiles
  table_s8_platform_topic.csv           Supplemental Table S8   topic distribution across platforms
  table_s9_chi_square.csv               Supplemental Table S9   sentiment chi-square tests
  table_s10_bonferroni.csv              Supplemental Table S10  Bonferroni-corrected pairwise comparisons
  panel_b_posts.csv                     Supplemental Figure S6, Panel B
  figure_s3_emotion_intensity.csv       Supplemental Figure S3 / Table S12
  topic_keywords_top20.csv              Supplemental Figure S5
  coherence_scores.csv                  Supplemental Table S3 / Figure S1 (computed by 02_coherence_gensim.py)
  posts_with_all_labels.csv             full post-level output (contains post text; not for redistribution)
  posts_with_topics_FINAL.csv           interchange file consumed by 03_sentiment_sensitivity.py
  comments_with_sentiment.csv           comment-level output (contains comment text; not for redistribution)
  post_level_results_deidentified.csv   de-identified post-level results (no text, no post id)

Note: posts_with_all_labels.csv, posts_with_topics_FINAL.csv and
comments_with_sentiment.csv contain the corpus text and must not be
redistributed (see DATA.md).
""")
