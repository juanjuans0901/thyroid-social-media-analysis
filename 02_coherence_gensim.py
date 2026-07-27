"""
02_coherence_gensim.py
Topic-model selection: coherence (c_v) scan for k = 2-10 on the analytic corpus
(Table S3, Figure S1; leave-one-platform-out sensitivity in Figure S2)
(N = 1,916 patient-authored posts), plus leave-one-platform-out sensitivity at k = 5.

Preprocessing is identical to 01_full_pipeline.py: jieba segmentation with the
65-term medical user dictionary (Table S1) and the 172-word stopword list
(Table S2), both imported from resources.py; vocabulary filtered at min_df = 5
and max_df = 0.5. Topic models are fit with scikit-learn
LatentDirichletAllocation (learning_method='batch', max_iter=20,
random_state=42), the same estimator used for the final topic assignment, and
coherence is computed with the gensim CoherenceModel.

REPRODUCIBILITY NOTE
  c_v depends on library versions. Reported values are k = 5: c_v = 0.4802
  (local peak, selected; Table S3 and Figure S1) and k = 8: c_v = 0.5237
  (global maximum). Pin gensim==4.3.2 as listed in requirements.txt.
"""

import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import jieba
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resources import MEDICAL_TERMS, STOPWORDS  # Table S1 (n = 65), Table S2 (n = 172)

assert len(STOPWORDS) == 172, "STOPWORDS must be the 172-word list in Table S2"

POSTS_FILE = "./data/posts_FINAL_CLEANED_v3.csv"  # FINAL analytic corpus, N = 1,916
OUTPUT_DIR = "./output"
RANDOM_STATE = 42
K_RANGE = range(2, 11)
TOPN = 10  # top words per topic used for c_v

os.makedirs(OUTPUT_DIR, exist_ok=True)
for term in MEDICAL_TERMS:
    jieba.add_word(term)


def tokenize_for_lda(text):
    """Segmentation identical to 01_full_pipeline.py (Methods 'Text preprocessing')."""
    text = str(text)
    cleaned = []
    for w in jieba.lcut(text):
        w = w.strip()
        if not w:
            continue
        if w in STOPWORDS:
            continue
        if len(w) == 1:
            continue
        if re.match(r"^[\d.]+$", w):
            continue
        if re.match(r"^[a-zA-Z]$", w):
            continue
        if re.match(r"^[\W]+$", w):
            continue
        cleaned.append(w)
    return cleaned


def build_matrices(texts):
    docs = [" ".join(t) for t in texts]
    vec = CountVectorizer(min_df=5, max_df=0.5, token_pattern=r"(?u)\b\w+\b")
    dtm = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    gdict = Dictionary(texts)
    return dtm, vocab, gdict


def coherence_cv(dtm, vocab, texts, gdict, k):
    """Fit sklearn LDA at k, take top words per topic, score with gensim c_v."""
    lda = LatentDirichletAllocation(
        n_components=k, learning_method="batch", max_iter=20, random_state=RANDOM_STATE
    )
    lda.fit(dtm)
    topics_words = [
        [vocab[i] for i in comp.argsort()[-TOPN:][::-1]] for comp in lda.components_
    ]
    cm = CoherenceModel(
        topics=topics_words, texts=texts, dictionary=gdict, coherence="c_v"
    )
    return cm.get_coherence()


def main():
    df = pd.read_csv(POSTS_FILE)
    print(f"Loaded {len(df)} posts (manuscript final corpus = 1,916)")
    texts = [tokenize_for_lda(t) for t in df["clean_text"]]
    dtm, vocab, gdict = build_matrices(texts)
    print(f"Vocabulary after min_df=5 / max_df=0.5 filtering: {len(vocab)} terms")

    # ---- k = 2..10 coherence scan ----
    print("\nCoherence (c_v) scan:")
    coh = {}
    for k in K_RANGE:
        coh[k] = coherence_cv(dtm, vocab, texts, gdict, k)
        print(f"  k={k:>2}: c_v = {coh[k]:.4f}")
    kmax = max(coh, key=coh.get)
    print(f"\nGlobal maximum: k={kmax} (c_v={coh[kmax]:.4f})")
    print(f"Selected solution: k=5 (c_v={coh[5]:.4f})")
    pd.DataFrame(sorted(coh.items()), columns=["k", "c_v"]).to_csv(
        os.path.join(OUTPUT_DIR, "coherence_scan_k2_10.csv"), index=False
    )

    # ---- leave-one-platform-out at k = 5 ----
    print("\nLeave-one-platform-out sensitivity (k=5):")
    full_cv = coherence_cv(dtm, vocab, texts, gdict, 5)
    rows = [("None (full model)", len(df), round(full_cv, 4))]
    print(f"  Full model (N={len(df)}): c_v={full_cv:.4f}")
    for plat in df["platform"].unique():
        keep = (df["platform"] != plat).to_numpy()
        sub_texts = [texts[i] for i in range(len(texts)) if keep[i]]
        s_dtm, s_vocab, s_gdict = build_matrices(sub_texts)
        cv = coherence_cv(s_dtm, s_vocab, sub_texts, s_gdict, 5)
        rows.append((f"Exclude {plat}", int(keep.sum()), round(cv, 4)))
        print(f"  Exclude {plat} (N={int(keep.sum())}): c_v={cv:.4f}")
    pd.DataFrame(rows, columns=["model", "N", "c_v"]).to_csv(
        os.path.join(OUTPUT_DIR, "loo_sensitivity_k5.csv"), index=False
    )
    print(f"\nSaved: {OUTPUT_DIR}/coherence_scan_k2_10.csv, {OUTPUT_DIR}/loo_sensitivity_k5.csv")


if __name__ == "__main__":
    main()
