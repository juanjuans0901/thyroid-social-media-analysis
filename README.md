# What Patients With Thyroid Nodules or Cancer Discuss on Chinese Social Media

Analysis code for *"What Patients With Thyroid Nodules or Cancer Discuss on Chinese Social
Media: Multi-Platform Mixed-Methods Analysis"* (Manuscript ID DHJ-26-0922, *Digital Health*).

Everything here has been checked against the analytic corpus. The three numbers a
reproducer is most likely to check all come out exactly:

| Check | Script | Result |
|---|---|---|
| Topic sizes T1–T5 = 263 / 206 / 434 / 552 / 461 | `01_full_pipeline.py` | 1,916 / 1,916 posts assigned identically |
| Sentiment labels (`sent_opt`) | `01_full_pipeline.py` | 1,916 / 1,916 identical; 1,297 posts (67.7%) match ≥1 lexicon term |
| Table S5 Panel C (accuracy 58.4%, macro F1 0.503, κ 0.284) | `04_lexicon_validation.py` | reproduced to three decimals, with assertions |

## Files

**Shared resources**

- `resources.py` — single source of truth for the domain word lists: `MEDICAL_TERMS`
  (n = 65, Table S1), `STOPWORDS` (n = 172, Table S2), `SENTIMENT_LEXICON`
  (n = 80, Table S4), `EMOTION_CATEGORIES` (7 DUTIR categories), and the sentiment
  scoring parameters (`NEGATION_WINDOW_CHARS`, `LENGTH_THRESHOLDS`).

**Pipeline**

- `01_full_pipeline.py` — jieba segmentation → lexicon sentiment classification and
  emotion scoring → scikit-learn LDA (k = 5) → topic labelling → the χ² / Bonferroni /
  topic-profile statistics (Tables S8–S11) and the emotion data behind Figure S3 and
  Table S12. Writes `posts_with_topics_FINAL.csv`, the input for scripts 03 and 04.
- `02_coherence_gensim.py` — coherence (c_v) scan for k = 2–10 and leave-one-platform-out
  sensitivity at k = 5 (Table S3, Figures S1 and S2).
- `03_sentiment_sensitivity.py` — Monte Carlo sentiment-misclassification sensitivity
  analysis (Table S6).
- `04_lexicon_validation.py` — sentiment-lexicon validation against the human consensus
  labels (Table S5, Panels A and C). Runs on `sentiment_validation_labels.csv`, which is
  included here, so it needs no other input.

**Figures** — plotted values are inline; each script runs standalone and writes to
an `output/` folder that is created automatically.

- `fig2_topic_distribution.py` — Figure 2
- `figS3_emotion.py` — Supplementary Figure S3 (emotion category intensity)
- `figS4_k_comparison.py` — Supplementary Figure S4 (k = 5 vs k = 8 topic sizes)
- `figS5_keywords.py` — Supplementary Figure S5 (top keywords per topic)
- `figS6_sentiment.py` — Supplementary Figure S6 (sentiment by platform and topic)

Descriptions of the annotation label table are in `DATA.md`.

## How sentiment is scored

This is the part most likely to be re-implemented differently, so it is spelled out.
Lexicon terms are matched against the cleaned text **as substrings, before
segmentation**, so that negation cues survive; a term's polarity is reversed when one of
the nine negation words appears in the **four characters** immediately preceding it
(roughly a two-word window in Chinese); and the net score is compared with a
**length-adaptive threshold** — greater than 2 for texts over 500 characters, greater
than 1 for 100–500 characters, greater than 0 for shorter texts. Emotion-category scores
are separate: raw intensity-weighted term frequencies with no negation handling, as
described in the Note on Emotion Category Scoring in the Methods.

Segmentation-based scoring, or a fixed zero threshold, will not reproduce `sent_opt`.

## Requirements

```
pip install -r requirements.txt
```

Python 3.11.9. `gensim==4.3.2` and `jieba==0.42.1` are pinned because c_v and the
segmentation are version-sensitive. The figure scripts set Latin text in a
metric-compatible Arial substitute so that the figures match Figure 1 and the manuscript
body text; for Chinese glyphs, place a Simplified-Chinese font named
`NotoSansSC-Regular.otf` (e.g. Noto Sans SC) next to the scripts.

## How to run

All files sit in the repository root. Scripts 01-03 read the corpus from a `data/`
folder; create that folder next to the scripts and place the input files listed below in
it. Outputs are written to `output/`, created automatically. Run e.g.
`python 01_full_pipeline.py`. `04_lexicon_validation.py` needs no extra input — it reads
`sentiment_validation_labels.csv` from this repository.

| Input file | Used by | Key columns | Included here |
|---|---|---|---|
| `posts_FINAL_CLEANED_v3.csv` | `01`, `02` | `clean_text`, `platform`, `post_id` | no — on request |
| `comments_CONSENSUS_FINAL.csv` | `01` | `clean_text`, `platform`, `post_id`, `comment_decision` | no — on request |
| `posts_with_topics_FINAL.csv` | `03` | `sent_opt`, `topic_label`, `platform` | produced by `01` |
| `sentiment_validation_labels.csv` | `04` | coder labels, consensus, lexicon and topic labels | **yes — included in this repository** |

Platform values in the raw data are Chinese (`小红书`, `微博`, `知乎`, `抖音`) and are
mapped to English (`Xiaohongshu`, `Weibo`, `Zhihu`, `Douyin`) inside the scripts.

## Data availability

Raw post- and comment-level data are **not** included, to protect user privacy (public,
de-identified, paraphrased content; Chongqing Medical University ethics approval 2024056).
De-identified data are available from the corresponding author on reasonable request.

`sentiment_validation_labels.csv` is the one exception: it holds only the annotation
identifier, platform, character count and the sentiment labels, with no post text and no
platform post identifier, so it can be shared openly and lets anyone verify Table S5
Panels A and C without access to the corpus.

## Notes for reviewers and reproducers

- Topic **assignment** uses scikit-learn LDA (`learning_method='batch'`, `max_iter=20`,
  `random_state=42`, alpha = beta = 1/k); topic **coherence** (c_v) uses the gensim
  CoherenceModel, as described in the Methods. All results correspond to the final
  1,916-post analytic corpus.
- `01_full_pipeline.py` asserts the published topic sizes and
  `04_lexicon_validation.py` asserts every Table S5 Panel C value, so a substituted
  word list or a changed parameter fails loudly rather than silently producing a
  different answer.
- Reported coherence values are k = 5: c_v = 0.4802 (selected) and k = 8: c_v = 0.5237
  (global maximum).

## Citation

Deng J, Hu D, He D, Xiang H, Deng Q. *What Patients With Thyroid Nodules or Cancer
Discuss on Chinese Social Media: Multi-Platform Mixed-Methods Analysis.* Digital Health
(under review; DHJ-26-0922).

## License

Released under the MIT License (see `LICENSE`).
