# What Patients With Thyroid Nodules or Cancer Discuss on Chinese Social Media

Analysis code and de-identified results for *"What Patients With Thyroid Nodules or Cancer
Discuss on Chinese Social Media: Multi-Platform Mixed-Methods Analysis"* (Manuscript ID
DHJ-26-0922, *Digital Health*).

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
  topic-profile statistics (Supplemental Tables S8–S11) and the emotion data behind
  Supplemental Figure S3 and Table S12. Writes `output/posts_with_topics_FINAL.csv`,
  the input for script 03.
- `02_coherence_gensim.py` — coherence (c_v) scan for k = 2–10 and leave-one-platform-out
  sensitivity at k = 5 (Supplemental Table S3, Figures S1 and S2).
- `03_sentiment_sensitivity.py` — Monte Carlo sentiment-misclassification sensitivity
  analysis (Supplemental Table S6).
- `04_lexicon_validation.py` — sentiment-lexicon validation against the human consensus
  labels (Supplemental Table S5, Panels A and C). Runs on `sentiment_validation_labels.csv`,
  which is included here, so it needs no other input.

**Deposited data and documentation** — column dictionaries are in `DATA.md`.

- `sentiment_validation_labels.csv` — de-identified label table for the sentiment
  annotation sample (299 rows) behind Supplemental Table S5, Panels A and C.
- `post_level_results_deidentified.csv` — de-identified post-level results for the full
  analytic corpus (1,916 rows): anonymised id, platform, assigned topic, dominant-topic
  probability, sentiment label and character count. No post text and no platform post
  identifier.
- `coding_manual.csv` — the full directed content analysis coding manual behind
  Supplemental Table S7: five content domains, 25 categories, 69 open codes with
  definitions, and the CSM dimension each category maps to.

**Figures** — each script runs standalone and writes to an `output/` folder that is
created automatically. Plotted values are held inline so that the scripts run without the
restricted corpus; every one of them is produced by `01_full_pipeline.py`, which writes
the corresponding CSV to `output/` on each run. If a figure value and the pipeline output
ever disagree, the pipeline output is authoritative.

- `fig2_topic_distribution.py` — Figure 2 (topic distribution)
- `fig3_platform_heatmap.py` — Figure 3 (cross-platform topic distribution heatmap;
  values exported by `01_full_pipeline.py` to `output/table_s8_platform_topic.csv`)
- `fig4_thematic_landscape.py` — Figure 4 (topics; the 25 categories in five content
  domains; sentiment profile; CSM mapping)
- `figS3_emotion.py` — Supplementary Figure S3 (emotion category intensity)
- `figS4_k_comparison.py` — Supplementary Figure S4 (k = 5 vs k = 8 topic sizes)
- `figS5_keywords.py` — Supplementary Figure S5 (top keywords per topic; values exported by
  `01_full_pipeline.py` to `output/topic_keywords_top20.csv`)
- `figS6_sentiment.py` — Supplementary Figure S6 (sentiment by platform and topic)

## Topic labels

The five raw LDA topic indices are mapped onto the published labels by a fixed dictionary
in `01_full_pipeline.py`:

| Raw topic | Label | Name |
|---|---|---|
| 1 | T1 | Postoperative Recovery |
| 2 | T2 | Postoperative Medication and Surveillance |
| 3 | T3 | Living With Thyroid Cancer |
| 0 | T4 | Treatment Decision and Debate |
| 4 | T5 | Healthcare Navigation |

The labels were assigned by two researchers who independently inspected the topic-word
distributions and representative posts for each raw topic, and were reconciled through
discussion (Methods, "Topic modeling"). The mapping is stated explicitly rather than
inferred at run time, so that the deposited code reproduces the published labelling
exactly and matches the procedure described in the manuscript. `01_full_pipeline.py`
prints the top ten terms of each raw topic alongside its label on every run, so the
labelling can be inspected directly.

The topic identifiers T1–T5 and the content-domain identifiers D1–D5 used in Figure 4 and
Supplemental Table S7 are **independent**. The coding frame was developed alongside rather
than after the topic solution, so the domains are not nested within the topics and D*n*
does not correspond to T*n*.

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

## Multiple comparisons

Bonferroni correction is applied **once**. Each unadjusted *p* is compared with the
adjusted alpha (.05 / 6 = .00833 for the six platform pairs in Supplemental Table S10
Panel A; .05 / 10 = .005 for the ten topic pairs in Panel B). `01_full_pipeline.py` also
exports the Bonferroni-adjusted *p* (*p* × *m*) for transparency; that column must be
compared with .05, never with the adjusted alpha. Comparing an adjusted *p* with an
adjusted alpha applies the correction twice and understates the number of significant
comparisons.

## Requirements

```
pip install -r requirements.txt
```

Python 3.11.9. `gensim==4.3.2` and `jieba==0.42.1` are pinned because c_v and the
segmentation are version-sensitive. `matplotlib>=3.6` is required by the figure scripts,
which rely on per-glyph font fallback.

### Fonts

The figure scripts set Latin text in a metric-compatible Arial substitute so that the
figures match Figure 1 and the manuscript body text. **`figS3_emotion.py` and
`figS5_keywords.py` also render Chinese glyphs and require a Simplified-Chinese font file
named `NotoSansSC-Regular.otf` placed next to the scripts** (Noto Sans SC, SIL Open Font
License, <https://fonts.google.com/noto/specimen/Noto+Sans+SC>). The font is not committed
here for licensing tidiness; those two scripts stop with an explanatory error if it is
absent rather than falling back silently, because a silent fallback is exactly what
produces a figure with the Chinese characters missing. All scripts set
`pdf.fonttype = 42` so that every face is embedded in the exported PDF — verify with
`pdffonts`, where every listed font must show `emb = yes`.

## How to run

All files sit in the repository root. Scripts 01 and 02 read the corpus from a `data/`
folder; create that folder next to the scripts and place the input files listed below in
it. Outputs are written to `output/`, created automatically. Run e.g.
`python 01_full_pipeline.py`. Script 03 reads `output/posts_with_topics_FINAL.csv`, which
script 01 writes, so run 01 first. `04_lexicon_validation.py` needs no extra input — it
reads `sentiment_validation_labels.csv` from this repository.

| Input file | Used by | Key columns | Included here |
|---|---|---|---|
| `posts_FINAL_CLEANED_v3.csv` | `01`, `02` | `clean_text`, `platform`, `post_id` | no — on request |
| `comments_CONSENSUS_FINAL.csv` | `01` | `clean_text`, `platform`, `post_id`, `comment_decision` | no — on request |
| `posts_with_topics_FINAL.csv` | `03` | `sent_opt`, `topic_label`, `platform` | written to `output/` by `01` |
| `sentiment_validation_labels.csv` | `04` | coder labels, consensus, lexicon and topic labels | **yes — included in this repository** |

Platform values in the raw data are Chinese (`小红书`, `微博`, `知乎`, `抖音`) and are
mapped to English (`Xiaohongshu`, `Weibo`, `Zhihu`, `Douyin`) inside the scripts.

## Data availability

Raw post- and comment-level data are **not** included, to protect user privacy (public,
de-identified, paraphrased content; Chongqing Medical University ethics approval 2024056).
De-identified data are available from the corresponding author on reasonable request.

Three de-identified files are deposited here and can be shared openly, because none of
them contains post text or a platform post identifier and none therefore carries any
re-identification risk:

- `sentiment_validation_labels.csv` — lets anyone verify Supplemental Table S5 Panels A
  and C without access to the corpus.
- `post_level_results_deidentified.csv` — lets anyone reproduce every topic × platform ×
  sentiment cross-tabulation, χ² test, Cramér's V and Bonferroni-corrected pairwise
  comparison reported in the paper without access to the corpus.
- `coding_manual.csv` — the complete coding manual for the directed content analysis.

The files `posts_with_all_labels.csv`, `posts_with_topics_FINAL.csv` and
`comments_with_sentiment.csv` that `01_full_pipeline.py` writes to `output/` do contain
the corpus text and must not be redistributed. `output/` is git-ignored for that reason.

## Notes for reviewers and reproducers

- Topic **assignment** uses scikit-learn LDA (`learning_method='batch'`, `max_iter=20`,
  `random_state=42`, alpha = beta = 1/k); topic **coherence** (c_v) uses the gensim
  CoherenceModel on the topic-word distributions of those same scikit-learn models, as
  described in the Methods. All results correspond to the final 1,916-post analytic
  corpus.
- `01_full_pipeline.py` asserts the published topic sizes and
  `04_lexicon_validation.py` asserts every Supplemental Table S5 Panel C value, so a
  substituted word list or a changed parameter fails loudly rather than silently producing
  a different answer.
- Coherence values (Supplemental Table S3, Supplemental Figure S1):

  | k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
  |---|---|---|---|---|---|---|---|---|---|
  | c_v | 0.4818 | 0.4892 | 0.4458 | **0.4995** | 0.5069 | 0.5079 | 0.4797 | 0.4996 | 0.4703 |

  Coherence is essentially flat between k = 5 and k = 7 (a range of 0.008), so it does not
  discriminate among those solutions; the maximum is at k = 7 and **k = 8 is lower than
  k = 5**. k = 5 was selected because k = 6 and k = 7 subdivide topics already present at
  k = 5 rather than identifying additional content, and because k = 5 yields no
  micro-topics (n < 50).
- At k = 8 the same estimator yields topic sizes 501 / 418 / 336 / 245 / 184 / 169 / 42 / 21,
  including two micro-topics (n < 50). These are the values plotted in Supplementary
  Figure S4.
- Leave-one-platform-out at k = 5 (Supplemental Figure S2): full model c_v = 0.4995;
  excluding Douyin 0.5054, Weibo 0.5030, Zhihu 0.5476, Xiaohongshu 0.5218.

## Citation

Deng J, Hu D, He D, Xiang H, Deng Q. *What Patients With Thyroid Nodules or Cancer
Discuss on Chinese Social Media: Multi-Platform Mixed-Methods Analysis.* Digital Health
(under review; DHJ-26-0922).

## License

Released under the MIT License (see `LICENSE`).
