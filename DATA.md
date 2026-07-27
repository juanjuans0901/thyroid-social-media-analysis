# Data notes

Raw post- and comment-level data are not distributed. The corpus consists of public,
de-identified, paraphrased social media content collected under Chongqing Medical
University ethics approval 2024056; de-identified data are available from the
corresponding author on reasonable request.

Create a `data/` folder next to the scripts and place the following files in it to run
scripts 01-03:

| File | Needed by | Notes |
|---|---|---|
| `posts_FINAL_CLEANED_v3.csv` | `01_full_pipeline.py`, `02_coherence_gensim.py` | analytic corpus, N = 1,916 |
| `comments_CONSENSUS_FINAL.csv` | `01_full_pipeline.py` | N = 26,243 included comments |
| `posts_with_topics_FINAL.csv` | `03_sentiment_sensitivity.py` | written by `01_full_pipeline.py` |

## sentiment_validation_labels.csv (included in this repository)

De-identified label table for the sentiment annotation sample behind Supplemental
Table S5, Panels A and C. One row per annotated post, 299 rows.

| Column | Meaning |
|---|---|
| `annotation_id` | 1–300; id 134 was withdrawn during coding and never re-issued, hence 299 rows |
| `platform` | Douyin 75, Zhihu 75, Xiaohongshu 75, Weibo 74 |
| `char_count` | characters in the post, used for the length-adaptive threshold |
| `coder_A_DJ`, `coder_B_DQ` | independent labels from the two coders |
| `agreed` | `Y` when the two coders matched |
| `discussion_result` | label agreed after discussion; filled only when `agreed = N` |
| `consensus_label` | coder label when they agreed, otherwise `discussion_result` |
| `in_analytic_corpus` | `Y` for the 260 posts that survived Stage-5 screening |
| `lexicon_label` | the automated label (`sent_opt`) for those 260 posts |
| `topic_label` | LDA topic for those 260 posts |

Cohen's κ between the two coders across all 299 rows is 0.729. Of the 260 posts matched
to the analytic corpus, 221 had coder agreement; those 221 form the reference standard
for Panel C. Annotation id 172 matched the same short post as id 16 and could not be
disambiguated, so it is treated as unmatched.

The file contains no post text and no platform post identifier, so it carries no
re-identification risk. The authors' internal workbook that pairs these labels with the
post text is not distributed.
