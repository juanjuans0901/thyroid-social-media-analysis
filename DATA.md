# Data notes

Raw post- and comment-level data are not distributed. The corpus consists of public,
de-identified, paraphrased social media content collected under Chongqing Medical
University ethics approval 2024056; de-identified data are available from the
corresponding author on reasonable request.

Three de-identified files **are** deposited in this repository and can be shared openly.
None of them contains post text or a platform post identifier, so none carries any
re-identification risk:

| File | What it supports |
|---|---|
| `sentiment_validation_labels.csv` | Supplemental Table S5, Panels A and C |
| `post_level_results_deidentified.csv` | every post-level topic × platform × sentiment analysis in the paper |
| `coding_manual.csv` | Supplemental Table S7 (directed content analysis codebook) |

## Restricted inputs

Create a `data/` folder next to the scripts and place the following files in it to run
scripts 01 and 02:

| File | Needed by | Notes |
|---|---|---|
| `posts_FINAL_CLEANED_v3.csv` | `01_full_pipeline.py`, `02_coherence_gensim.py` | analytic corpus, N = 1,916 |
| `comments_CONSENSUS_FINAL.csv` | `01_full_pipeline.py` | N = 26,243 included comments |

`03_sentiment_sensitivity.py` reads `output/posts_with_topics_FINAL.csv`, which
`01_full_pipeline.py` writes, so run script 01 first. Everything `01_full_pipeline.py`
writes to `output/` whose name contains post or comment text —
`posts_with_all_labels.csv`, `posts_with_topics_FINAL.csv` and
`comments_with_sentiment.csv` — must not be redistributed. `output/` is git-ignored for
that reason.

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

Cohen's κ between the two coders across all 299 rows is 0.729, with 252 of 299 (84.3%)
raw agreement; the 47 disagreements were resolved through discussion.

Of the 299 annotated posts, 260 are matched to the analytic corpus. The 39 that are not
break down as follows: 38 were sampled before the final eligibility review and were
subsequently excluded at Stage 5 of the screening pipeline, and one (annotation id 172)
matched the same short post as id 16 and could not be disambiguated, so it is treated as
unmatched. Of the 260 matched posts, 221 had coder agreement without discussion; those
221 form the reference standard for Panel C.

The file contains no post text and no platform post identifier, so it carries no
re-identification risk. The authors' internal workbook that pairs these labels with the
post text is not distributed.

## post_level_results_deidentified.csv (included in this repository)

De-identified post-level results for the complete analytic corpus. One row per post,
1,916 rows, in the same order as `posts_FINAL_CLEANED_v3.csv`.

| Column | Meaning |
|---|---|
| `anon_id` | sequential anonymous identifier, `P0001`–`P1916`; not derivable from and not linkable to any platform identifier |
| `platform` | `Xiaohongshu`, `Weibo`, `Zhihu` or `Douyin` |
| `topic` | assigned LDA topic, `T1`–`T5` (dominant topic; see the topic label table in `README.md`) |
| `topic_probability` | dominant-topic probability, 4 decimal places; 1,257 posts exceed 0.60 in the final model |
| `sentiment` | three-class polarity from the optimised 80-term lexicon (`sent_opt`): `Negative`, `Neutral` or `Positive` |
| `char_count` | characters in the cleaned post, which determines the length-adaptive threshold |

This file reproduces, without access to the corpus: the topic sizes (263 / 206 / 434 /
552 / 461), the platform × topic distribution (Supplemental Table S8), every sentiment
χ² test and Cramér's V (Supplemental Table S9), all sixteen Bonferroni-corrected pairwise
comparisons (Supplemental Table S10), the topic sentiment profiles (Supplemental Table
S11 and Figure 4 Panel C) and the platform × topic × sentiment breakdown (Supplemental
Figure S6). It does not reproduce the emotion-category scores or the coherence scan,
which need the text.

It contains no post text, no platform post identifier, no author identifier, no URL and
no timestamp. The longest value in any cell is 11 characters. `01_full_pipeline.py`
regenerates it to `output/` on every run, so the deposited copy and the pipeline stay in
step.

## coding_manual.csv (included in this repository)

The complete coding manual for the directed content analysis, and the source of
Supplemental Table S7. One row per open-code-within-category, 70 rows covering five
content domains, 25 categories and 69 distinct open codes. One open code
(`外科手术经历叙述`, accounts of surgery) is deliberately cross-listed under two
categories — as evidence used in a decision under D2-4, and as personal experience under
D5-1 — which is why there are 70 rows and 69 codes.

| Column | Meaning |
|---|---|
| `Domain` | content domain D1–D5, with the number of coded posts in that domain |
| `Category ID` | `D1-1` … `D5-5` |
| `Category` | category name as published |
| `Category definition` | the category definition the coders applied |
| `Category prominence n (%)` | coded posts in the domain to which the category was applied, and the percentage of that domain |
| `CSM dimension` | Common Sense Model dimension the category maps to (Figure 4, Panel D) |
| `Open code (Chinese)` | the open code as recorded during coding |
| `Open code (English)` | English gloss |
| `Open code status` | `in initial coding manual`, or `added during coding` for the four codes that were introduced while coding was under way |
| `Open code definition (CN)` | the definition applied to that open code |
| `Inclusion criteria (CN)` | what the code does and does not cover |
| `Key keywords (CN)` | the indicative terms used when applying the code |

Sixty-five of the sixty-nine open codes were specified before coding began. Four
(`手术相关经历叙述`, `医疗纠纷与法律`, `甲状腺相关信息一般性描述`, `医疗系统相关内容`)
were introduced during coding, as directed content analysis allows, and are flagged in the
`Open code status` column.

Coding covered 1,300 high-loading posts read during the analysis phase, of which 1,284
were coded after excluding 16 institutional or marketing accounts. That set was assembled
during the analysis phase, before the final topic solution was fitted, and does not
correspond exactly to the 1,257 posts above a dominant-topic probability of 0.60 in
`post_level_results_deidentified.csv`. Two coders
coded independently; mean Cohen's κ across categories was 0.722 and disagreements were
resolved by discussion. Percentages are of coded posts **within a domain**, so they do
not sum to 100: a post can carry more than one category.

Domain identifiers D1–D5 are independent of topic identifiers T1–T5. The coding frame was
developed alongside rather than after the final topic solution, so the domains are not
nested within the topics and D*n* does not correspond to T*n*.

The workbook that pairs these codes with the post text is not distributed.
