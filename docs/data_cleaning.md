# Data cleaning, step by step

**What this file is:** the written record of every data-preparation decision, the reason for it,
and the function that implements it. The numbers are NOT filled in here because they come from
your copy of the Kaggle file: run `python -m src.data_quality --csv data/raw/accepted_2007_to_2018Q4.csv --out outputs/tables`
and paste the resulting attrition table into the "Your numbers" section at the bottom.

## 1. Source

LendingClub **accepted loans**, 2007 to 2018Q4, downloaded from Kaggle (on the order of two
million rows and about 150 columns; the exact counts depend on the Kaggle version, so take them
from the attrition table below, not from here). Only 29 columns are read (`USECOLS` in `src/config.py`), all of them known at
origination. Post-origination fields (payments received, recoveries, last credit pull, etc.) are
never loaded, because using them would leak the outcome.

The related paper (Aste and a co-author, UCL, arXiv 1907.01800) used the same Kaggle source and
similar filtering (resolved loans only, recent vintages cut because of censoring); see the README
for what was borrowed from it and what was not.

## 2. Row-level steps (before splitting; they learn nothing from the data)

| # | Step | Why | Code |
|---|------|-----|------|
| 1 | Keep only resolved loans: `Fully Paid` = 0; `Charged Off`, `Default` and the two "does not meet the credit policy" variants map to 0/1 accordingly. Drop `Current`, `Late`, `In Grace Period`. | An open loan has no final outcome, so it cannot be labelled. | `clean_target` |
| 2 | Stratified sample (default 200 000 rows), preserving the default rate. | Keeps development fast; set `sample_size=None` for the full run. | `stratified_sample` |
| 3 | Drop `emp_title`, `title`, `grade`. | Free text with thousands of levels; `grade` is redundant with `sub_grade`. | `engineer_features` |
| 4 | Optionally drop `int_rate` and `sub_grade` (**default for the headline run**). | LendingClub's own underwriting outputs. A lender building its own model would not have them a priori. `--with-lender-score` / `--both-feature-sets` keep them. | `engineer_features(drop_lender_assigned=True)` |
| 5 | Parse `term` (" 36 months" -> 36), `emp_length` ("< 1 year" -> 0, "10+ years" -> 10), `issue_d`, `earliest_cr_line` -> year and month. | Text to numbers. Unparseable dates become missing; rows with no issue date are removed. | `engineer_features` |
| 6 | Combine `fico_range_low` and `fico_range_high` into one `fico_score` (midpoint). | LendingClub publishes FICO as a 5-point band; the midpoint is the usable score. | `engineer_features` |
| 7 | Merge `home_ownership` levels `ANY` and `NONE` into `OTHER`. | Tiny levels. | `engineer_features` |
| 8 | Keep loans issued up to `max_issue_year` (default 2016). | Loans have 36 or 60 month terms, so recent vintages are **right-censored**: many have not had time to default, and early repayments are over-represented. This biases the test set. The year is configurable; check `default_rate_by_year` for where your data starts to fall. | `filter_vintages` |

## 3. Split

Chronological, not random: train on earlier origination months, test on the latest 20 %. The
cut-off is a whole month so no month appears on both sides. A random split would let future loans
help predict past ones. Function: `temporal_split`.

## 4. Steps that learn from data (fitted on training rows only)

These run inside the sklearn pipeline, so they are refit on whatever rows a model is trained on
and can never see the test set (checked by a unit test).

| Step | Detail | Code |
|------|--------|------|
| Sparse-feature filter | Drop features observed for fewer than 70 % of **training** loans, from both train and test. | `find_sparse_features` |
| Imputation | Median for numerics, most frequent level for categoricals. | `build_preprocessor` |
| Log transform | `log(1 + x)` on `annual_inc` and `revol_bal` (very right-skewed) before scaling. The raw columns stay in the data so stress tests can still shock them multiplicatively. | `build_preprocessor` |
| Scaling | Standardise numerics to zero mean, unit variance. | `build_preprocessor` |
| Encoding | One-hot; levels with fewer than 25 training rows pooled into an "infrequent" level; unseen levels in test handled gracefully. | `build_preprocessor` |

## 5. Things to watch (not fixed by cleaning)

* **Selection bias.** Only accepted loans have outcomes. Rejected applicants are never observed, so
  results describe the accepted population, not all applicants (reject inference is out of scope).
* **Residual censoring.** Even with the year cut-off, 60-month loans from 2014 to 2016 are not fully
  resolved by the end of 2018. Treat late-vintage default rates with care.
* **Class imbalance.** Defaults are roughly 15 to 20 % of resolved loans. No re-weighting is applied
  by default; if you use `--class-weight balanced`, recalibrate before interpreting probabilities.
* **Policy changes.** LendingClub changed its underwriting over time, so a model trained on early
  years may drift on later ones (see the PSI and backtest sections).

## 6. Your numbers

Paste the output of `python -m src.data_quality` here after running it on the real file:

| Step | Rows | Rows removed | Default rate |
|------|------|--------------|--------------|
| *(fill in)* | | | |
